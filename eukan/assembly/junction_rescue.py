"""Soft-clip-driven splice junction rescue.

Companion to :mod:`eukan.assembly.bam_diagnostic`. STAR's mapper is
biased toward canonical GT/AG donors, so non-canonical introns get
overshot — the read aligns past the true donor and the unconsumed
exon-2 sequence comes back as a soft-clipped tail. Every such "lost"
read carries the answer in its soft-clip: search the nearby genome
window for a copy of the clip's anchor-end consensus and the landing
position pins the missing exon-2 boundary.

This module operates on the BAM-orientation per-locus state that
:func:`eukan.assembly.bam_diagnostic.diagnose_bam` builds when called
with ``rescue_junctions=True``. The rescue is opt-in and orthogonal to
the trans-splicing / non-canonical-splice classification.

Output: per-junction aggregates (one row per ``(chrom, intron_start,
intron_end, strand)`` deduped across supporting loci), written as a
rich TSV for inspection and optionally as a 4-column STAR
``--sjdbFileChrStartEnd``-format file for a Stage-2 remap.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from eukan.assembly.bam_diagnostic import (
    _cluster_consensus,
    _cluster_key,
    _reverse_complement,
)

if TYPE_CHECKING:
    from eukan.assembly.bam_diagnostic import (
        BamLocusData,
        LocusConsistencyStats,
    )

_ALPHABET = "ACGT"

# Canonical + semi-canonical splice site dinucleotide pairs (donor-acceptor)
# in genome + strand orientation. Includes both forward and reverse-strand
# canonicals so callers don't have to infer transcribed strand a priori.
_DEFAULT_DINUC_ALLOWLIST: tuple[str, ...] = (
    "GT-AG", "CT-AC",        # canonical (+ / -)
    "GC-AG", "CT-GC",        # semi-canonical
    "AT-AC", "GT-AT",        # minor U12 spliceosome
)

# Per-pair strand inference. Dinucleotide pairs that imply forward-strand
# splicing (donor on + strand) → "+"; reverse-strand splicing → "-".
_STRAND_BY_DINUC: dict[str, str] = {
    "GT-AG": "+", "GC-AG": "+", "AT-AC": "+",
    "CT-AC": "-", "CT-GC": "-", "GT-AT": "-",
}

# Threshold (n_loci) above which a soft-clip cluster motif is treated as
# trans-splicing-derived and excluded from the rescue search. Matches the
# MODERATE-call threshold in bam_diagnostic.compute_verdict.
_TRANS_SPLICING_NLOCI_THRESHOLD = 100

# How many extra bp past the Q-bp seed match an outward extension can
# walk before stopping; bounds work on degenerate windows.
_OUTWARD_MAX = 60

# Margin (STAR over-extension) range to consider.
_MARGIN_RANGE = range(0, 6)

# Threshold for adopting a non-canonical dinucleotide pair from the STAR
# splice-site summary into the allowlist. A pair must clear BOTH a
# minimum total-fraction (so the long tail of noise pairs is excluded
# on heavily non-canonical organisms) and an absolute count floor
# (so small datasets aren't gutted by the fraction filter).
_STAR_SUMMARY_MIN_FRACTION = 0.01  # 1% of all STAR-called introns
_STAR_SUMMARY_MIN_UNIQUE_READS = 50


@dataclass
class JunctionCandidate:
    """One rescue candidate for a single locus."""

    chrom: str
    intron_start: int  # 0-based inclusive
    intron_end: int    # 0-based exclusive
    strand: str        # "+", "-", or "."
    donor: str         # 2 bases at intron 5' end
    acceptor: str      # 2 bases at intron 3' end
    score: float
    outward_match: int  # bp matched outward past the Q-bp seed
    margin: int         # STAR over-extension into the intron (0..5)
    n_reads: int        # n_clips at the contributing BAM-orient locus
    example_consensus: str


@dataclass
class JunctionRecord:
    """Per-junction aggregate across supporting loci."""

    chrom: str
    intron_start: int
    intron_end: int
    strand: str
    donor: str
    acceptor: str
    n_loci: int
    n_reads: int
    max_outward_match: int
    margin: int
    was_in_star_sj: bool
    example_consensus: str


@dataclass
class JunctionRescueResult:
    """Aggregated rescue output for one BAM walk."""

    n_loci_attempted: int = 0
    n_loci_rescued: int = 0
    n_junctions_unique: int = 0
    n_junctions_novel_vs_star: int = 0
    n_junctions_emitted_sj: int = 0
    rescue_rate_pct: float = 0.0
    dinuc_allowlist: list[str] = field(default_factory=list)
    records: list[JunctionRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rescue_junctions_from_diagnostic(
    *,
    bam_locus_data: dict[tuple[str, int, str], BamLocusData],
    locus_consistency: LocusConsistencyStats,
    top_clusters: list[tuple[str, int, int]],
    genome_path: Path,
    cluster_key_len: int,
    consensus_min_majority_fraction: float,
    min_intron_len: int,
    max_intron_len: int,
    min_locus_depth: int,
    min_seed_extension: int,
    dinuc_allowlist: list[str] | None,
    star_sj_path: Path | None,
    by_dinucleotide: dict[str, int] | None,
) -> JunctionRescueResult:
    """Walk per-(chrom, anchor_pos_bam, bam_side) loci and call rescued junctions.

    For each candidate locus, build a column-majority consensus over the
    BAM-orientation clip samples, search the upstream / downstream
    genome window for the consensus's anchor-end seed, validate hits
    with the dinucleotide allowlist + margin check, and aggregate
    surviving per-locus candidates into per-junction records.
    """
    from eukan.infra.genome import ContigIndex

    effective_allowlist = _resolve_allowlist(dinuc_allowlist, by_dinucleotide)
    allowlist_set = set(effective_allowlist)

    # Exclusion set: K-bp motifs of large soft-clip clusters (likely
    # trans-splicing). Match against either orientation to handle
    # forward/reverse-strand contributing reads.
    trans_motifs = {
        seed for seed, n_loci, _ in top_clusters
        if n_loci >= _TRANS_SPLICING_NLOCI_THRESHOLD
    }
    trans_motifs_with_rc = trans_motifs | {_reverse_complement(s) for s in trans_motifs}

    star_sj_set = (
        load_star_sj_set(star_sj_path) if star_sj_path is not None else set()
    )

    per_locus_candidates: list[JunctionCandidate] = []
    n_loci_attempted = 0
    n_loci_rescued = 0

    with ContigIndex(genome_path) as contigs:
        for bam_locus, bld in bam_locus_data.items():
            if bld.n_clips < min_locus_depth:
                continue
            if not bld.clip_samples:
                continue
            chrom, anchor_pos, bam_side = bam_locus
            # Build consensus over the clip-sample reservoir.
            consensus = _cluster_consensus(
                bld.clip_samples,
                "5p" if bam_side == "left" else "3p",
                min_coverage=max(1, min_locus_depth // 2),
                min_majority_fraction=consensus_min_majority_fraction,
            )
            if len(consensus) < cluster_key_len:
                continue
            # Trans-splicing exclusion: drop loci whose anchor key matches
            # a high-share cluster motif (in either orientation).
            anchor_key = _cluster_key(
                "5p" if bam_side == "left" else "3p",
                consensus,
                cluster_key_len,
            )
            if anchor_key in trans_motifs_with_rc:
                continue

            adj_consensus = _cluster_consensus(
                bld.anchor_adjacent_samples,
                "5p" if bam_side == "left" else "3p",
                min_coverage=max(1, min_locus_depth // 2),
                min_majority_fraction=consensus_min_majority_fraction,
            )

            n_loci_attempted += 1
            candidate = rescue_junction_for_locus(
                chrom=chrom,
                anchor_pos=anchor_pos,
                bam_side=bam_side,
                clip_consensus=consensus,
                anchor_adjacent_consensus=adj_consensus,
                contigs=contigs,
                min_intron_len=min_intron_len,
                max_intron_len=max_intron_len,
                min_seed_extension=min_seed_extension,
                dinuc_allowlist=allowlist_set,
                n_reads=bld.n_clips,
            )
            if candidate is not None:
                per_locus_candidates.append(candidate)
                n_loci_rescued += 1

    records = aggregate_junctions(per_locus_candidates, star_sj_set=star_sj_set)
    n_emitted_sj = sum(
        1 for r in records
        if r.n_reads >= 3 and f"{r.donor}-{r.acceptor}" in allowlist_set
    )
    n_novel = sum(1 for r in records if not r.was_in_star_sj)

    rescue_rate = (
        100.0 * n_loci_rescued / n_loci_attempted if n_loci_attempted else 0.0
    )
    return JunctionRescueResult(
        n_loci_attempted=n_loci_attempted,
        n_loci_rescued=n_loci_rescued,
        n_junctions_unique=len(records),
        n_junctions_novel_vs_star=n_novel,
        n_junctions_emitted_sj=n_emitted_sj,
        rescue_rate_pct=rescue_rate,
        dinuc_allowlist=list(effective_allowlist),
        records=records,
    )


# ---------------------------------------------------------------------------
# Per-locus rescue
# ---------------------------------------------------------------------------


def rescue_junction_for_locus(
    *,
    chrom: str,
    anchor_pos: int,
    bam_side: str,
    clip_consensus: str,
    anchor_adjacent_consensus: str,
    contigs,
    min_intron_len: int,
    max_intron_len: int,
    min_seed_extension: int,
    dinuc_allowlist: set[str],
    n_reads: int,
) -> JunctionCandidate | None:
    """Search for the clip consensus in the nearby genome window.

    Returns the best-scoring candidate clearing the seed-extension and
    dinucleotide allowlist filters, or ``None`` if nothing qualifies.

    ``min_seed_extension`` is the minimum number of clip bases past the
    Q-bp seed that must continue to match the genome at the hit. A
    positive value rejects fortuitous 20-mer matches and demands real
    extension into the candidate exon-2 region.
    """
    rec = contigs.get(chrom)
    if rec is None or rec.seq is None:
        return None
    contig_seq = str(rec.seq).upper()
    contig_len = len(contig_seq)

    q = min(20, len(clip_consensus))
    if q < 12:
        return None
    if bam_side == "right":
        seed = clip_consensus[:q]
        win_start = anchor_pos + min_intron_len
        win_end = min(contig_len, anchor_pos + max_intron_len + q)
    else:
        seed = clip_consensus[-q:]
        win_start = max(0, anchor_pos - max_intron_len - q)
        win_end = anchor_pos - min_intron_len
    if win_end <= win_start:
        return None

    # Clips too short to support the extension requirement after the
    # seed → reject up front (would never pass the filter anyway).
    if len(clip_consensus) - q < min_seed_extension:
        return None

    window = contig_seq[win_start:win_end]
    hits = _find_seed_hits(window, seed)
    if not hits:
        return None

    best: JunctionCandidate | None = None
    for hit_in_window in hits:
        hit = win_start + hit_in_window
        outward = _outward_match_len(
            contig_seq, hit, q, bam_side, clip_consensus,
        )
        if outward < min_seed_extension:
            continue

        for margin in _MARGIN_RANGE:
            if margin > 0 and not _margin_consistent(
                contig_seq, hit, q, bam_side, margin,
                anchor_pos, anchor_adjacent_consensus,
            ):
                continue
            intron_start, intron_end = _intron_coords(
                bam_side, hit, q, anchor_pos, margin,
            )
            if intron_start < 0 or intron_end > contig_len:
                continue
            if intron_end - intron_start < min_intron_len:
                continue
            donor = contig_seq[intron_start : intron_start + 2]
            acceptor = contig_seq[intron_end - 2 : intron_end]
            pair = f"{donor}-{acceptor}"
            if pair not in dinuc_allowlist:
                continue
            score = _score_candidate(outward, pair, margin)
            if best is None or score > best.score:
                strand = _STRAND_BY_DINUC.get(pair, ".")
                best = JunctionCandidate(
                    chrom=chrom,
                    intron_start=intron_start,
                    intron_end=intron_end,
                    strand=strand,
                    donor=donor,
                    acceptor=acceptor,
                    score=score,
                    outward_match=q + outward,
                    margin=margin,
                    n_reads=n_reads,
                    example_consensus=clip_consensus,
                )
    return best


def _find_seed_hits(window: str, seed: str) -> list[int]:
    """Return all exact-match positions of ``seed`` in ``window``.

    Falls back to a 1-mismatch sliding-window scan when no exact match
    is found, so a single sequencing error in the consensus doesn't
    abort the rescue.
    """
    if not seed or not window or len(seed) > len(window):
        return []
    hits: list[int] = []
    start = 0
    while True:
        idx = window.find(seed, start)
        if idx == -1:
            break
        hits.append(idx)
        start = idx + 1
    if hits:
        return hits

    # 1-mismatch fallback
    q = len(seed)
    for i in range(len(window) - q + 1):
        diffs = 0
        for j in range(q):
            if window[i + j] != seed[j]:
                diffs += 1
                if diffs > 1:
                    break
        if diffs <= 1:
            hits.append(i)
    return hits


def _outward_match_len(
    contig_seq: str, hit: int, q: int, bam_side: str, clip_consensus: str,
) -> int:
    """Extend the match outward (away from the alignment anchor).

    For ``bam_left``, the seed sits at the end of the upstream-exon
    match — extend leftward (toward smaller genome positions, deeper
    into upstream exon) by walking the clip's BAM-order bases just
    before the anchor end and comparing to genome bases just before
    ``hit``.

    For ``bam_right``, the seed sits at the start of the
    downstream-exon match — extend rightward into the clip body and the
    genome bases past ``hit + q``.

    Returns the number of additional bases matched past the seed
    (not counting the seed itself).
    """
    extended = 0
    if bam_side == "right":
        max_walk = min(
            len(clip_consensus) - q,
            len(contig_seq) - (hit + q),
            _OUTWARD_MAX,
        )
        for i in range(max_walk):
            if clip_consensus[q + i] == contig_seq[hit + q + i]:
                extended += 1
            else:
                break
    else:
        max_walk = min(
            len(clip_consensus) - q,
            hit,
            _OUTWARD_MAX,
        )
        # clip_consensus order: ... [-q-1, -q] is the seed start, the
        # bases immediately before that (more upstream in BAM order)
        # are at indices -q-1, -q-2, ...
        for i in range(max_walk):
            clip_idx = -q - 1 - i
            genome_idx = hit - 1 - i
            if clip_consensus[clip_idx] == contig_seq[genome_idx]:
                extended += 1
            else:
                break
    return extended


def _margin_consistent(
    contig_seq: str, hit: int, q: int, bam_side: str, margin: int,
    anchor_pos: int, anchor_adjacent_consensus: str,
) -> bool:
    """Verify a candidate ``margin`` against the genome and adjacent consensus.

    For ``bam_right`` with over-extension ``m``, the last ``m`` bases of
    the aligned block on the read (= last ``m`` chars of the adjacent
    consensus) should equal ``contig_seq[hit - m : hit]`` (those are
    the bases at the true exon-2 prefix that STAR mistakenly aligned to
    intron positions). For ``bam_left`` with ``m``, the first ``m``
    bases of the adjacent consensus should equal
    ``contig_seq[hit + q : hit + q + m]`` (true exon-1 tail).
    """
    if margin <= 0:
        return True
    adj = anchor_adjacent_consensus
    if len(adj) < margin:
        return False
    if bam_side == "right":
        if hit - margin < 0:
            return False
        expected = contig_seq[hit - margin : hit]
        observed = adj[-margin:]
    else:
        if hit + q + margin > len(contig_seq):
            return False
        expected = contig_seq[hit + q : hit + q + margin]
        observed = adj[:margin]
    return expected == observed


def _intron_coords(
    bam_side: str, hit: int, q: int, anchor_pos: int, margin: int,
) -> tuple[int, int]:
    """Derive the (intron_start, intron_end) 0-based half-open pair.

    ``bam_right`` (STAR aligned to upstream exon, clip is downstream):
      donor at ``anchor_pos - margin``, acceptor at ``hit - margin``.
    ``bam_left`` (STAR aligned to downstream exon, clip is upstream):
      donor at ``hit + q + margin``, acceptor at ``anchor_pos + margin``.
    """
    if bam_side == "right":
        intron_start = anchor_pos - margin
        intron_end = hit - margin
    else:
        intron_start = hit + q + margin
        intron_end = anchor_pos + margin
    return intron_start, intron_end


def _score_candidate(outward: int, dinuc_pair: str, margin: int) -> float:
    """Score a rescue candidate.

    Outward-match length dominates (the bulk of the evidence). A
    canonical/semi-canonical dinucleotide pair gives a small bonus.
    Margin > 0 incurs an Occam penalty so we prefer the simpler "no
    over-extension" interpretation when both fit.
    """
    bonus = 0.0
    if dinuc_pair in ("GT-AG", "CT-AC"):
        bonus = 2.0
    elif dinuc_pair in ("GC-AG", "CT-GC", "AT-AC", "GT-AT"):
        bonus = 1.0
    return float(outward) + bonus - 0.5 * margin


# ---------------------------------------------------------------------------
# Aggregation, STAR-SJ I/O, and writers
# ---------------------------------------------------------------------------


def aggregate_junctions(
    candidates: list[JunctionCandidate],
    *,
    star_sj_set: set[tuple[str, int, int, str]],
) -> list[JunctionRecord]:
    """Collapse per-locus candidates into per-junction records.

    Keyed by ``(chrom, intron_start, intron_end, strand)``. For each
    key, sums supporting reads, counts contributing loci, keeps the
    best outward-match seen, and marks ``was_in_star_sj`` if the
    junction matched an entry in the STAR SJ.out.tab.
    """
    by_key: dict[tuple[str, int, int, str], dict] = {}
    for c in candidates:
        key = (c.chrom, c.intron_start, c.intron_end, c.strand)
        agg = by_key.setdefault(key, {
            "donor": c.donor, "acceptor": c.acceptor,
            "n_loci": 0, "n_reads": 0,
            "max_outward_match": 0, "margin": c.margin,
            "example_consensus": c.example_consensus,
        })
        agg["n_loci"] += 1
        agg["n_reads"] += c.n_reads
        if c.outward_match > agg["max_outward_match"]:
            agg["max_outward_match"] = c.outward_match
            agg["margin"] = c.margin
            agg["example_consensus"] = c.example_consensus
    records = []
    for (chrom, istart, iend, strand), agg in by_key.items():
        records.append(JunctionRecord(
            chrom=chrom, intron_start=istart, intron_end=iend, strand=strand,
            donor=agg["donor"], acceptor=agg["acceptor"],
            n_loci=agg["n_loci"], n_reads=agg["n_reads"],
            max_outward_match=agg["max_outward_match"],
            margin=agg["margin"],
            was_in_star_sj=(chrom, istart, iend, strand) in star_sj_set,
            example_consensus=agg["example_consensus"],
        ))
    records.sort(key=lambda r: (-r.n_reads, r.chrom, r.intron_start))
    return records


def load_star_sj_set(sj_path: Path) -> set[tuple[str, int, int, str]]:
    """Load STAR ``SJ.out.tab`` into a set of ``(chrom, start, end, strand)`` tuples.

    Coordinates are converted from STAR's 1-based inclusive format to
    the 0-based half-open convention used internally
    (``intron_start = star_col2 - 1``, ``intron_end = star_col3``).
    Strand is mapped from STAR's ``0/1/2`` to ``./+/-``.
    """
    out: set[tuple[str, int, int, str]] = set()
    if not sj_path.exists():
        return out
    strand_map = {"0": ".", "1": "+", "2": "-"}
    with open(sj_path) as fin:
        reader = csv.reader(fin, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            chrom = row[0]
            try:
                star_start = int(row[1])
                star_end = int(row[2])
            except ValueError:
                continue
            strand = strand_map.get(row[3], ".")
            out.add((chrom, star_start - 1, star_end, strand))
    return out


def _resolve_allowlist(
    user_allowlist: list[str] | None,
    by_dinucleotide: dict[str, int] | None,
) -> list[str]:
    """Build the effective dinucleotide allowlist.

    Defaults to canonical + semi-canonical pairs. If a STAR-summary
    ``by_dinucleotide`` map is provided, non-canonical pairs are added
    when they clear BOTH ``_STAR_SUMMARY_MIN_FRACTION`` of all
    STAR-called introns AND ``_STAR_SUMMARY_MIN_UNIQUE_READS`` total
    supporting reads. The double threshold lets the rescue self-adapt
    to the organism's real splice landscape on small datasets, while
    excluding the long tail of noise pairs on organisms (like
    kinetoplastids) where most introns are non-canonical and almost
    every dinucleotide pair has thousands of reads.
    """
    if user_allowlist is not None:
        return list(user_allowlist)
    allow = list(_DEFAULT_DINUC_ALLOWLIST)
    if by_dinucleotide:
        total = sum(by_dinucleotide.values())
        for pair, n in by_dinucleotide.items():
            if pair in allow:
                continue
            if "-" not in pair or pair.startswith("-") or pair.endswith("-"):
                continue
            if n < _STAR_SUMMARY_MIN_UNIQUE_READS:
                continue
            if total and n / total < _STAR_SUMMARY_MIN_FRACTION:
                continue
            allow.append(pair)
    return allow


def write_junctions_tsv(records: list[JunctionRecord], path: Path) -> None:
    """Write the rich per-junction evidence TSV."""
    with open(path, "w") as f:
        f.write(
            "chrom\tintron_start\tintron_end\tstrand\tdonor\tacceptor\t"
            "n_loci\tn_reads\tmax_outward_match\tmargin\twas_in_star_sj\t"
            "example_consensus\n"
        )
        for r in records:
            f.write(
                f"{r.chrom}\t{r.intron_start}\t{r.intron_end}\t{r.strand}\t"
                f"{r.donor}\t{r.acceptor}\t{r.n_loci}\t{r.n_reads}\t"
                f"{r.max_outward_match}\t{r.margin}\t"
                f"{'true' if r.was_in_star_sj else 'false'}\t"
                f"{r.example_consensus}\n"
            )


def write_junctions_sj_tab(
    records: list[JunctionRecord], path: Path,
    *, min_reads: int = 3, dinuc_allowlist: list[str] | None = None,
) -> int:
    """Write a 4-col STAR ``--sjdbFileChrStartEnd``-format file.

    Filtered to records clearing ``min_reads`` AND falling within the
    dinucleotide allowlist (defaults to the canonical + semi-canonical
    set when ``dinuc_allowlist`` is ``None``).

    Returns the count of records actually written.
    """
    allow = set(dinuc_allowlist) if dinuc_allowlist is not None else set(_DEFAULT_DINUC_ALLOWLIST)
    n = 0
    with open(path, "w") as f:
        for r in records:
            if r.n_reads < min_reads:
                continue
            pair = f"{r.donor}-{r.acceptor}"
            if pair not in allow:
                continue
            # STAR SJ format: 1-based inclusive of both intron ends.
            star_start = r.intron_start + 1
            star_end = r.intron_end
            f.write(f"{r.chrom}\t{star_start}\t{star_end}\t{r.strand}\n")
            n += 1
    return n


def histogram_margins(records: list[JunctionRecord]) -> Counter:
    """Count records by margin value (0..5). Useful for spot-checks."""
    return Counter(r.margin for r in records)


def write_junctions_gff3(records: list[JunctionRecord], path: Path) -> None:
    """Write per-junction records as a GFF3 file for genome-browser inspection.

    Each rescue becomes an ``intron`` feature with the supporting
    evidence packed into the attributes column. Score column carries
    ``n_reads`` so viewers can size/colour by depth.
    """
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        for i, r in enumerate(records):
            attrs = ";".join([
                f"ID=rescue_{i + 1}",
                f"donor={r.donor}",
                f"acceptor={r.acceptor}",
                f"n_loci={r.n_loci}",
                f"n_reads={r.n_reads}",
                f"match={r.max_outward_match}",
                f"margin={r.margin}",
                f"known_in_STAR={'yes' if r.was_in_star_sj else 'no'}",
                f"example_consensus={r.example_consensus}",
            ])
            # GFF3 is 1-based inclusive on both ends. intron_start is
            # 0-based inclusive → +1 for GFF. intron_end is 0-based
            # exclusive → leave as-is (it's the 1-based inclusive end).
            f.write(
                f"{r.chrom}\teukan_rescue\tintron\t"
                f"{r.intron_start + 1}\t{r.intron_end}\t"
                f"{r.n_reads}\t{r.strand}\t.\t{attrs}\n"
            )


def read_junctions_tsv(path: Path) -> list[JunctionRecord]:
    """Round-trip helper: parse a ``rescued_junctions.tsv`` back into records."""
    records: list[JunctionRecord] = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            records.append(JunctionRecord(
                chrom=parts[col["chrom"]],
                intron_start=int(parts[col["intron_start"]]),
                intron_end=int(parts[col["intron_end"]]),
                strand=parts[col["strand"]],
                donor=parts[col["donor"]],
                acceptor=parts[col["acceptor"]],
                n_loci=int(parts[col["n_loci"]]),
                n_reads=int(parts[col["n_reads"]]),
                max_outward_match=int(parts[col["max_outward_match"]]),
                margin=int(parts[col["margin"]]),
                was_in_star_sj=parts[col["was_in_star_sj"]].lower() == "true",
                example_consensus=parts[col["example_consensus"]],
            ))
    return records
