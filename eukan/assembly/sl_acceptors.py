"""Spliced-leader (SL) trans-splice acceptor detection (genomic coordinates).

A trans-spliced organism adds a constant spliced leader to the 5' end of every
mature mRNA from a separate SL-RNA locus, so the leader is not genomic at the
gene. Two orthogonal signals expose the acceptor site once sequences are mapped
to the genome:

* **reads → genome** (the aligner BAM): a read spanning the trans-splice junction
  soft-clips its SL bases; the genomic boundary of the clip is the acceptor.
* **de novo transcripts → genome** (the map_transcripts BAMs): the SL shows up as
  a terminal soft-clip (a 5' leader) or, in a fused / multi-leader contig, as an
  *internal insertion* — the dominant signal in gene-dense trans-spliced genomes.

This module pools both signals, derives one SL consensus jointly (an explicit
``sl_sequence`` override wins; else the read-side soft-clip verdict; else the
dominant de novo insertion motif), detects acceptor sites in every BAM, then
consolidates and persists them as ``sl_acceptors.gff3``. The SL-cut step
(:mod:`eukan.assembly.sl_cut`) consumes the file. With no SL signal the step is a
no-op (an empty acceptor file is written), so non-trans-spliced organisms are
unaffected.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pysam

from eukan.assembly.bam_diagnostic import (
    _CIGAR_M,
    _CIGAR_REF_CONSUMING,
    _CIGAR_SOFT_CLIP,
    _iter_primary_alignments,
)
from eukan.assembly.sl_depletion import (
    _MAX_MISMATCHES,
    _MIN_MOTIF_LEN,
    _find_sites,
    _revcomp,
    _variants,
)
from eukan.infra.artifacts import Artifact
from eukan.infra.logging import get_logger
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

# Insertion CIGAR op (bam_diagnostic only tracks soft-clips, so define it here).
_CIGAR_INSERTION = 1
_CIGAR_QUERY_CONSUMING = frozenset([_CIGAR_M, _CIGAR_INSERTION, _CIGAR_SOFT_CLIP, 7, 8])

# Length of the anchored window used to derive an SL consensus from de novo
# insertions (only when neither an override nor the read verdict supplies one).
_CONSENSUS_LEN = 16
_MIN_DENOVO_SUPPORT = 3

# De novo transcript→genome BAMs scanned for Source B (clips + insertions).
_DENOVO_BAMS = ("rnaspades.genome.bam",)
_GENOME_BAM_SUFFIX = ".genome.bam"


@dataclass(frozen=True)
class AcceptorSite:
    """A consolidated SL trans-splice acceptor: the mature mRNA's 5' genomic base."""

    chrom: str
    pos: int  # 1-based genomic acceptor
    strand: str  # '+' / '-'
    support: int
    sources: tuple[str, ...]


def _iter_sl_ops(
    read: pysam.AlignedSegment,
    *,
    min_clip_len: int,
    min_ins_len: int,
    scan_insertions: bool,
):
    """Yield ``(acceptor_1based, strand, bases)`` for each candidate SL op.

    The candidates are the mRNA-5' soft-clip (the leader is a 5' addition) and,
    when *scan_insertions*, every *internal* insertion. ``acceptor_1based`` is the
    mature mRNA's first genomic base; ``strand`` is the gene strand inferred from
    the read geometry; ``bases`` are the reference-forward clip/insertion bases to
    test against the SL patterns (matched on both strands, so orientation of the
    returned bases does not matter).
    """
    cigar = read.cigartuples
    seq = read.query_sequence
    ref_end = read.reference_end
    if cigar is None or seq is None or ref_end is None:
        return
    rev = read.is_reverse
    ref_start = read.reference_start

    # The SL sits at the mRNA 5' end: a leading soft-clip on a forward read, a
    # trailing soft-clip on a reverse read. (reference_end is 0-based exclusive,
    # i.e. the 1-based position of the last aligned base.)
    op0, len0 = cigar[0]
    if not rev and op0 == _CIGAR_SOFT_CLIP and len0 >= min_clip_len:
        yield ref_start + 1, "+", seq[:len0]
    op_last, len_last = cigar[-1]
    if rev and op_last == _CIGAR_SOFT_CLIP and len_last >= min_clip_len:
        yield ref_end, "-", seq[-len_last:]

    if not scan_insertions:
        return
    qpos = 0
    rpos = ref_start  # 0-based reference cursor
    last = len(cigar) - 1
    for idx, (op, length) in enumerate(cigar):
        if op == _CIGAR_INSERTION:
            if 0 < idx < last and length >= min_ins_len:
                ins = seq[qpos : qpos + length]
                # The mature mRNA continues just past the leader: downstream is
                # the higher coordinate on '+', the lower on '-'.
                if not rev:
                    yield rpos + 1, "+", ins
                else:
                    yield rpos, "-", ins
            qpos += length
            continue
        if op in _CIGAR_QUERY_CONSUMING:
            qpos += length
        if op in _CIGAR_REF_CONSUMING:
            rpos += length


def _read_verdict_consensus(wd: Path) -> str | None:
    """The read-side SL consensus from the soft-clip diagnostic, gated on verdict."""
    summary = wd / Artifact.SOFTCLIP_DIAGNOSTIC.value
    if not summary.exists():
        return None
    ts = json.loads(summary.read_text()).get("verdict", {}).get("trans_splicing", {})
    if ts.get("call") not in ("STRONG", "MODERATE"):
        return None
    motif = (
        ts.get("top_non_trivial_cluster_consensus")
        or ts.get("top_non_trivial_cluster_key")
        or ""
    ).upper()
    return motif if len(motif) >= _MIN_MOTIF_LEN else None


def _dominant_denovo_motif(bams: list[Path], *, min_support: int) -> str | None:
    """Most common anchored SL window across de novo clip/insertion bases.

    Pools the head and tail ``_CONSENSUS_LEN``-mers of every candidate SL op; the
    leader is constant, so its windows dominate. Used only as a fallback when the
    read-side verdict is too weak to supply a consensus.
    """
    counts: Counter[str] = Counter()
    for bam_path in bams:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in _iter_primary_alignments(bam, min_mapq=0):
                for _pos, _strand, bases in _iter_sl_ops(
                    read,
                    min_clip_len=_CONSENSUS_LEN,
                    min_ins_len=_CONSENSUS_LEN,
                    scan_insertions=True,
                ):
                    b = bases.upper()
                    if len(b) >= _CONSENSUS_LEN:
                        counts[b[:_CONSENSUS_LEN]] += 1
                        counts[b[-_CONSENSUS_LEN:]] += 1
    if not counts:
        return None
    motif, n = counts.most_common(1)[0]
    return motif if n >= min_support and len(motif) >= _MIN_MOTIF_LEN else None


def build_joint_consensus(
    config: AssemblyConfig, denovo_bams: list[Path]
) -> str | None:
    """The SL consensus to detect with, pooling read and de novo signal.

    Priority: explicit ``sl_sequence`` override → read-side soft-clip verdict
    (authoritative when trans-splicing is called STRONG/MODERATE) → the dominant
    de novo insertion motif (so strong de novo signal still drives detection when
    the read verdict is borderline). ``None`` means no SL signal.
    """
    if config.sl_sequence:
        return config.sl_sequence.strip().upper()
    read_motif = _read_verdict_consensus(config.work_dir)
    if read_motif:
        log.info("SL consensus from read soft-clip verdict: %s", read_motif)
        return read_motif
    denovo_motif = _dominant_denovo_motif(denovo_bams, min_support=_MIN_DENOVO_SUPPORT)
    if denovo_motif:
        log.info("SL consensus from de novo insertions (read verdict weak): %s", denovo_motif)
        return denovo_motif
    return None


def _consolidate(
    raw: dict[tuple[str, str, int], tuple[int, set[str]]], *, window: int
) -> list[AcceptorSite]:
    """Merge per-(chrom, strand) acceptor positions within *window* bp.

    Each cluster's representative position is its highest-support member; support
    is summed and sources unioned across the cluster.
    """
    by_cs: dict[tuple[str, str], list[tuple[int, int, set[str]]]] = {}
    for (chrom, strand, pos), (support, sources) in raw.items():
        by_cs.setdefault((chrom, strand), []).append((pos, support, sources))

    sites: list[AcceptorSite] = []
    for (chrom, strand), entries in by_cs.items():
        entries.sort()
        cluster: list[tuple[int, int, set[str]]] = []
        for entry in entries:
            if cluster and entry[0] - cluster[-1][0] > window:
                sites.append(_collapse(chrom, strand, cluster))
                cluster = []
            cluster.append(entry)
        if cluster:
            sites.append(_collapse(chrom, strand, cluster))
    sites.sort(key=lambda s: (s.chrom, s.pos, s.strand))
    return sites


def _collapse(
    chrom: str, strand: str, cluster: list[tuple[int, int, set[str]]]
) -> AcceptorSite:
    best = max(cluster, key=lambda e: e[1])
    total = sum(e[1] for e in cluster)
    srcs: set[str] = set().union(*(e[2] for e in cluster))
    return AcceptorSite(chrom, best[0], strand, total, tuple(sorted(srcs)))


def _write_acceptors(sites: list[AcceptorSite], out: Path) -> None:
    with open(out, "w") as fh:
        fh.write("##gff-version 3\n")
        for i, s in enumerate(sites, start=1):
            fh.write(
                f"{s.chrom}\teukan-sl\tSL_acceptor\t{s.pos}\t{s.pos}\t{s.support}\t"
                f"{s.strand}\t.\tID=sl{i};sources={','.join(s.sources)}\n"
            )


def load_sl_acceptors(path: str | Path) -> list[AcceptorSite]:
    """Parse an ``sl_acceptors.gff3`` back into :class:`AcceptorSite` records."""
    sites: list[AcceptorSite] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "SL_acceptor":
                continue
            attrs = dict(p.split("=", 1) for p in cols[8].split(";") if "=" in p)
            support = int(cols[5]) if cols[5].lstrip("-").isdigit() else 0
            raw_src = attrs.get("sources", "")
            sources = tuple(raw_src.split(",")) if raw_src else ()
            sites.append(AcceptorSite(cols[0], int(cols[3]), cols[6], support, sources))
    return sites


def detect_sl_acceptors(config: AssemblyConfig) -> None:
    """Detect, consolidate, and persist SL trans-splice acceptor sites."""
    wd = config.work_dir
    read_bam = wd / config.aligner_bam
    denovo_bams = [wd / b for b in _DENOVO_BAMS if (wd / b).exists()]
    out = wd / Artifact.SL_ACCEPTORS.value

    consensus = build_joint_consensus(config, denovo_bams)
    if consensus is None:
        log.info("No spliced-leader signal; SL acceptor detection is a no-op.")
        _write_acceptors([], out)
        return

    patterns = _variants(consensus, _MAX_MISMATCHES) | _variants(
        _revcomp(consensus), _MAX_MISMATCHES
    )
    motif_len = len(consensus)

    raw: dict[tuple[str, str, int], tuple[int, set[str]]] = {}

    def _scan(bam_path: Path, source: str, *, scan_insertions: bool) -> None:
        if not bam_path.exists():
            return
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in _iter_primary_alignments(bam, min_mapq=0):
                chrom = read.reference_name
                if chrom is None:
                    continue
                for pos, strand, bases in _iter_sl_ops(
                    read,
                    min_clip_len=config.min_sl_clip_len,
                    min_ins_len=config.min_sl_insertion_len,
                    scan_insertions=scan_insertions,
                ):
                    if _find_sites(bases.upper(), patterns, motif_len):
                        support, sources = raw.get((chrom, strand, pos), (0, set()))
                        sources.add(source)
                        raw[(chrom, strand, pos)] = (support + 1, sources)

    _scan(read_bam, "reads", scan_insertions=False)
    for bam_path in denovo_bams:
        _scan(bam_path, bam_path.name[: -len(_GENOME_BAM_SUFFIX)], scan_insertions=True)

    sites = _consolidate(raw, window=config.sl_cluster_window)
    _write_acceptors(sites, out)
    log.info(
        "SL acceptor detection: %d sites from %d raw positions (consensus %s).",
        len(sites), len(raw), consensus,
    )
