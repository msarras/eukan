"""Genomic spliced-leader cut: split transcript models at trans-splice acceptors.

Reuses the jaccard GFF3/GTF split logic (:mod:`eukan.assembly.jaccard`) — the cut
is the same exon-segregating split, fed SL acceptor *genomic* coordinates
(:mod:`eukan.assembly.sl_acceptors`) instead of read-coverage troughs. For each
transcript whose exons contain a same-strand acceptor, the model is cut so the
mature mRNA begins at the acceptor. An SL site imposes its strand on an
otherwise-unstranded (``.``) transcript.

Inputs cut: the two Trinity transcript→genome model sets (de novo + genome-guided),
each converted from its BAM to gene>mRNA>exon GFF3 upstream by strand_correct. All
outputs are genome-coordinate GFF3 that ``combinr assemble`` ingests directly. Both
the BAM→GFF3 conversion and
the cut **stream** one transcript at a time — never materialising the full
alignment set — so a genome-wide BAM with millions of records stays bounded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pysam

from eukan.assembly.bam_diagnostic import (
    _CIGAR_D,
    _CIGAR_EQ,
    _CIGAR_M,
    _CIGAR_N,
    _CIGAR_X,
)
from eukan.assembly.jaccard import (
    _parse_attrs,
    _split_transcript,
    _Tx,
)
from eukan.assembly.sl_acceptors import AcceptorSite, load_sl_acceptors
from eukan.assembly.tracks import mapped_transcript_stems, resolve_model_source
from eukan.infra.artifacts import Artifact
from eukan.infra.logging import get_logger
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

# Ref-consuming CIGAR ops that stay within one exon block (N splits exons).
_EXON_REF = frozenset([_CIGAR_M, _CIGAR_D, _CIGAR_EQ, _CIGAR_X])

_DENOVO_BAMS = ("trinity-denovo.genome.bam", "trinity-gg.genome.bam")
_GENOME_BAM_SUFFIX = ".genome.bam"


def _alignment_exons(read: pysam.AlignedSegment) -> list[tuple[int, int]]:
    """Genomic 1-based exon blocks for one alignment (split at CIGAR N)."""
    cigar = read.cigartuples
    if cigar is None:
        return []
    blocks: list[tuple[int, int]] = []
    cur_start = read.reference_start  # 0-based
    pos = cur_start
    for op, length in cigar:
        if op == _CIGAR_N:
            blocks.append((cur_start + 1, pos))
            pos += length
            cur_start = pos
        elif op in _EXON_REF:
            pos += length
    blocks.append((cur_start + 1, pos))
    return [b for b in blocks if b[1] >= b[0]]


def _write_tx(fh, tx: _Tx) -> None:
    """Write one transcript as gene>mRNA>exon GFF3 (combinr-ingestible).

    Matches :func:`jaccard._write_transcript_models_gff3`'s per-model format so
    the output round-trips through :func:`jaccard._parse_transcript_models`.
    """
    gid, gstart, gend = f"{tx.tid}.gene", tx.exons[0][0], tx.exons[-1][1]
    loc = f"{tx.chrom}\t{tx.source}"
    fh.write(f"{loc}\tgene\t{gstart}\t{gend}\t.\t{tx.strand}\t.\tID={gid}\n")
    fh.write(f"{loc}\tmRNA\t{gstart}\t{gend}\t.\t{tx.strand}\t.\tID={tx.tid};Parent={gid}\n")
    for k, (s, e) in enumerate(tx.exons, start=1):
        fh.write(
            f"{loc}\texon\t{s}\t{e}\t.\t{tx.strand}\t.\tID={tx.tid}.exon{k};Parent={tx.tid}\n"
        )


def _iter_transcript_models(gff: str | Path) -> Iterator[_Tx]:
    """Stream transcript models from a GFF3/GTF, one ``_Tx`` at a time.

    Groups consecutive ``exon`` rows by transcript id (``Parent=`` / ``transcript_id``),
    assuming each transcript's exon rows are contiguous — true for StringTie GTF
    and the BAM-derived GFF3 written here. Streaming keeps memory bounded on a
    genome-wide model set.
    """
    cur: _Tx | None = None
    with open(gff) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "exon":
                continue
            attrs = _parse_attrs(cols[8])
            tid = attrs.get("Parent") or attrs.get("transcript_id")
            if not tid:
                continue
            if cur is None or cur.tid != tid:
                if cur is not None and cur.exons:
                    cur.exons.sort()
                    yield cur
                cur = _Tx(tid, cols[0], cols[6], cols[1])
            cur.exons.append((int(cols[3]), int(cols[4])))
    if cur is not None and cur.exons:
        cur.exons.sort()
        yield cur


def bam_to_transcript_gff3(bam_path: Path, out_gff: Path, source: str) -> int:
    """Stream one mRNA per mapped alignment (exons from CIGAR N) to *out_gff*.

    Each alignment gets a unique mRNA id (``<qname>.m<n>``) — an aligner reporting
    a transcript at every locus it maps to would otherwise collide ids, and combinr
    groups exons by ``Parent``, silently fusing distinct loci. Secondary alignments
    are kept (extra loci of multi-copy genes); supplementary/unmapped are skipped.
    Writes incrementally so a genome-wide BAM never builds a full in-memory list.
    """
    seen: Counter[str] = Counter()
    n = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, open(out_gff, "w") as fh:
        fh.write("##gff-version 3\n")
        for read in bam:
            if read.is_unmapped or read.is_supplementary or read.cigartuples is None:
                continue
            chrom = read.reference_name
            if chrom is None:
                continue
            exons = _alignment_exons(read)
            if not exons:
                continue
            qname = read.query_name or "tx"
            seen[qname] += 1
            strand = "-" if read.is_reverse else "+"
            _write_tx(fh, _Tx(f"{qname}.m{seen[qname]}", chrom, strand, source, exons))
            n += 1
    return n


def _project_genomic_to_spliced(
    exons_5to3: list[tuple[int, int]], strand: str, pos: int
) -> int | None:
    """1-based spliced offset of genomic *pos*, or ``None`` if intronic/outside.

    Inverse of :func:`jaccard._partition_exons`'s span accumulation: walk exons in
    5'→3' order (ascending genomic for ``+``, descending for ``-``) summing
    spliced length until *pos* lands inside an exon.
    """
    consumed = 0
    for gstart, gend in exons_5to3:
        if gstart <= pos <= gend:
            if strand == "-":
                return consumed + (gend - pos) + 1
            return consumed + (pos - gstart) + 1
        consumed += gend - gstart + 1
    return None


def _long_intron_cut_offsets(
    exons_5to3: list[tuple[int, int]], strand: str, max_intron_len: int
) -> set[int]:
    """Spliced "cut after base P" offsets severing every intron > *max_intron_len*.

    *exons_5to3* are 1-based inclusive genomic blocks in 5'->3' order (ascending for
    ``+``/``.``, descending for ``-``) — the order :func:`jaccard._split_transcript`
    and :func:`jaccard._partition_exons` expect. Each gap between consecutive blocks
    is an intron; when its genomic length exceeds the limit, cut at the cumulative
    spliced end of the 5'-side exon, which severs the model at that intron. Returns
    offsets in the same spliced space as the SL clips so the two unite cleanly.
    ``max_intron_len <= 0`` disables the cut.
    """
    if max_intron_len <= 0 or len(exons_5to3) < 2:
        return set()
    cuts: set[int] = set()
    consumed = 0
    for i, (gstart, gend) in enumerate(exons_5to3):
        consumed += gend - gstart + 1
        if i + 1 < len(exons_5to3):
            ns, ne = exons_5to3[i + 1]
            intron_len = (ns - gend - 1) if strand != "-" else (gstart - ne - 1)
            if intron_len > max_intron_len:
                cuts.add(consumed)
    return cuts


def _count_long_introns(tx: _Tx, max_intron_len: int) -> int:
    """Number of introns in *tx* longer than *max_intron_len* (genomic gaps)."""
    if max_intron_len <= 0:
        return 0
    ex = tx.exons  # start-sorted ascending
    return sum(
        1 for i in range(len(ex) - 1) if ex[i + 1][0] - ex[i][1] - 1 > max_intron_len
    )


def _cut_one(
    tx: _Tx, sites: list[AcceptorSite], min_segment: int, *, max_intron_len: int
) -> list[_Tx] | None:
    """Split *tx* at same-strand SL acceptors and at over-long introns; ``None`` if
    nothing to cut.

    A ``.``-strand transcript is oriented by the acceptors when they agree on a
    single strand (the SL imposes strand); conflicting strands skip the SL cut. The
    max-intron cut is strand-agnostic (a genomic gap is severed regardless), so it
    applies even with no usable SL acceptor — the two cut sets unite in one
    partition pass.
    """
    usable: list[AcceptorSite] = []
    strand = tx.strand
    if sites:
        if tx.strand in ("+", "-"):
            usable = [s for s in sites if s.strand == tx.strand]
        else:
            strands = {s.strand for s in sites}
            if len(strands) == 1:
                strand = next(iter(strands))
                usable = sites
            # conflicting SL strands: no SL-imposed strand, but long introns still cut

    exons_5to3 = tx.exons if strand != "-" else list(reversed(tx.exons))
    total_len = sum(e - s + 1 for s, e in tx.exons)

    clips: set[int] = set()
    for site in usable:
        off = _project_genomic_to_spliced(exons_5to3, strand, site.pos)
        # Cut *before* the acceptor base so the downstream piece starts at it.
        if off is not None and 1 < off <= total_len:
            clips.add(off - 1)
    clips |= _long_intron_cut_offsets(exons_5to3, strand, max_intron_len)

    valid = sorted(c for c in clips if 0 < c < total_len)
    if not valid:
        return None

    oriented = (
        tx if tx.strand == strand else _Tx(tx.tid, tx.chrom, strand, tx.source, tx.exons)
    )
    return _split_transcript(oriented, valid, min_segment)


def cut_models_at_sl(
    gff_or_gtf: str | Path,
    sites: list[AcceptorSite],
    out_gff: str | Path,
    *,
    min_segment: int,
    max_intron_len: int,
) -> tuple[int, int]:
    """Cut every transcript in *gff_or_gtf* at SL acceptors and over-long introns.

    Streams: reads one transcript model, cuts it, writes the result, and moves on —
    no full model list is held. Transcripts with no applicable acceptor *and* no
    over-long intron pass through unchanged. Returns
    ``(transcripts_cut, over_long_introns_severed)``.
    """
    by_chrom: dict[str, list[AcceptorSite]] = {}
    for site in sites:
        by_chrom.setdefault(site.chrom, []).append(site)

    n_cut = 0
    n_long = 0
    n_conflict = 0
    with open(out_gff, "w") as fh:
        fh.write("##gff-version 3\n")
        for tx in _iter_transcript_models(gff_or_gtf):
            span_lo, span_hi = tx.exons[0][0], tx.exons[-1][1]
            relevant = [s for s in by_chrom.get(tx.chrom, []) if span_lo <= s.pos <= span_hi]
            long_here = _count_long_introns(tx, max_intron_len)
            result = _cut_one(tx, relevant, min_segment, max_intron_len=max_intron_len)
            if result is None:
                # A multi-exon, stranded transcript whose only in-span SL acceptors
                # sit on the opposite strand (and no over-long intron forced a cut):
                # the splice-derived strand and the SL disagree. Trust the introns —
                # keep the strand, skip the cut.
                if (
                    relevant and long_here == 0 and len(tx.exons) > 1
                    and tx.strand in ("+", "-")
                    and all(s.strand != tx.strand for s in relevant)
                ):
                    n_conflict += 1
                _write_tx(fh, tx)
            else:
                for piece in result:
                    _write_tx(fh, piece)
                n_cut += 1
                n_long += long_here
    if n_conflict:
        log.warning(
            "%d multi-exon transcript(s) had SL acceptors only on the opposite "
            "strand; kept the splice-derived strand and skipped those cuts.",
            n_conflict,
        )
    return n_cut, n_long


def run_sl_cut(config: AssemblyConfig) -> None:
    """Cut StringTie and de novo transcript models at SL acceptors and over-long introns.

    Inputs come from the ``strand_correct`` step: prefer its ``*.stranded.gff3``
    (homology-corrected strands), falling back to the raw models when correction was
    a no-op (stranded library or no ``--uniprot``). The de novo BAM→GFF3 conversion
    now lives in that step (:mod:`eukan.assembly.strand_correction`).

    Both model sources stream through here, so this is also where the
    ``max_intron_len`` limit is hard-imposed on the transcript models: any intron
    longer than it splits the model into separate genes (the de novo segemehl path
    has no native intron bound). The resume fingerprint folds in ``max_intron_len``
    (see :mod:`eukan.assembly.pipeline`), so tightening ``-M`` re-runs this step.
    """
    wd = config.work_dir
    acc_path = wd / Artifact.SL_ACCEPTORS.value
    sites = load_sl_acceptors(acc_path) if acc_path.exists() else []
    min_segment = config.min_sl_fragment

    # Model-source precedence per Trinity track (latest variant wins, via
    # tracks.resolve_model_source): the homology de-fused ``*.defuse.gff3``
    # (defuse.py, --defuse), then the homology-stranded ``*.stranded.gff3``
    # (strand_correct), then the raw genome GFF3 (built from jaccard-clipped FASTAs
    # upstream, so it needs no resolver).
    for stem in mapped_transcript_stems():
        src = resolve_model_source(wd, stem)
        if src is None:
            continue
        out = wd / f"{stem}.sl_cut.gff3"
        n_cut, n_long = cut_models_at_sl(
            src, sites, out,
            min_segment=min_segment, max_intron_len=config.max_intron_len,
        )
        log.info(
            "SL + max-intron cut %s -> %s (%d transcripts cut, %d over-long introns severed).",
            src.name, out.name, n_cut, n_long,
        )
