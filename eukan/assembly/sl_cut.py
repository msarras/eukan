"""Genomic spliced-leader cut: split transcript models at trans-splice acceptors.

Reuses the jaccard GFF3/GTF split logic (:mod:`eukan.assembly.jaccard`) — the cut
is the same exon-segregating split, fed SL acceptor *genomic* coordinates
(:mod:`eukan.assembly.sl_acceptors`) instead of read-coverage troughs. For each
transcript whose exons contain a same-strand acceptor, the model is cut so the
mature mRNA begins at the acceptor. An SL site imposes its strand on an
otherwise-unstranded (``.``) StringTie transcript.

Inputs cut: the StringTie GTF (genome-guided) and the de novo transcript→genome
BAMs (converted to gene>mRNA>exon GFF3 first). All outputs are genome-coordinate
GFF3 that ``combinr assemble`` ingests directly. Both the BAM→GFF3 conversion and
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
from eukan.assembly.jaccard import _parse_attrs, _split_transcript, _Tx
from eukan.assembly.sl_acceptors import AcceptorSite, load_sl_acceptors
from eukan.infra.artifacts import Artifact
from eukan.infra.logging import get_logger
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

# Ref-consuming CIGAR ops that stay within one exon block (N splits exons).
_EXON_REF = frozenset([_CIGAR_M, _CIGAR_D, _CIGAR_EQ, _CIGAR_X])

_DENOVO_BAMS = ("rnaspades.genome.bam",)
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


def _cut_one(tx: _Tx, sites: list[AcceptorSite], min_segment: int) -> list[_Tx] | None:
    """Split *tx* at same-strand acceptors; ``None`` if nothing to cut.

    A ``.``-strand transcript is oriented by the acceptors when they agree on a
    single strand (the SL imposes strand); conflicting strands leave it unchanged.
    """
    if not sites:
        return None
    if tx.strand in ("+", "-"):
        strand = tx.strand
        usable = [s for s in sites if s.strand == tx.strand]
    else:
        strands = {s.strand for s in sites}
        if len(strands) != 1:
            return None
        strand = next(iter(strands))
        usable = sites
    if not usable:
        return None

    exons_5to3 = tx.exons if strand != "-" else list(reversed(tx.exons))
    total_len = sum(e - s + 1 for s, e in tx.exons)
    clips: set[int] = set()
    for site in usable:
        off = _project_genomic_to_spliced(exons_5to3, strand, site.pos)
        # Cut *before* the acceptor base so the downstream piece starts at it.
        if off is not None and 1 < off <= total_len:
            clips.add(off - 1)
    valid = sorted(c for c in clips if 0 < c < total_len)
    if not valid:
        return None

    oriented = (
        tx if tx.strand == strand else _Tx(tx.tid, tx.chrom, strand, tx.source, tx.exons)
    )
    return _split_transcript(oriented, valid, min_segment)


def cut_models_at_sl(
    gff_or_gtf: str | Path, sites: list[AcceptorSite], out_gff: str | Path, *, min_segment: int
) -> int:
    """Cut every transcript in *gff_or_gtf* at its same-strand SL acceptors.

    Streams: reads one transcript model, cuts it, writes the result, and moves on —
    no full model list is held. Transcripts with no applicable acceptor pass through
    unchanged. Returns the number of transcripts that were cut.
    """
    by_chrom: dict[str, list[AcceptorSite]] = {}
    for site in sites:
        by_chrom.setdefault(site.chrom, []).append(site)

    n_cut = 0
    with open(out_gff, "w") as fh:
        fh.write("##gff-version 3\n")
        for tx in _iter_transcript_models(gff_or_gtf):
            span_lo, span_hi = tx.exons[0][0], tx.exons[-1][1]
            relevant = [s for s in by_chrom.get(tx.chrom, []) if span_lo <= s.pos <= span_hi]
            result = _cut_one(tx, relevant, min_segment)
            if result is None:
                _write_tx(fh, tx)
            else:
                for piece in result:
                    _write_tx(fh, piece)
                n_cut += 1
    return n_cut


def run_sl_cut(config: AssemblyConfig) -> None:
    """Cut the StringTie GTF and de novo transcript→genome models at SL acceptors."""
    wd = config.work_dir
    acc_path = wd / Artifact.SL_ACCEPTORS.value
    sites = load_sl_acceptors(acc_path) if acc_path.exists() else []
    min_segment = config.min_sl_fragment

    stringtie = wd / "stringtie.gtf"
    if stringtie.exists():
        out = wd / "stringtie.sl_cut.gff3"
        n_cut = cut_models_at_sl(stringtie, sites, out, min_segment=min_segment)
        log.info("SL cut stringtie.gtf -> %s (%d transcripts cut).", out.name, n_cut)

    for bam_name in _DENOVO_BAMS:
        bam_path = wd / bam_name
        if not bam_path.exists():
            continue
        stem = bam_name[: -len(_GENOME_BAM_SUFFIX)]
        gff = wd / f"{stem}.genome.gff3"
        n_models = bam_to_transcript_gff3(bam_path, gff, source=stem)
        out = wd / f"{stem}.genome.sl_cut.gff3"
        n_cut = cut_models_at_sl(gff, sites, out, min_segment=min_segment)
        log.info(
            "SL cut %s (%d models) -> %s (%d cut).", gff.name, n_models, out.name, n_cut
        )
