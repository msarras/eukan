"""segemehl read mapping — splice-agnostic alternative to STAR.

Unlike STAR, segemehl does not enforce canonical GT-AG splice sites, so it
captures non-canonical introns (e.g. the dominant CG-AG introns of diplonemids
such as *Hemistasia*) that STAR would miss or misplace. segemehl has no native
splice-junction table, so we derive a STAR-format ``SJ.out.tab`` from the BAM's
N-CIGAR junctions (:func:`align_hints.sj_table_from_bam`) and reuse the shared
post-alignment processing verbatim. Downstream steps (GeneMark, AUGUSTUS,
Trinity, PASA) therefore see the identical contract STAR produces — including
the ``splice_site_summary.json`` that lets AUGUSTUS allow the non-canonical
splice sites via ``--allow_hinted_splicesites``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from eukan.assembly.align_hints import generate_rnaseq_hints, sj_table_from_bam
from eukan.exceptions import ExternalToolError
from eukan.infra.logging import get_logger
from eukan.infra.runner import run_cmd, run_piped
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

_BAM = "segemehl_Aligned.sortedByCoord.out.bam"
_UNSORTED_BAM = "segemehl_unsorted.bam"
_SJ = "segemehl_SJ.out.tab"
_INDEX = "segemehl.idx"
# segemehl emits split-read byproducts as <base>.sngl.bed / .mult.bed / .trns.txt;
# pin <base> to a known path so they can be cleaned up afterwards.
_SPLITS_BASE = "segemehl_splits"
_SPLITS_SUFFIXES = (".sngl.bed", ".mult.bed", ".trns.txt")

# The coordinate-sort needs transient room for its temp spill (~= the sorted
# size) plus the final BAM, both landing on the run-dir filesystem — roughly
# twice the unsorted BAM. Warn before sorting when free space is below this.
_SORT_DISK_HEADROOM = 2
# samtools' multithreaded BGZF writer reports a failed write on a full disk
# with a misleading errno (e.g. "...: Illegal seek"); match these stderr
# fragments so we can translate the cryptic failure into a clear out-of-space
# message instead of letting the user chase a phantom seek bug.
_DISK_FULL_MARKERS = ("illegal seek", "no space", "failed writing", "write failed")


def _bam_is_complete(path: Path) -> bool:
    """True if *path* is a non-empty BAM with a valid BGZF EOF (``quickcheck``).

    Lets a re-run reuse an unsorted BAM from a previous attempt that failed
    only in the downstream sort (segemehl mapping is the multi-hour step, so we
    never want to redo it when its output survived). A truncated BAM from an
    interrupted or killed segemehl fails quickcheck and is re-mapped.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        run_cmd(["samtools", "quickcheck", str(path)], cwd=path.parent)
    except ExternalToolError:
        return False
    return True


def _coordinate_sort_and_filter(
    unsorted: Path, out_bam: str, wd: Path, num_cpu: int
) -> None:
    """Drop unmapped reads (``-F 4``) and coordinate-sort into *out_bam*.

    Mirrors the validated recipe's ``samtools sort | samtools view -bh -F 4``;
    filtering before the sort is cheaper and yields the same result. On a tight
    disk this is the step that fails — samtools needs roughly twice the unsorted
    BAM in transient temp + output space — so we warn up front and, on a
    write/seek failure, re-raise with an explicit out-of-space hint.
    """
    if unsorted.exists():
        src = unsorted.stat().st_size
        free = shutil.disk_usage(wd).free
        if free < _SORT_DISK_HEADROOM * src:
            log.warning(
                "Low disk for coordinate-sort: %.1f GB free vs ~%.1f GB likely "
                "needed (temp spill + output for a %.1f GB BAM). If it fails, "
                "free space and re-run — the unsorted BAM is reused, so segemehl "
                "will not re-map.",
                free / 1e9, _SORT_DISK_HEADROOM * src / 1e9, src / 1e9,
            )
    try:
        run_piped(
            ["samtools", "view", "-bh", "-F", "4", str(unsorted)],
            ["samtools", "sort", "-@", str(num_cpu), "-o", out_bam, "-"],
            cwd=wd,
        )
    except ExternalToolError as exc:
        snippet = (exc.stderr_snippet or "").lower()
        if any(marker in snippet for marker in _DISK_FULL_MARKERS):
            raise ExternalToolError(
                "samtools sort failed writing the coordinate-sorted BAM, which "
                "on a full filesystem samtools reports as 'Illegal seek'. The "
                "sort needs roughly twice the unsorted BAM in transient temp + "
                "output space.",
                tool=exc.tool, returncode=exc.returncode, cmd=exc.cmd,
                stderr_snippet=exc.stderr_snippet,
                hint=(
                    "Free space on the run-dir filesystem (or run the assemble "
                    "step on a larger disk), then re-run: segemehl reuses the "
                    "existing unsorted BAM and skips the slow re-mapping."
                ),
            ) from exc
        raise


def map_reads_segemehl(config: AssemblyConfig) -> None:
    """Map RNA-seq reads to the genome using segemehl in split/spliced mode."""
    wd = config.work_dir
    log.info("Running segemehl read mapping...")

    index = wd / _INDEX
    unsorted = wd / _UNSORTED_BAM

    # segemehl mapping is the multi-hour step. If a previous run already
    # produced a complete unsorted BAM (and only the downstream sort failed —
    # e.g. on a full disk), reuse it rather than re-mapping. quickcheck rejects
    # a truncated BAM from an interrupted segemehl, which is then re-mapped.
    if _bam_is_complete(unsorted):
        log.info(
            "Reusing %s from a previous run; skipping segemehl mapping "
            "(delete it to force a full re-map).",
            unsorted.name,
        )
    else:
        if not index.exists():
            run_cmd(["segemehl.x", "-x", str(index), "-d", str(config.genome)], cwd=wd)

        # segemehl loads the whole suffix-array index + genome into memory, so
        # it is RAM-hungry on large genomes. Write the BAM straight to disk
        # with `-b -o` instead of piping into samtools: run_cmd then checks
        # segemehl's *own* exit status, so an OOM kill, a mate-count mismatch,
        # or any other crash surfaces as a clear segemehl error rather than a
        # misleading downstream "samtools: failed to read header".
        #
        # Flags mirror the validated `segemehl-H0.bam` recipe: `-H 0` reports
        # all (not just best-scoring) alignments, capturing multi-mapping
        # spliced reads; `-S <base>` enables split/spliced mapping and pins the
        # byproduct BED files to a known location for cleanup. segemehl reads
        # gzipped FASTQ natively, so `.fastq.gz` inputs need no decompression.
        run_cmd(
            [
                "segemehl.x",
                "-H", "0",
                "-i", str(index),
                "-d", str(config.genome),
                *config.reads_args_segemehl,
                "-S", str(wd / _SPLITS_BASE),
                "-t", str(config.num_cpu),
                "-b",
                "-o", str(unsorted),
            ],
            cwd=wd,
        )

    # Mapping is done; the index and split-read BEDs are never read again
    # (junctions come from the BAM). Delete them now, *before* the sort, so
    # their several GB don't compete with the sort's temp spill + output on a
    # tight disk — a `-H 0` run plus sort scratch can need well over 10 GB.
    index.unlink(missing_ok=True)
    for suffix in _SPLITS_SUFFIXES:
        (wd / f"{_SPLITS_BASE}{suffix}").unlink(missing_ok=True)

    _coordinate_sort_and_filter(unsorted, _BAM, wd, config.num_cpu)
    run_cmd(["samtools", "index", _BAM], cwd=wd)

    # Derive a STAR-format SJ.out.tab from the BAM, then reuse STAR's
    # post-processing so the downstream hints / splice summary are identical.
    sj = sj_table_from_bam(
        wd / _BAM, config.genome, wd,
        min_intron=config.min_intron_len,
        max_intron=config.max_intron_len,
        out_name=_SJ,
    )
    generate_rnaseq_hints(
        sj, wd / _BAM, config.genome, wd,
        diagnose=config.diagnose_softclips, source_label="segemehl",
    )

    # The unsorted BAM is no longer needed once the sorted BAM + hints exist.
    unsorted.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Transcript -> genome mapping (feeds the combinr consolidation step)
# ---------------------------------------------------------------------------

_TX_INDEX = "segemehl_tx.idx"
# (query FASTA, output BAM) for each assembly. The de novo assemblies are
# SL-depleted upstream (Phase 3); genome-guided Trinity is mapped as-is.
_TRANSCRIPT_SETS: tuple[tuple[str, str], ...] = (
    ("trinity-gg.fasta", "trinity-gg.genome.bam"),
    ("trinity-denovo.sl_depleted.fasta", "trinity-denovo.genome.bam"),
    ("rnaspades.sl_depleted.fasta", "rnaspades.genome.bam"),
)
_GENOME_BAM_SUFFIX = ".genome.bam"


def _resolve_query(wd: Path, query_name: str) -> Path:
    """The transcript FASTA to map: the jaccard-clipped sibling if it exists.

    The jaccard step (``assembly/jaccard.py``) rewrites each de novo / genome-
    guided FASTA into a ``<stem>.jaccard.fasta``; prefer it so fused contigs are
    split before consolidation. Falls back to the original when clipping is off.
    """
    clipped = wd / query_name.replace(".fasta", ".jaccard.fasta")
    return clipped if clipped.exists() and clipped.stat().st_size > 0 else wd / query_name


def _map_one_transcript_set(
    config: AssemblyConfig, index: Path, query: Path, out_bam: str
) -> None:
    """Map one assembly's transcripts to the genome → sorted, indexed *out_bam*."""
    wd = config.work_dir
    final = wd / out_bam
    if _bam_is_complete(final):
        log.info("Reusing %s; skipping segemehl transcript mapping.", final.name)
        return

    stem = out_bam[: -len(_GENOME_BAM_SUFFIX)]
    unsorted = wd / f"{stem}.genome.unsorted.bam"
    splits_base = wd / f"{stem}_splits"
    if not _bam_is_complete(unsorted):
        # Same recipe as the read mapper plus `-e` (brief M-only CIGAR so combinr
        # and samtools parse standard ops): `-H 0` reports all alignments
        # (multi-copy genes map to several loci), `-S` enables split/spliced
        # mapping (transcripts span introns; trans-spliced ones split across
        # loci). One transcript == one query record, so `-q` takes the FASTA.
        run_cmd(
            [
                "segemehl.x",
                "-H", "0",
                "-e",
                "-i", str(index),
                "-d", str(config.genome),
                "-q", str(query),
                "-S", str(splits_base),
                "-t", str(config.num_cpu),
                "-b",
                "-o", str(unsorted),
            ],
            cwd=wd,
        )

    _coordinate_sort_and_filter(unsorted, out_bam, wd, config.num_cpu)
    run_cmd(["samtools", "index", out_bam], cwd=wd)
    unsorted.unlink(missing_ok=True)
    for suffix in _SPLITS_SUFFIXES:
        Path(f"{splits_base}{suffix}").unlink(missing_ok=True)


def map_transcripts_segemehl(config: AssemblyConfig) -> None:
    """Map assembled transcripts to the genome with segemehl (input to combinr).

    Produces one coordinate-sorted BAM per assembly present in the work dir:
    genome-guided Trinity, SL-depleted de novo Trinity, and (when enabled)
    SL-depleted rnaSPAdes. Each query is mapped in split/spliced mode (``-S``),
    reporting all alignments (``-H 0``) with brief CIGARs (``-e``). The genome
    index is built once and reused across queries. Per-query resume: a BAM that
    passes ``samtools quickcheck`` is left untouched.
    """
    wd = config.work_dir
    sets = [
        (query, out_bam)
        for query_name, out_bam in _TRANSCRIPT_SETS
        if (query := _resolve_query(wd, query_name)).exists() and query.stat().st_size > 0
    ]
    if not sets:
        log.warning("No assembled transcripts found to map; skipping.")
        return

    index = wd / _TX_INDEX
    if not index.exists():
        run_cmd(["segemehl.x", "-x", str(index), "-d", str(config.genome)], cwd=wd)
    try:
        for query, out_bam in sets:
            log.info("segemehl mapping transcripts %s -> %s ...", query.name, out_bam)
            _map_one_transcript_set(config, index, query, out_bam)
    finally:
        # The genome index is regenerable; drop it so it doesn't linger in the
        # run dir (a partial run rebuilds it cheaply next time).
        index.unlink(missing_ok=True)
