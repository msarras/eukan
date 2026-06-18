"""STAR read mapping and hint generation."""

from __future__ import annotations

import shutil
from pathlib import Path

from eukan.assembly.align_hints import generate_rnaseq_hints
from eukan.exceptions import ExternalToolError
from eukan.infra.logging import get_logger
from eukan.infra.runner import run_cmd
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

# STAR resource/index tuning for typical small eukaryotic genomes.
_STAR_SA_INDEX_NBASES = "3"                 # suffix-array pre-index size (small genomes)
_STAR_GENOME_GENERATE_RAM = "40317074816"   # byte cap for --limitGenomeGenerateRAM (~37 GiB)
_STAR_BAM_SORT_RAM = "27643756136"          # byte cap for --limitBAMsortRAM (~26 GiB)
# --outSJfilterIntronMaxVsReadN: max intron length allowed for junctions
# supported by 1, 2, 3+ reads respectively.
_STAR_SJ_FILTER_INTRON_MAX_VS_READN = ("100", "300", "500")

_BAM = "STAR_Aligned.sortedByCoord.out.bam"
_SJ = "STAR_SJ.out.tab"


def _is_gzipped(path: Path) -> bool:
    """Check if a file is gzip-compressed by reading the magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def map_reads(config: AssemblyConfig) -> None:
    """Map RNA-seq reads to the genome using STAR."""
    wd = config.work_dir
    log.info("Running STAR read mapping...")

    index_dir = wd / "build-index"

    # Build genome index
    if not index_dir.exists():
        index_dir.mkdir()
        run_cmd(
            [
                "STAR",
                "--genomeSAindexNbases", _STAR_SA_INDEX_NBASES,
                "--limitGenomeGenerateRAM", _STAR_GENOME_GENERATE_RAM,
                "--runThreadN", str(config.num_cpu),
                "--runMode", "genomeGenerate",
                "--genomeDir", str(index_dir),
                "--genomeFastaFiles", str(config.genome),
            ],
            cwd=wd,
        )

    # Detect compressed input
    reads = config.reads_args_star
    zcat_args = []
    first_read_file = Path(reads[0])
    if first_read_file.suffix in (".gz", ".gzip") or _is_gzipped(first_read_file):
        zcat_args = ["--readFilesCommand", "zcat"]

    quality_args = ["--outQSconversionAdd", "-31"] if config.phred_quality == 64 else []

    max_intron_args = (
        ["--alignIntronMax", str(config.max_intron_len)]
        if config.max_intron_len
        else []
    )

    star_cmd = [
        "STAR",
        "--runThreadN", str(config.num_cpu),
        "--genomeDir", str(index_dir),
        "--alignEndsType", config.align_mode,
        "--readFilesIn", *reads,
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outSJfilterIntronMaxVsReadN", *_STAR_SJ_FILTER_INTRON_MAX_VS_READN,
        "--alignIntronMin", str(config.min_intron_len),
        *max_intron_args,
        "--outFileNamePrefix", "STAR_",
        "--outSAMattributes", "All",
        "--outSAMattrIHstart", "0",
        "--outSAMstrandField", "intronMotif",
        "--limitBAMsortRAM", _STAR_BAM_SORT_RAM,
        *zcat_args,
        *quality_args,
    ]

    try:
        run_cmd(star_cmd, cwd=wd)
    except ExternalToolError:
        log.warning("STAR failed, falling back to STARlong")
        star_long_cmd = ["STARlong", *star_cmd[1:]]
        run_cmd(star_long_cmd, cwd=wd)

    # Report mapping rate from STAR log
    _log_mapping_rate(wd)

    # Splice junctions → hints + splice summary + diagnostic (shared with segemehl).
    generate_rnaseq_hints(
        wd / _SJ, wd / _BAM, config.genome, wd,
        diagnose=config.diagnose_softclips, source_label="STAR",
    )

    # Cleanup build index (large)
    shutil.rmtree(index_dir, ignore_errors=True)


def map_transcripts_star(config: AssemblyConfig) -> None:
    """Map de novo assembled transcripts to the genome UNGAPPED with STAR.

    The transcripts are the spliced product of ungapped reads, so they are mapped
    with ``--alignIntronMax 1`` (no introns/splits — the spliced leader is not
    split off to the SL-RNA locus, and memory stays bounded) and
    ``--alignEndsType Local`` so the leader SOFT-CLIPS at the trans-splice acceptor
    — the signal SL detection (:mod:`eukan.assembly.sl_acceptors`) keys on.
    (segemehl ``-S`` both split the leader away and exploded memory; segemehl
    without ``-S`` emits no soft-clips at all, so STAR ungapped-Local is the only
    aligner that is OOM-safe *and* preserves the clip.)

    Produces one coordinate-sorted, indexed ``<stem>.genome.bam`` per de novo
    assembly present, with unmapped transcripts saved to
    ``<stem>.unmapped_transcripts.fasta``. Long transcripts fall back to STARlong.
    Per-query resume: a BAM passing ``samtools quickcheck`` is left untouched.
    """
    from eukan.assembly.jaccard import _chr_bin_nbits, _genome_stats, _sa_index_nbases
    from eukan.assembly.segemehl import _GENOME_BAM_SUFFIX, _TRANSCRIPT_SETS, _resolve_query

    wd = config.work_dir
    sets = [
        (query, out_bam)
        for query_name, out_bam in _TRANSCRIPT_SETS
        if (query := _resolve_query(wd, query_name)).exists() and query.stat().st_size > 0
    ]
    if not sets:
        log.warning("No assembled transcripts found to map; skipping.")
        return

    index_dir = wd / "star_tx_index"
    total_len, n_seqs = _genome_stats(config.genome)
    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir()
    run_cmd(
        [
            "STAR",
            "--runMode", "genomeGenerate",
            "--genomeDir", str(index_dir),
            "--genomeFastaFiles", str(config.genome),
            "--genomeSAindexNbases", _sa_index_nbases(total_len),
            "--genomeChrBinNbits", _chr_bin_nbits(total_len, n_seqs),
            "--limitGenomeGenerateRAM", _STAR_GENOME_GENERATE_RAM,
            "--runThreadN", str(config.num_cpu),
        ],
        cwd=wd,
    )
    try:
        for query, out_bam in sets:
            log.info("STAR mapping transcripts %s -> %s ...", query.name, out_bam)
            _star_map_one_transcript_set(config, index_dir, query, out_bam, _GENOME_BAM_SUFFIX)
    finally:
        # The index is regenerable; drop it so it doesn't linger in the run dir.
        shutil.rmtree(index_dir, ignore_errors=True)


def _star_map_one_transcript_set(
    config: AssemblyConfig, index_dir: Path, query: Path, out_bam: str, bam_suffix: str
) -> None:
    """Map one transcript FASTA to the genome index → sorted, indexed *out_bam*."""
    from eukan.assembly.segemehl import _bam_is_complete

    wd = config.work_dir
    final = wd / out_bam
    if _bam_is_complete(final):
        log.info("Reusing %s; skipping STAR transcript mapping.", final.name)
        return

    stem = out_bam[: -len(bam_suffix)]
    prefix = f"startx_{stem}_"
    zcat_args = ["--readFilesCommand", "zcat"] if _is_gzipped(query) else []
    star_cmd = [
        "STAR",
        "--runThreadN", str(config.num_cpu),
        "--genomeDir", str(index_dir),
        "--readFilesIn", str(query),
        "--alignEndsType", "Local",   # soft-clip the spliced leader at the acceptor
        "--alignIntronMax", "1",      # ungapped: no intron/SL split, bounded memory
        "--alignMatesGapMax", "1",
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outReadsUnmapped", "Fastx",
        "--outFileNamePrefix", prefix,
        "--limitBAMsortRAM", _STAR_BAM_SORT_RAM,
        *zcat_args,
    ]
    try:
        run_cmd(star_cmd, cwd=wd)
    except ExternalToolError:
        log.warning("STAR failed mapping %s, retrying with STARlong.", query.name)
        run_cmd(["STARlong", *star_cmd[1:]], cwd=wd)

    (wd / f"{prefix}Aligned.sortedByCoord.out.bam").rename(final)
    run_cmd(["samtools", "index", out_bam], cwd=wd)
    unmapped = wd / f"{prefix}Unmapped.out.mate1"
    if unmapped.exists():
        unmapped.rename(wd / f"{stem}.unmapped_transcripts.fasta")


def _log_mapping_rate(wd: Path) -> None:
    """Parse STAR Log.final.out and log the overall mapping rate."""
    log_file = wd / "STAR_Log.final.out"
    if not log_file.exists():
        return

    unique_pct = 0.0
    multi_pct = 0.0
    for line in log_file.read_text().splitlines():
        if "Uniquely mapped reads %" in line:
            unique_pct = float(line.split("|")[-1].strip().rstrip("%"))
        elif "% of reads mapped to multiple loci" in line:
            multi_pct = float(line.split("|")[-1].strip().rstrip("%"))

    total_pct = unique_pct + multi_pct
    if total_pct < 75:
        log.warning(
            "Low read mapping rate: %.1f%% (%.1f%% unique, %.1f%% multi) "
            "— check genome/reads compatibility",
            total_pct, unique_pct, multi_pct,
        )
    else:
        log.info(
            "Read mapping rate: %.1f%% (%.1f%% unique, %.1f%% multi)",
            total_pct, unique_pct, multi_pct,
        )
