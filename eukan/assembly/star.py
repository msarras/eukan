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
