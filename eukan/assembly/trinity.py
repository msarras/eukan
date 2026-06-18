"""Trinity genome-guided and de novo assembly."""

from __future__ import annotations

import shutil

from eukan.infra.logging import get_logger
from eukan.infra.runner import run_cmd
from eukan.settings import AssemblyConfig

log = get_logger(__name__)


def _run_trinity_mode(
    config: AssemblyConfig,
    *,
    prefix: str,
    cleanup_name: str,
    log_message: str,
    mode_args: list[str],
) -> None:
    """Run one Trinity mode and normalize its output to ``<prefix>.fasta``.

    Skips when ``<prefix>.fasta`` already exists. *mode_args* carries the
    mode-specific flags (genome-guided BAM vs de-novo reads); the shared
    memory/CPU/cleanup/strand flags are added here. Handles both the
    ``--full_cleanup`` output (``<prefix>.<cleanup_name>`` beside the dir) and
    the no-cleanup layout (``<prefix>/<cleanup_name>`` inside it), then removes
    the working dir.

    Jaccard clipping is *not* delegated to Trinity (``--jaccard_clip``): the
    standalone :mod:`eukan.assembly.jaccard` step clips every assembly uniformly
    (including rnaSPAdes, which Trinity cannot), so passing it here too would
    double-clip Trinity's contigs.
    """
    wd = config.work_dir
    final = wd / f"{prefix}.fasta"
    if final.exists():
        return

    log.info(log_message)
    lib_type_args = (
        ["--SS_lib_type", config.strand_specific] if config.strand_specific else []
    )
    run_cmd(
        [
            "Trinity",
            *mode_args,
            "--max_memory", f"{config.memory_gb}G",
            "--CPU", str(config.num_cpu),
            "--full_cleanup",
            "--output", prefix,
            *lib_type_args,
        ],
        cwd=wd,
    )
    # --full_cleanup puts output at <prefix>.<cleanup_name> beside the dir;
    # without it the file is <prefix>/<cleanup_name> inside the dir.
    produced = wd / f"{prefix}.{cleanup_name}"
    if not produced.exists():
        produced = wd / prefix / cleanup_name
    if produced.exists():
        shutil.move(str(produced), str(final))
    shutil.rmtree(wd / prefix, ignore_errors=True)


def run_trinity(config: AssemblyConfig) -> None:
    """Run genome-guided and de novo Trinity assembly."""
    _run_trinity_mode(
        config,
        prefix="trinity-gg",
        cleanup_name="Trinity-GG.fasta",
        log_message="Running genome-guided Trinity assembly...",
        mode_args=[
            "--genome_guided_bam", config.aligner_bam,
            "--genome_guided_max_intron", str(config.max_intron_len),
        ],
    )
    _run_trinity_mode(
        config,
        prefix="trinity-denovo",
        cleanup_name="Trinity.fasta",
        log_message="Running de novo Trinity assembly...",
        mode_args=["--seqType", "fq", *config.reads_args_trinity],
    )
