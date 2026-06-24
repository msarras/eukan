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
    (with STAR rather than Trinity's slower bowtie2 pass, and with the tunable
    ``--jaccard-*`` knobs), so passing it here too would double-clip Trinity's
    contigs.

    ``--no_salmon`` skips Trinity's final salmon-based expression filtering of
    isoforms. bioconda's Trinity 2.15.2 pulls salmon 2.x (the Rust rewrite),
    whose CLI dropped the ``--minAssignedFrags``/``--validateMappings`` flags
    Trinity still passes, so the filter step errors out on every platform; the
    C++ salmon 1.x that *does* accept them is compiled with AVX2 and SIGILLs on
    pre-Haswell CPUs. Skipping it lets Trinity finish; the combinr consolidation
    step downstream removes the redundant isoforms the filter would have.
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
            "--no_salmon",
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
    """Run genome-guided and de novo Trinity assembly.

    Both modes emit transcript-coordinate FASTAs (``trinity-gg.fasta`` and
    ``trinity-denovo.fasta``); the genome-guided BAM only *clusters* reads per
    locus, so like the de novo set the result is mapped back to the genome by
    :func:`eukan.assembly.star.map_transcripts`. The two sets overlap heavily
    (same reads) — combinr consolidates the redundancy downstream.
    """
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
