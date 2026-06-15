"""Assembly pipeline: STAR mapping → Trinity assembly → PASA alignment."""

from __future__ import annotations

from eukan.assembly.pasa import run_pasa
from eukan.assembly.rnaspades import run_rnaspades
from eukan.assembly.segemehl import map_reads_segemehl
from eukan.assembly.star import map_reads
from eukan.assembly.trinity import run_trinity
from eukan.infra.artifacts import Artifact
from eukan.infra.manifest import ASSEMBLY
from eukan.infra.pipeline import (
    StepSpec,
    run_simple_pipeline,
)
from eukan.infra.pipeline import (
    force_steps_from_run_flags as _force_steps_from_run_flags,
)
from eukan.settings import AssemblyConfig


def _aligner_step(aligner: str) -> StepSpec:
    """The read-mapping StepSpec for the selected aligner."""
    if aligner == "segemehl":
        return StepSpec(
            "segemehl", map_reads_segemehl,
            "segemehl_Aligned.sortedByCoord.out.bam", "--run-segemehl",
        )
    return StepSpec(
        "star", map_reads,
        "STAR_Aligned.sortedByCoord.out.bam", "-A / --run-star",
    )


def _steps_for(aligner: str) -> list[StepSpec]:
    """Assembly steps for the aligner: <aligner> → trinity → rnaspades → pasa."""
    return [
        _aligner_step(aligner),
        StepSpec("trinity", run_trinity, "trinity-gg.fasta", "-T / --run-trinity"),
        StepSpec("rnaspades", run_rnaspades, "rnaspades.fasta", "--run-rnaspades"),
        StepSpec("pasa", run_pasa, Artifact.NR_TRANSCRIPTS_FASTA.value, "-P / --run-pasa"),
    ]


def force_steps_from_run_flags(
    *,
    aligner: str = "star",
    run_star: bool = False,
    run_segemehl: bool = False,
    run_trinity: bool = False,
    run_rnaspades: bool = False,
    run_pasa: bool = False,
    force: bool = False,
) -> list[str]:
    """Translate ``--run-X`` / ``--force`` flags into manifest keys to force.

    The inactive aligner's flag is a harmless no-op (its step is not in the
    selected step list).
    """
    return _force_steps_from_run_flags(
        ASSEMBLY, _steps_for(aligner),
        force=force,
        run_star=run_star, run_segemehl=run_segemehl,
        run_trinity=run_trinity, run_rnaspades=run_rnaspades, run_pasa=run_pasa,
    )


def run_assembly(
    config: AssemblyConfig,
    *,
    force_steps: list[str] | None = None,
) -> None:
    """Run the assembly pipeline with manifest tracking."""
    run_simple_pipeline(
        ASSEMBLY, _steps_for(config.aligner), config,
        force_steps=force_steps,
        skip=lambda s: s.name == "rnaspades" and not config.rnaspades,
    )
