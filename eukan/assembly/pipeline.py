"""Assembly pipeline: read mapping → de novo + genome-guided assembly → SL
trans-splice cut → combinr consolidation."""

from __future__ import annotations

from eukan.assembly.combinr import run_combinr
from eukan.assembly.jaccard import run_jaccard
from eukan.assembly.rnaspades import run_rnaspades
from eukan.assembly.segemehl import map_reads_segemehl
from eukan.assembly.sl_acceptors import detect_sl_acceptors
from eukan.assembly.sl_cut import run_sl_cut
from eukan.assembly.star import map_reads, map_transcripts_star
from eukan.assembly.stringtie import run_stringtie
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
    """Assembly steps: <aligner> → trinity (de novo) → stringtie (genome-guided)
    → rnaspades → jaccard → map_transcripts → sl_detect → sl_cut → combinr."""
    return [
        _aligner_step(aligner),
        StepSpec("trinity", run_trinity, "trinity-denovo.fasta", "-T / --run-trinity"),
        StepSpec("stringtie", run_stringtie, "stringtie.gtf", "--run-stringtie"),
        StepSpec("rnaspades", run_rnaspades, "rnaspades.fasta", "--run-rnaspades"),
        # No declared output: jaccard rewrites each transcript FASTA into a
        # ``.jaccard.fasta`` sibling, but on single-end input (or zero clips) it
        # legitimately writes nothing, so stale-output validation must not fire.
        StepSpec("jaccard", run_jaccard, None, "--run-jaccard"),
        StepSpec(
            "map_transcripts", map_transcripts_star,
            "trinity-denovo.genome.bam", "--run-map-transcripts",
        ),
        # sl_detect/sl_cut have no declared output: with no SL signal sl_detect
        # writes a header-only sl_acceptors.gff3 (zero features) and sl_cut is a
        # pass-through, so stale-output GFF validation must not fire on either.
        StepSpec("sl_detect", detect_sl_acceptors, None, "--run-sl-detect"),
        StepSpec("sl_cut", run_sl_cut, None, "--run-sl-cut"),
        StepSpec(
            "combinr", run_combinr,
            Artifact.NR_TRANSCRIPTS_FASTA.value, "--run-combinr",
        ),
    ]


def force_steps_from_run_flags(
    *,
    aligner: str = "star",
    run_star: bool = False,
    run_segemehl: bool = False,
    run_trinity: bool = False,
    run_stringtie: bool = False,
    run_rnaspades: bool = False,
    run_jaccard: bool = False,
    run_map_transcripts: bool = False,
    run_sl_detect: bool = False,
    run_sl_cut: bool = False,
    run_combinr: bool = False,
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
        run_trinity=run_trinity, run_stringtie=run_stringtie,
        run_rnaspades=run_rnaspades, run_jaccard=run_jaccard,
        run_map_transcripts=run_map_transcripts,
        run_sl_detect=run_sl_detect, run_sl_cut=run_sl_cut,
        run_combinr=run_combinr,
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
        skip=lambda s: (s.name == "rnaspades" and not config.rnaspades)
        or (s.name == "jaccard" and not config.jaccard_clip),
    )
