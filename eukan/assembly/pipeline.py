"""Assembly pipeline: read mapping → de novo + genome-guided assembly → SL
trans-splice cut → combinr consolidation."""

from __future__ import annotations

from pathlib import Path

from eukan.assembly.combinr import run_combinr
from eukan.assembly.jaccard import run_jaccard
from eukan.assembly.rnaspades import run_rnaspades
from eukan.assembly.segemehl import map_reads_segemehl
from eukan.assembly.sl_acceptors import detect_sl_acceptors
from eukan.assembly.sl_cut import run_sl_cut
from eukan.assembly.star import map_reads, map_reads_auto, map_transcripts
from eukan.assembly.strand_correction import run_strand_correction
from eukan.assembly.stringtie import run_stringtie
from eukan.infra.artifacts import Artifact
from eukan.infra.manifest import ASSEMBLY, step_key
from eukan.infra.pipeline import (
    StepSpec,
    run_simple_pipeline,
)
from eukan.settings import AssemblyConfig

# ---------------------------------------------------------------------------
# Per-step input resolvers (for fingerprint-based resume invalidation)
# ---------------------------------------------------------------------------
# Each names the files a step consumes from upstream steps; the driver records
# their fingerprint on completion and re-runs the step on resume if they change
# (eukan/infra/steps.py::fingerprint_inputs). This is what makes a non-forced
# resume self-heal when, say, the assembly is rebuilt: map_transcripts re-maps
# the new transcripts, and the changed BAM cascades through strand_correct →
# sl_detect → sl_cut → combinr, each re-running as its input flips.


def _resolved_transcript_queries(config: AssemblyConfig) -> list[Path]:
    """The transcript FASTAs map_transcripts maps (the jaccard sibling if present)."""
    from eukan.assembly.segemehl import _TRANSCRIPT_SETS, _resolve_query

    return [_resolve_query(config.work_dir, name) for name, _ in _TRANSCRIPT_SETS]


def _denovo_genome_bams(config: AssemblyConfig) -> list[Path]:
    """The de novo transcript→genome BAMs map_transcripts produces."""
    from eukan.assembly.sl_cut import _DENOVO_BAMS

    return [config.work_dir / b for b in _DENOVO_BAMS]


def _strand_correct_inputs(config: AssemblyConfig) -> list[Path]:
    from eukan.assembly.jaccard import resolve_stringtie_models

    return [*_denovo_genome_bams(config), resolve_stringtie_models(config.work_dir)]


def _sl_detect_inputs(config: AssemblyConfig) -> list[Path]:
    # The read aligner BAM also feeds SL detection, but it is large and only
    # changes when the aligner is re-run (which cascades via the run flags), so
    # only the de novo BAMs — the map_transcripts result — are fingerprinted.
    return _denovo_genome_bams(config)


def _sl_cut_inputs(config: AssemblyConfig) -> list[Path]:
    from eukan.assembly.jaccard import resolve_stringtie_models
    from eukan.assembly.strand_correction import (
        _DENOVO_GFF3,
        _DENOVO_STRANDED,
        _STRINGTIE_STRANDED,
    )

    wd = config.work_dir
    return [
        wd / _STRINGTIE_STRANDED, resolve_stringtie_models(wd),
        wd / _DENOVO_STRANDED, wd / _DENOVO_GFF3,
        wd / Artifact.SL_ACCEPTORS.value,
    ]


def _combinr_inputs(config: AssemblyConfig) -> list[Path]:
    from eukan.assembly.combinr import _CUT_MODELS

    return [config.work_dir / m for m in _CUT_MODELS]


# Scalar (non-file) inputs folded into a step's resume fingerprint, so changing the
# value re-runs the step even when its input files are byte-identical. ``-M`` is
# enforced post-mapping (StringTie reads a bounded BAM; sl_cut splits models;
# combinr passes --max-intron), so tightening it must re-run exactly those steps —
# but NOT the read/transcript mappers (segemehl ignores --max-intron natively, so a
# scalar there would force a multi-hour re-map for an identical BAM).


def _max_intron_scalar(config: AssemblyConfig) -> list[str]:
    return [f"max_intron_len={config.max_intron_len}"]


def _stringtie_scalars(config: AssemblyConfig) -> list[str]:
    # StringTie reads a max-intron-bounded BAM and runs at a configurable -c/-f
    # stringency; changing any of these re-assembles the genome-guided set.
    return [
        f"max_intron_len={config.max_intron_len}",
        f"stringtie_min_coverage={config.stringtie_min_coverage}",
        f"stringtie_min_isoform_fraction={config.stringtie_min_isoform_fraction}",
    ]


def _sl_cut_scalars(config: AssemblyConfig) -> list[str]:
    return [
        f"max_intron_len={config.max_intron_len}",
        f"min_sl_fragment={config.min_sl_fragment}",
    ]


def _aligner_step(aligner: str) -> StepSpec:
    """The read-mapping StepSpec for the selected aligner.

    ``auto`` (default) and ``star`` share the ``star`` step (STAR runs first either
    way); ``auto`` additionally re-maps with segemehl when non-canonical splicing is
    extensive (see :func:`star.map_reads_auto`). ``segemehl`` maps with segemehl only.
    """
    if aligner == "segemehl":
        return StepSpec(
            "segemehl", map_reads_segemehl,
            "segemehl_Aligned.sortedByCoord.out.bam", "--run-segemehl",
        )
    return StepSpec(
        "star", map_reads_auto if aligner == "auto" else map_reads,
        "STAR_Aligned.sortedByCoord.out.bam", "-A / --run-star",
    )


def _steps_for(aligner: str) -> list[StepSpec]:
    """Assembly steps: <aligner> → stringtie (genome-guided) → rnaspades (de novo)
    → jaccard → map_transcripts → strand_correct → sl_detect → sl_cut → combinr."""
    return [
        _aligner_step(aligner),
        # stringtie reads a max-intron-bounded copy of the segemehl read BAM, so a
        # changed -M must re-run it (scalars); the bounded BAM is built in-step.
        StepSpec(
            "stringtie", run_stringtie, "stringtie.gtf", "--run-stringtie",
            scalars=_stringtie_scalars,
        ),
        StepSpec("rnaspades", run_rnaspades, "rnaspades.fasta", "--run-rnaspades"),
        # No declared output: jaccard rewrites each transcript FASTA into a
        # ``.jaccard.fasta`` sibling, but on single-end input (or zero clips) it
        # legitimately writes nothing, so stale-output validation must not fire.
        # Having no declared output it always re-runs on resume, so a changed
        # ``jaccard_greediness`` re-clips and cascades to map_transcripts (which
        # fingerprints the rewritten ``.jaccard.fasta``) — a scalar here would be
        # inert, since is_step_complete never reaches the fingerprint check.
        StepSpec("jaccard", run_jaccard, None, "--run-jaccard"),
        StepSpec(
            "map_transcripts", map_transcripts,
            "rnaspades.genome.bam", "--run-map-transcripts",
            inputs=_resolved_transcript_queries,
        ),
        # No declared output: strand_correct always converts the de novo BAM to
        # rnaspades.genome.gff3 (for the SL cut), but only writes the *.stranded.gff3
        # homology-corrected models when --uniprot is given on an unstranded library,
        # so stale-output validation must not fire when it's a no-op.
        StepSpec(
            "strand_correct", run_strand_correction, None, "--run-strand-correct",
            inputs=_strand_correct_inputs,
        ),
        # sl_detect/sl_cut have no declared output: with no SL signal sl_detect
        # writes a header-only sl_acceptors.gff3 (zero features) and sl_cut is a
        # pass-through, so stale-output GFF validation must not fire on either.
        StepSpec(
            "sl_detect", detect_sl_acceptors, None, "--run-sl-detect",
            inputs=_sl_detect_inputs,
        ),
        StepSpec(
            "sl_cut", run_sl_cut, None, "--run-sl-cut",
            inputs=_sl_cut_inputs, scalars=_sl_cut_scalars,
        ),
        StepSpec(
            "combinr", run_combinr,
            Artifact.NR_TRANSCRIPTS_FASTA.value, "--run-combinr",
            inputs=_combinr_inputs, scalars=_max_intron_scalar,
        ),
    ]


# Within-pipeline data dependencies: re-running a step (via its --run-* flag)
# invalidates every step that consumes its output, so those are forced too. The
# read aligner is cascaded because StringTie (genome-guided) and SL read-side
# detection read its BAM — and in 'auto' mode that BAM may switch from STAR to
# segemehl. Each edge below is "producer -> direct consumers".
_DOWNSTREAM: dict[str, tuple[str, ...]] = {
    "star": ("stringtie", "sl_detect"),
    "segemehl": ("stringtie", "sl_detect"),
    "stringtie": ("strand_correct", "sl_cut"),
    "rnaspades": ("jaccard",),
    # jaccard now also clips the StringTie GTF -> stringtie.jaccard.gff3, which
    # strand_correct/sl_cut read, so re-running it must re-run strand_correct too
    # (map_transcripts already reaches it, but the edge is now direct).
    "jaccard": ("map_transcripts", "strand_correct"),
    "map_transcripts": ("strand_correct", "sl_detect"),
    "strand_correct": ("sl_cut",),
    "sl_detect": ("sl_cut",),
    "sl_cut": ("combinr",),
}


def _expand_downstream(selected: set[str]) -> set[str]:
    """*selected* plus the transitive closure of their ``_DOWNSTREAM`` dependents."""
    out = set(selected)
    stack = list(selected)
    while stack:
        for dep in _DOWNSTREAM.get(stack.pop(), ()):
            if dep not in out:
                out.add(dep)
                stack.append(dep)
    return out


def force_steps_from_run_flags(
    *,
    aligner: str = "star",
    run_star: bool = False,
    run_segemehl: bool = False,
    run_stringtie: bool = False,
    run_rnaspades: bool = False,
    run_jaccard: bool = False,
    run_map_transcripts: bool = False,
    run_strand_correct: bool = False,
    run_sl_detect: bool = False,
    run_sl_cut: bool = False,
    run_combinr: bool = False,
    force: bool = False,
) -> list[str]:
    """Translate ``--run-X`` / ``--force`` flags into manifest keys to force.

    ``--force`` re-runs every step. An individual ``--run-X`` re-runs that step
    *and every downstream step that consumes its output* (``_DOWNSTREAM``): e.g.
    ``--run-map-transcripts`` re-maps the de novo transcripts and then re-runs
    strand_correct, sl_detect, sl_cut, and combinr, which read the new BAM
    (directly or transitively). The inactive aligner's flag is a harmless no-op.
    Returns keys in pipeline order.
    """
    steps = _steps_for(aligner)
    valid = {s.name for s in steps}
    flags = {
        "star": run_star, "segemehl": run_segemehl, "stringtie": run_stringtie,
        "rnaspades": run_rnaspades, "jaccard": run_jaccard,
        "map_transcripts": run_map_transcripts,
        "strand_correct": run_strand_correct,
        "sl_detect": run_sl_detect, "sl_cut": run_sl_cut, "combinr": run_combinr,
    }
    selected = {name for name, on in flags.items() if on and name in valid}
    # Individual --run-X flags take precedence over --force (scope to the
    # selected steps + their downstream); --force alone re-runs everything.
    if selected:
        expanded = _expand_downstream(selected)
        return [step_key(ASSEMBLY, s.name) for s in steps if s.name in expanded]
    if force:
        return [step_key(ASSEMBLY, s.name) for s in steps]
    return []


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
