"""eukan assemble — transcriptome assembly pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from click_option_group import optgroup

from eukan.cli._framework import (
    PASA_CODE_TABLE,
    PreformattedEpilogCommand,
    drop_none,
    force_option,
    genome_option,
    numcpu_option,
    resolve_optional_path,
)


@click.command(cls=PreformattedEpilogCommand, epilog=PASA_CODE_TABLE)
@optgroup.group("Required input")
@genome_option("Genome FASTA file.")
@optgroup.option("--left", "-l", type=click.Path(exists=True, path_type=Path), help="Left paired-end reads.")
@optgroup.option("--right", "-r", type=click.Path(exists=True, path_type=Path), help="Right paired-end reads.")
@optgroup.option("--single", "-s", type=click.Path(exists=True, path_type=Path), help="Single-end reads.")
@optgroup.group("Pipeline parameters")
@numcpu_option
@optgroup.option(
    "--strand-specific", "-S", type=click.Choice(["RF", "FR", "R", "F"]), default=None,
    help="Strand-specific library type.",
)
@optgroup.option(
    "--aligner", type=click.Choice(["star", "segemehl"]),
    default="star", show_default=True,
    help="Read aligner. segemehl is splice-agnostic (captures non-canonical "
    "splice sites STAR misses); STAR is the default.",
)
@optgroup.option(
    "--align-mode", "-t", type=click.Choice(["EndToEnd", "Local"]),
    default="Local", show_default=True,
    help="STAR read alignment mode (end-to-end vs soft-clipped local). "
    "STAR only; ignored when --aligner segemehl.",
)
@optgroup.option(
    "--splice-permissive", is_flag=True, default=False,
    help="Allow non-canonical splice sites (GC-AG, AT-AC). "
    "Sets PASA splice boundary stringency to 0 and retains non-canonical junctions.",
)
@optgroup.option(
    "--diagnose-softclips/--no-diagnose-softclips", default=True,
    show_default=True,
    help="Run the soft-clip + intron diagnostic after STAR. "
    "Detects trans-splicing (via de novo splice-leader clusters) and "
    "non-canonical splice prevalence; surfaces both as INFO/WARNING.",
)
@optgroup.option(
    "--code", "-c",
    type=click.Choice(["1", "6", "10", "12"]),
    default="1", show_default=True,
    help="NCBI genetic code for PASA. Supported: 1=standard, 6=Tetrahymena, 10=Euplotes, 12=Candida.",
)
@optgroup.option("--min-intron", "-m", type=int, default=20, show_default=True, help="Minimum intron length.")
@optgroup.option("--max-intron", "-M", type=int, default=5000, show_default=True, help="Maximum intron length.")
@optgroup.option("--phred", type=click.Choice(["33", "64"]), default="33", show_default=True, help="Phred quality score.")
@optgroup.option("--jaccard-clip", "-j", is_flag=True, help="Enable jaccard clipping.")
@optgroup.option(
    "--rnaspades/--no-rnaspades", default=True, show_default=True,
    help="Run rnaSPAdes de novo assembly alongside Trinity (consolidated by combinr).",
)
@optgroup.option(
    "--memory-gb", type=int, default=None,
    help="Trinity --max_memory cap in GiB. Defaults to 60 percent of "
         "currently-available memory (floored at 4 GiB).",
)
@optgroup.group("Re-run steps")
@optgroup.option("--run-star", "-A", is_flag=True, help="Force re-run STAR read mapping.")
@optgroup.option("--run-segemehl", is_flag=True, help="Force re-run segemehl read mapping.")
@optgroup.option("--run-trinity", "-T", is_flag=True, help="Force re-run Trinity assembly.")
@optgroup.option("--run-rnaspades", is_flag=True, help="Force re-run rnaSPAdes assembly.")
@optgroup.option("--run-sl-deplete", is_flag=True, help="Force re-run SL depletion.")
@optgroup.option(
    "--run-map-transcripts", is_flag=True,
    help="Force re-run segemehl transcript→genome mapping.",
)
@optgroup.option("--run-pasa", "-P", is_flag=True, help="Force re-run PASA alignment.")
@force_option
def assemble(
    genome: Path,
    left: Path | None,
    right: Path | None,
    single: Path | None,
    min_intron: int,
    max_intron: int,
    phred: str,
    numcpu: int,
    strand_specific: str | None,
    aligner: str,
    align_mode: str,
    run_star: bool,
    run_segemehl: bool,
    run_trinity: bool,
    run_rnaspades: bool,
    run_sl_deplete: bool,
    run_map_transcripts: bool,
    run_pasa: bool,
    jaccard_clip: bool,
    rnaspades: bool,
    splice_permissive: bool,
    diagnose_softclips: bool,
    code: str,
    memory_gb: int | None,
    force: bool,
) -> None:
    """Assemble transcriptome from RNA-seq reads.

    \b
    Provide either paired-end reads (--left and --right together) or
    single-end reads (--single). If using paired-end reads, both --left
    and --right are required.
    """
    from eukan.assembly import run_assembly
    from eukan.assembly.pipeline import force_steps_from_run_flags
    from eukan.infra.layout import step_work_dir
    from eukan.settings import AssemblyConfig

    if not left and not right and not single:
        raise click.UsageError("Provide --left/--right (paired) or --single reads.")
    if (left or right) and not (left and right):
        raise click.UsageError("Paired-end mode requires both --left and --right.")

    if strand_specific:
        if single and strand_specific in ("RF", "FR"):
            raise click.UsageError(
                "Paired-end strand types (RF/FR) cannot be used with single-end reads."
            )
        if (left or right) and strand_specific in ("R", "F"):
            raise click.UsageError(
                "Single-end strand types (R/F) cannot be used with paired-end reads."
            )

    if memory_gb is not None and memory_gb < 1:
        raise click.UsageError("--memory-gb must be at least 1 GiB.")

    config = AssemblyConfig(**drop_none(
        genome=genome.resolve(),
        work_dir=step_work_dir("assemble"),
        manifest_dir=Path.cwd(),
        min_intron_len=min_intron,
        max_intron_len=max_intron,
        phred_quality=int(phred),
        num_cpu=numcpu,
        aligner=aligner,
        align_mode=align_mode,
        jaccard_clip=jaccard_clip,
        rnaspades=rnaspades,
        splice_permissive=splice_permissive,
        diagnose_softclips=diagnose_softclips,
        genetic_code=code,
        left_reads=resolve_optional_path(left),
        right_reads=resolve_optional_path(right),
        single_reads=resolve_optional_path(single),
        strand_specific=strand_specific,
        memory_gb=memory_gb,
    ))

    force_steps = force_steps_from_run_flags(
        aligner=aligner,
        run_star=run_star, run_segemehl=run_segemehl,
        run_trinity=run_trinity, run_rnaspades=run_rnaspades,
        run_sl_deplete=run_sl_deplete, run_map_transcripts=run_map_transcripts,
        run_pasa=run_pasa, force=force,
    )
    run_assembly(config, force_steps=force_steps or None)
    click.echo("Done.")
