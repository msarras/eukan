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

from eukan.assembly.align_hints import generate_rnaseq_hints, sj_table_from_bam
from eukan.infra.logging import get_logger
from eukan.infra.runner import run_cmd, run_piped
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

_BAM = "segemehl_Aligned.sortedByCoord.out.bam"
_SJ = "segemehl_SJ.out.tab"
_INDEX = "segemehl.idx"


def map_reads_segemehl(config: AssemblyConfig) -> None:
    """Map RNA-seq reads to the genome using segemehl in split/spliced mode."""
    wd = config.work_dir
    log.info("Running segemehl read mapping...")

    index = wd / _INDEX
    if not index.exists():
        run_cmd(["segemehl.x", "-x", str(index), "-d", str(config.genome)], cwd=wd)

    # segemehl writes SAM to stdout; pipe straight into samtools for a sorted
    # BAM. `-S` enables split-read (spliced) mapping; `-` makes samtools read
    # the alignment stream from stdin.
    run_piped(
        [
            "segemehl.x",
            "-i", str(index),
            "-d", str(config.genome),
            *config.reads_args_segemehl,
            "-S",
            "-t", str(config.num_cpu),
        ],
        ["samtools", "sort", "-@", str(config.num_cpu), "-o", _BAM, "-"],
        cwd=wd,
    )
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

    # The genome index can be large; drop it once mapping is done.
    index.unlink(missing_ok=True)
