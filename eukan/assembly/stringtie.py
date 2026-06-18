"""Genome-guided transcript assembly with StringTie.

StringTie assembles transcript models directly from the read→genome BAM
(``config.aligner_bam``), emitting a genomic GTF. It replaces Trinity's
genome-guided mode: the output is already in genome coordinates, so it needs no
transcript→genome mapping. The library is treated as unstranded (no
``--rf``/``--fr``); the SL-cut step (:mod:`eukan.assembly.sl_cut`) later imposes
strand at trans-splice acceptor sites for any ``.``-strand models.
"""

from __future__ import annotations

from eukan.infra.logging import get_logger
from eukan.infra.runner import run_cmd
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

_GTF = "stringtie.gtf"


def run_stringtie(config: AssemblyConfig) -> None:
    """Genome-guided assembly off the aligner BAM → ``stringtie.gtf``.

    Skips when ``stringtie.gtf`` already exists (mirrors the Trinity/rnaSPAdes
    idempotence guards). The coordinate-sorted aligner BAM is read directly; no
    BAM index is required by StringTie.
    """
    wd = config.work_dir
    out = wd / _GTF
    if out.exists():
        return
    log.info("Running StringTie genome-guided assembly...")
    run_cmd(
        [
            "stringtie",
            str(wd / config.aligner_bam),
            "-p", str(config.num_cpu),
            "-o", _GTF,
        ],
        cwd=wd,
    )
