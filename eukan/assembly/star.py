"""STAR read mapping and hint generation."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from eukan.exceptions import ExternalToolError
from eukan.infra.artifacts import Artifact
from eukan.infra.logging import get_logger
from eukan.infra.runner import run_cmd
from eukan.infra.utils import concat_files
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

# STAR resource/index tuning for typical small eukaryotic genomes.
_STAR_SA_INDEX_NBASES = "3"                 # suffix-array pre-index size (small genomes)
_STAR_GENOME_GENERATE_RAM = "40317074816"   # byte cap for --limitGenomeGenerateRAM (~37 GiB)
_STAR_BAM_SORT_RAM = "27643756136"          # byte cap for --limitBAMsortRAM (~26 GiB)
# --outSJfilterIntronMaxVsReadN: max intron length allowed for junctions
# supported by 1, 2, 3+ reads respectively.
_STAR_SJ_FILTER_INTRON_MAX_VS_READN = ("100", "300", "500")


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

    # Generate hints from splice junctions and analyze splice sites
    _generate_hints_from_star(
        wd, config.genome, diagnose=config.diagnose_softclips,
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


# STAR motif codes → canonical splice site dinucleotide pairs
_STAR_MOTIF_NAMES: dict[int, str] = {
    1: "GT-AG",
    2: "CT-AC",
    3: "GC-AG",
    4: "CT-GC",
    5: "AT-AC",
    6: "GT-AT",
}


def _analyze_splice_sites(sj_file: Path, genome: Path, wd: Path) -> None:
    """Extract splice site dinucleotides from STAR junctions and write a summary.

    For each junction in SJ.out.tab, extracts the donor and acceptor
    dinucleotides from the genome FASTA.  Writes ``splice_site_summary.json``
    with per-type counts and read support.

    STAR SJ.out.tab columns (from STAR source, OutSJ.cpp):
      col2 = first base of intron (1-based)
      col3 = last base of intron (1-based)
      col5 = motif (0=non-canonical, 1=GT/AG, 2=CT/AC, 3=GC/AG, ...)
      col7 = unique reads, col8 = multi-mapping reads
    """
    from eukan.infra.genome import ContigIndex

    # Tally: splice_type → {"count": int, "unique_reads": int}
    tallies: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "unique_reads": 0})

    with ContigIndex(genome) as contigs, open(sj_file) as fin:
        reader = csv.reader(fin, delimiter="\t")
        for row in reader:
            chrom = row[0]
            intron_start = int(row[1])  # 1-based, first base of intron
            intron_end = int(row[2])    # 1-based, last base of intron
            motif = int(row[4])
            unique = int(row[6])

            if motif != 0:
                # Use STAR's motif classification for canonical/semi-canonical
                splice_type = _STAR_MOTIF_NAMES[motif]
            else:
                # Extract actual dinucleotides from the genome
                seq = contigs.get(chrom)
                if seq is None or seq.seq is None or intron_end > len(seq):
                    splice_type = "unknown"
                else:
                    genome_seq = seq.seq
                    donor = str(genome_seq[intron_start - 1 : intron_start + 1]).upper()
                    acceptor = str(genome_seq[intron_end - 2 : intron_end]).upper()
                    splice_type = f"{donor}-{acceptor}"

            tallies[splice_type]["count"] += 1
            tallies[splice_type]["unique_reads"] += unique

    summary = dict(sorted(tallies.items(), key=lambda kv: -kv[1]["count"]))
    with open(wd / Artifact.SPLICE_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    # Log summary
    for stype, counts in summary.items():
        if stype in ("GT-AG", "CT-AC"):
            continue  # skip canonical in log — they dominate
        log.info(
            "Splice sites (%s): %d junctions, %d unique reads",
            stype, counts["count"], counts["unique_reads"],
        )


def _run_softclip_diagnostic(wd: Path, genome: Path) -> None:
    """Walk the STAR BAM for soft-clip + intron motifs and log a verdict.

    Idempotent: if the summary JSON already exists, this is a no-op. The
    verdict surfaces trans-splicing and non-canonical splice prevalence
    so the user knows whether downstream gene prediction will need
    special handling (read pre-processing for trans-splicing; the
    ``--splice-permissive`` flag for non-canonical splice landscapes).
    """
    from eukan.assembly.bam_diagnostic import (
        compute_verdict,
        diagnose_bam,
        to_summary_dict,
    )

    bam = wd / "STAR_Aligned.sortedByCoord.out.bam"
    summary_path = wd / Artifact.SOFTCLIP_DIAGNOSTIC.value
    if not bam.exists():
        return
    if summary_path.exists():
        log.info("Soft-clip diagnostic already produced %s, skipping", summary_path.name)
        return

    log.info("Running soft-clip / intron diagnostic over %s...", bam.name)
    report = diagnose_bam(bam, genome)
    verdict = compute_verdict(report)

    with open(summary_path, "w") as f:
        json.dump(to_summary_dict(report, verdict), f, indent=2)

    ts = verdict.trans_splicing
    if ts.call in ("STRONG", "MODERATE"):
        sl_label = ts.top_non_trivial_cluster_consensus or ts.top_non_trivial_cluster_key
        log.warning(
            "Trans-splicing signal %s: top motif %s spans %d loci (%d reads). "
            "Reads may need splice-leader trimming before annotation.",
            ts.call,
            sl_label,
            ts.top_non_trivial_cluster_n_loci,
            ts.top_non_trivial_cluster_n_reads,
        )
    else:
        log.info("Trans-splicing signal: ABSENT")

    nc = verdict.non_canonical_splice
    if nc.call in ("EXTENSIVE", "MODERATE"):
        log.warning(
            "Non-canonical splice signal %s: canonical fraction %.2f%% "
            "(top non-canonical %s). Consider --splice-permissive on the assemble step.",
            nc.call, nc.canonical_pct, nc.top_non_canonical_dinuc,
        )
    else:
        log.info("Canonical splice site usage typical: %.2f%%", nc.canonical_pct)


def _generate_hints_from_star(
    wd: Path, genome: Path, *, diagnose: bool = True,
) -> None:
    """Generate AUGUSTUS hints from STAR splice junction and coverage output."""
    # Parse splice junctions into intron hints
    sj_file = wd / "STAR_SJ.out.tab"
    if sj_file.exists():
        _analyze_splice_sites(sj_file, genome, wd)
    if diagnose:
        _run_softclip_diagnostic(wd, genome)
    if sj_file.exists():
        strand_map = {"0": ".", "1": "+", "2": "-"}
        with open(sj_file) as fin, open(wd / "hints_introns.gff", "w") as fout:
            reader = csv.reader(fin, delimiter="\t")
            for row in reader:
                chrom, start, end = row[0], row[1], row[2]
                strand = strand_map.get(row[3], ".")
                unique = int(row[6]) + int(row[7])
                fout.write(
                    f"{chrom}\tSTAR\tintron\t{start}\t{end}\t{unique}\t"
                    f"{strand}\t.\tmult={unique};pri=4;src=E\n"
                )

    # Generate coverage hints from BAM
    bam = wd / "STAR_Aligned.sortedByCoord.out.bam"
    if bam.exists():
        run_cmd(
            ["samtools", "view", "-b", "-f", "0x10", str(bam)],
            cwd=wd, out_file="STAR_reverse.bam", binary=True,
        )
        run_cmd(
            ["samtools", "view", "-b", "-F", "0x10", str(bam)],
            cwd=wd, out_file="STAR_forward.bam", binary=True,
        )

        for direction, _strand, wig in [
            ("STAR_reverse.bam", "-", "minus.wig"),
            ("STAR_forward.bam", "+", "plus.wig"),
        ]:
            run_cmd(["bam2wig", direction], cwd=wd, out_file=wig)

        # Generate exonpart hints from coverage
        # wig2hints.pl reads from stdin, so we pipe the wig file in
        for wig_file, strand, hints_file in [
            ("minus.wig", "-", "hints.ep.minus.gff"),
            ("plus.wig", "+", "hints.ep.plus.gff"),
        ]:
            _run_wig2hints(wd, wig_file, strand, hints_file)

        # Merge coverage hints
        concat_files(
            [wd / hf for hf in ["hints.ep.minus.gff", "hints.ep.plus.gff"] if (wd / hf).exists()],
            wd / "hints_coverage.gff",
        )

        # Cleanup intermediate files
        for f in ["STAR_reverse.bam", "STAR_forward.bam", "minus.wig", "plus.wig",
                   "hints.ep.minus.gff", "hints.ep.plus.gff"]:
            (wd / f).unlink(missing_ok=True)


def _run_wig2hints(wd: Path, wig_file: str, strand: str, out_file: str) -> None:
    """Run wig2hints.pl, reading the wig from stdin and writing GFF to stdout."""
    run_cmd(
        [
            "wig2hints.pl",
            "--width=10", "--margin=10", "--minthresh=2",
            "--minscore=4", "--prune=0.1", "--src=W",
            "--type=exonpart", "--radius=4.5", "--pri=4",
            f"--strand={strand}",
        ],
        cwd=wd,
        in_file=wig_file,
        out_file=out_file,
    )
