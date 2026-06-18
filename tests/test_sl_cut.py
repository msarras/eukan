"""Unit tests for eukan.assembly.sl_cut (genomic SL cut of transcript models)."""

from __future__ import annotations

import pysam
import pytest

from eukan.assembly.jaccard import _parse_transcript_models
from eukan.assembly.sl_acceptors import AcceptorSite
from eukan.assembly.sl_cut import (
    _project_genomic_to_spliced,
    bam_to_transcript_gff3,
    cut_models_at_sl,
)


def _make_bam(path, reads, ref="chr1", ref_len=100_000):
    """reads = [(name, flag, start, cigar, seq), ...]."""
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": ref, "LN": ref_len}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for name, flag, start, cigar, seq in reads:
            s = pysam.AlignedSegment(out.header)
            s.query_name = name
            s.flag = flag
            s.reference_id = 0
            s.reference_start = start
            s.mapping_quality = 60
            s.cigartuples = cigar
            s.query_sequence = seq
            out.write(s)


def _transcript_gff(tid, strand, exons):
    lo, hi = exons[0][0], exons[-1][1]
    rows = [
        "##gff-version 3",
        f"chr1\ttest\tgene\t{lo}\t{hi}\t.\t{strand}\t.\tID={tid}.g",
        f"chr1\ttest\tmRNA\t{lo}\t{hi}\t.\t{strand}\t.\tID={tid};Parent={tid}.g",
    ]
    for k, (s, e) in enumerate(exons, 1):
        rows.append(f"chr1\ttest\texon\t{s}\t{e}\t.\t{strand}\t.\tID={tid}.e{k};Parent={tid}")
    return "\n".join(rows) + "\n"


# --- _project_genomic_to_spliced -------------------------------------------


def test_project_plus_strand():
    exons = [(1, 40), (51, 90)]  # 5'->3' for '+'
    assert _project_genomic_to_spliced(exons, "+", 60) == 50
    assert _project_genomic_to_spliced(exons, "+", 1) == 1
    assert _project_genomic_to_spliced(exons, "+", 45) is None  # intronic (41-50)


def test_project_minus_strand():
    exons_5to3 = [(51, 90), (1, 40)]  # descending genomic for '-'
    assert _project_genomic_to_spliced(exons_5to3, "-", 90) == 1  # 5'-most base
    assert _project_genomic_to_spliced(exons_5to3, "-", 60) == 31


# --- bam_to_transcript_gff3 ------------------------------------------------


def test_bam_to_gff3_splits_at_intron(tmp_path):
    bam = tmp_path / "tx.bam"
    _make_bam(bam, [("q1", 0, 100, [(0, 10), (3, 50), (0, 10)], "A" * 20)])
    out = tmp_path / "tx.gff3"
    assert bam_to_transcript_gff3(bam, out, "rnaspades") == 1
    (m,) = _parse_transcript_models(out)
    assert m.exons == [(101, 110), (161, 170)]
    assert m.strand == "+"


def test_bam_to_gff3_minus_strand(tmp_path):
    bam = tmp_path / "tx.bam"
    _make_bam(bam, [("q1", 16, 100, [(0, 10), (3, 50), (0, 10)], "A" * 20)])
    out = tmp_path / "tx.gff3"
    bam_to_transcript_gff3(bam, out, "rnaspades")
    (m,) = _parse_transcript_models(out)
    assert m.strand == "-" and m.exons == [(101, 110), (161, 170)]


def test_bam_to_gff3_unique_ids_for_multimaps(tmp_path):
    bam = tmp_path / "tx.bam"
    _make_bam(bam, [
        ("q1", 0, 100, [(0, 20)], "A" * 20),
        ("q1", 256, 500, [(0, 20)], "A" * 20),  # secondary: a second locus
    ])
    out = tmp_path / "tx.gff3"
    assert bam_to_transcript_gff3(bam, out, "rnaspades") == 2
    ids = {m.tid for m in _parse_transcript_models(out)}
    assert ids == {"q1.m1", "q1.m2"}  # no Parent collision → loci stay distinct


# --- cut_models_at_sl ------------------------------------------------------


@pytest.mark.parametrize("strand,expected", [
    ("+", {"t1.j1": [(1, 40), (51, 59)], "t1.j2": [(60, 90)]}),
    ("-", {"t1.j1": [(61, 90)], "t1.j2": [(1, 40), (51, 60)]}),
])
def test_cut_both_strands(tmp_path, strand, expected):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", strand, [(1, 40), (51, 90)]))
    sites = [AcceptorSite("chr1", 60, strand, 5, ("reads",))]
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, sites, out, min_segment=25) == 1
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == expected


def test_cut_dot_strand_oriented_by_acceptor(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", ".", [(1, 40), (51, 90)]))
    sites = [AcceptorSite("chr1", 60, "+", 5, ("reads",))]
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, sites, out, min_segment=25) == 1
    models = {m.tid: (m.strand, m.exons) for m in _parse_transcript_models(out)}
    assert all(strand == "+" for strand, _ in models.values())  # SL imposed strand
    assert models["t1.j2"][1] == [(60, 90)]


def test_cut_ignores_opposite_strand_site(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (51, 90)]))
    sites = [AcceptorSite("chr1", 60, "-", 5, ("reads",))]  # wrong strand
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, sites, out, min_segment=25) == 0
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1": [(1, 40), (51, 90)]}


def test_cut_passthrough_without_sites(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (51, 90)]))
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25) == 0
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1": [(1, 40), (51, 90)]}
