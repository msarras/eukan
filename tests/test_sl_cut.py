"""Unit tests for eukan.assembly.sl_cut (genomic SL cut of transcript models)."""

from __future__ import annotations

import pysam
import pytest

from eukan.assembly.jaccard import _parse_transcript_models, _Tx
from eukan.assembly.sl_acceptors import AcceptorSite
from eukan.assembly.sl_cut import (
    _count_long_introns,
    _long_intron_cut_offsets,
    _project_genomic_to_spliced,
    bam_to_transcript_gff3,
    cut_models_at_sl,
    run_sl_cut,
)
from eukan.settings import AssemblyConfig

# A limit large enough that the SL-only test cases never trip the max-intron cut.
_NO_MAX = 1_000_000


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
    assert cut_models_at_sl(gff, sites, out, min_segment=25, max_intron_len=_NO_MAX) == (1, 0)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == expected


def test_cut_dot_strand_oriented_by_acceptor(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", ".", [(1, 40), (51, 90)]))
    sites = [AcceptorSite("chr1", 60, "+", 5, ("reads",))]
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, sites, out, min_segment=25, max_intron_len=_NO_MAX) == (1, 0)
    models = {m.tid: (m.strand, m.exons) for m in _parse_transcript_models(out)}
    assert all(strand == "+" for strand, _ in models.values())  # SL imposed strand
    assert models["t1.j2"][1] == [(60, 90)]


def test_cut_ignores_opposite_strand_site(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (51, 90)]))
    sites = [AcceptorSite("chr1", 60, "-", 5, ("reads",))]  # wrong strand
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, sites, out, min_segment=25, max_intron_len=_NO_MAX) == (0, 0)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1": [(1, 40), (51, 90)]}


def test_cut_passthrough_without_sites(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (51, 90)]))
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=_NO_MAX) == (0, 0)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1": [(1, 40), (51, 90)]}


# --- max-intron split ------------------------------------------------------


def test_max_intron_splits_plus(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (5100, 5140)]))  # intron 5059 nt
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=5000) == (1, 1)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1.j1": [(1, 40)], "t1.j2": [(5100, 5140)]}


def test_max_intron_short_untouched(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (51, 90)]))  # intron 10 nt
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=5000) == (0, 0)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1": [(1, 40), (51, 90)]}


def test_combined_sl_and_long_intron(tmp_path):
    # 3 exons: a short intron with an SL acceptor in exon 2, then an over-long intron.
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (51, 90), (6000, 6040)]))
    sites = [AcceptorSite("chr1", 60, "+", 5, ("reads",))]
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, sites, out, min_segment=25, max_intron_len=5000) == (1, 1)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {
        "t1.j1": [(1, 40), (51, 59)],
        "t1.j2": [(60, 90)],
        "t1.j3": [(6000, 6040)],
    }


def test_min_fragment_drops_tiny_tail(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (6000, 6010)]))  # 11-nt 3' tail
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=5000) == (1, 1)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1.j1": [(1, 40)]}  # tiny tail past the long intron dropped


def test_max_intron_splits_minus(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "-", [(1, 40), (6000, 6040)]))
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=5000) == (1, 1)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    # 5'->3' for '-' starts at the high-coordinate exon
    assert models == {"t1.j1": [(6000, 6040)], "t1.j2": [(1, 40)]}


def test_dot_strand_long_intron_only(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", ".", [(1, 40), (6000, 6040)]))
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=5000) == (1, 1)
    models = {m.tid: (m.strand, m.exons) for m in _parse_transcript_models(out)}
    assert models == {"t1.j1": (".", [(1, 40)]), "t1.j2": (".", [(6000, 6040)])}


def test_mono_exon_untouched(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 200)]))
    out = tmp_path / "cut.gff3"
    assert cut_models_at_sl(gff, [], out, min_segment=25, max_intron_len=5000) == (0, 0)
    models = {m.tid: m.exons for m in _parse_transcript_models(out)}
    assert models == {"t1": [(1, 200)]}


def test_max_intron_idempotent(tmp_path):
    gff = tmp_path / "t.gff3"
    gff.write_text(_transcript_gff("t1", "+", [(1, 40), (5100, 5140)]))
    out1 = tmp_path / "cut1.gff3"
    cut_models_at_sl(gff, [], out1, min_segment=25, max_intron_len=5000)
    out2 = tmp_path / "cut2.gff3"
    assert cut_models_at_sl(out1, [], out2, min_segment=25, max_intron_len=5000) == (0, 0)
    a = {m.tid: m.exons for m in _parse_transcript_models(out1)}
    b = {m.tid: m.exons for m in _parse_transcript_models(out2)}
    assert a == b


def test_long_intron_cut_offsets_plus():
    exons = [(1, 40), (5100, 5140), (5150, 5190)]  # gap1 5059 > 5000; gap2 9 short
    assert _long_intron_cut_offsets(exons, "+", 5000) == {40}


def test_long_intron_cut_offsets_minus():
    exons_5to3 = [(5100, 5140), (1, 40)]  # descending genomic; intron 5059 nt
    assert _long_intron_cut_offsets(exons_5to3, "-", 5000) == {41}


def test_long_intron_cut_offsets_none():
    assert _long_intron_cut_offsets([(1, 40), (51, 90)], "+", 5000) == set()
    assert _long_intron_cut_offsets([(1, 40)], "+", 5000) == set()


def test_count_long_introns():
    tx = _Tx("t", "chr1", "+", "src", [(1, 40), (5100, 5140), (5150, 5190)])
    assert _count_long_introns(tx, 5000) == 1
    tx2 = _Tx("t", "chr1", "+", "src", [(1, 40), (51, 90)])
    assert _count_long_introns(tx2, 5000) == 0


def test_run_sl_cut_prefers_jaccard_stringtie_gff3(tmp_path):
    # When the jaccard step produced stringtie.jaccard.gff3 (the de-fused StringTie
    # models), run_sl_cut must read it instead of the raw stringtie.gtf.
    (tmp_path / "stringtie.gtf").write_text(
        'chr1\tStringTie\texon\t1\t100\t.\t+\t.\ttranscript_id "FUSED";\n'
    )
    (tmp_path / "stringtie.jaccard.gff3").write_text(
        "##gff-version 3\n"
        "chr1\tj\tgene\t1\t40\t.\t+\t.\tID=A.gene\n"
        "chr1\tj\tmRNA\t1\t40\t.\t+\t.\tID=A;Parent=A.gene\n"
        "chr1\tj\texon\t1\t40\t.\t+\t.\tID=A.e1;Parent=A\n"
        "chr1\tj\tgene\t60\t100\t.\t+\t.\tID=B.gene\n"
        "chr1\tj\tmRNA\t60\t100\t.\t+\t.\tID=B;Parent=B.gene\n"
        "chr1\tj\texon\t60\t100\t.\t+\t.\tID=B.e1;Parent=B\n"
    )
    config = AssemblyConfig(genome=tmp_path / "g.fa", work_dir=tmp_path, num_cpu=1)
    run_sl_cut(config)  # no SL acceptors, no long introns -> passthrough

    out = (tmp_path / "stringtie.sl_cut.gff3").read_text()
    assert out.count("\tmRNA\t") == 2          # the two de-fused models, not the 1 fused
    assert "ID=A;" in out and "ID=B;" in out
    assert "FUSED" not in out
