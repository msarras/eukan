"""Unit tests for eukan.assembly.sl_acceptors (SL trans-splice acceptor detection)."""

from __future__ import annotations

import json

import pysam

from eukan.assembly.sl_acceptors import (
    AcceptorSite,
    _iter_sl_ops,
    build_joint_consensus,
    detect_sl_acceptors,
    load_sl_acceptors,
)
from eukan.infra.artifacts import Artifact
from eukan.settings import AssemblyConfig

SL = "GTACTTTATT"            # 10 bp, meets _MIN_MOTIF_LEN
SL16 = "GTACTTTATTCCGGAA"    # 16 bp, meets _CONSENSUS_LEN


def _header(ref="chr1", ref_len=100_000):
    return pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"}, "SQ": [{"SN": ref, "LN": ref_len}]})


def _seg(flag, start, cigar, seq, header=None):
    s = pysam.AlignedSegment(header or _header())
    s.query_name = "r"
    s.flag = flag
    s.reference_id = 0
    s.reference_start = start
    s.mapping_quality = 60
    s.cigartuples = cigar
    s.query_sequence = seq
    return s


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


def _config(tmp_path, **kw):
    return AssemblyConfig(
        genome=tmp_path / "g.fa", work_dir=tmp_path, aligner="segemehl", num_cpu=1, **kw
    )


# --- _iter_sl_ops geometry -------------------------------------------------


def test_iter_sl_ops_forward_5p_clip():
    seg = _seg(0, 100, [(4, 10), (0, 20)], SL + "A" * 20)
    ops = list(_iter_sl_ops(seg, min_clip_len=8, min_ins_len=10, scan_insertions=False))
    assert ops == [(101, "+", SL)]  # acceptor = reference_start + 1


def test_iter_sl_ops_reverse_5p_clip():
    seg = _seg(16, 100, [(0, 20), (4, 10)], "A" * 20 + SL)
    ops = list(_iter_sl_ops(seg, min_clip_len=8, min_ins_len=10, scan_insertions=False))
    assert ops == [(120, "-", SL)]  # acceptor = reference_end (1-based last aligned base)


def test_iter_sl_ops_internal_insertion():
    seg = _seg(0, 100, [(0, 10), (1, 10), (0, 10)], "A" * 10 + SL + "A" * 10)
    ops = list(_iter_sl_ops(seg, min_clip_len=8, min_ins_len=10, scan_insertions=True))
    assert ops == [(111, "+", SL)]  # acceptor = first aligned base after the insertion


def test_iter_sl_ops_terminal_insertion_ignored():
    # A leading insertion is not internal → never an acceptor.
    seg = _seg(0, 100, [(1, 10), (0, 20)], SL + "A" * 20)
    ops = list(_iter_sl_ops(seg, min_clip_len=8, min_ins_len=10, scan_insertions=True))
    assert ops == []


# --- build_joint_consensus -------------------------------------------------


def _write_verdict(tmp_path, call, consensus=SL):
    summary = {"verdict": {"trans_splicing": {
        "call": call, "top_non_trivial_cluster_consensus": consensus,
        "top_non_trivial_cluster_key": consensus,
    }}}
    (tmp_path / Artifact.SOFTCLIP_DIAGNOSTIC.value).write_text(json.dumps(summary))


def test_consensus_override_wins(tmp_path):
    _write_verdict(tmp_path, "ABSENT")
    cfg = _config(tmp_path, sl_sequence="acgtacgtac")
    assert build_joint_consensus(cfg, []) == "ACGTACGTAC"


def test_consensus_from_strong_verdict(tmp_path):
    _write_verdict(tmp_path, "STRONG")
    assert build_joint_consensus(_config(tmp_path), []) == SL


def test_consensus_from_de_novo_fallback(tmp_path):
    # No override, no verdict file → fall back to the dominant de novo insertion.
    bam = tmp_path / "trinity-denovo.genome.bam"
    _make_bam(bam, [
        ("t1", 0, 100, [(0, 10), (1, 16), (0, 10)], "A" * 10 + SL16 + "A" * 10),
        ("t2", 0, 200, [(0, 10), (1, 16), (0, 10)], "A" * 10 + SL16 + "A" * 10),
    ])
    assert build_joint_consensus(_config(tmp_path), [bam]) == SL16


def test_consensus_none_without_signal(tmp_path):
    assert build_joint_consensus(_config(tmp_path), []) is None


# --- detect_sl_acceptors end-to-end ----------------------------------------


def test_detect_pools_reads_and_de_novo(tmp_path):
    cfg = _config(tmp_path, sl_sequence=SL)
    _make_bam(tmp_path / "segemehl_Aligned.sortedByCoord.out.bam",
              [("r1", 0, 100, [(4, 10), (0, 30)], SL + "A" * 30)])           # acceptor 101 (+)
    _make_bam(tmp_path / "trinity-denovo.genome.bam",
              [("t1", 0, 100, [(0, 10), (1, 10), (0, 10)], "A" * 10 + SL + "A" * 10)])  # 111 (+)

    detect_sl_acceptors(cfg)

    sites = {(s.pos, s.strand): s for s in load_sl_acceptors(tmp_path / "sl_acceptors.gff3")}
    assert (101, "+") in sites and sites[(101, "+")].sources == ("reads",)
    assert (111, "+") in sites and sites[(111, "+")].sources == ("trinity-denovo",)


def test_detect_clusters_and_unions_sources(tmp_path):
    cfg = _config(tmp_path, sl_sequence=SL, sl_cluster_window=5)
    # Two acceptors 2 bp apart (within the window) from different sources → one site.
    _make_bam(tmp_path / "segemehl_Aligned.sortedByCoord.out.bam",
              [("r1", 0, 100, [(4, 10), (0, 30)], SL + "A" * 30)])                       # 101
    _make_bam(tmp_path / "trinity-denovo.genome.bam",
              [("t1", 0, 102, [(4, 10), (0, 30)], SL + "A" * 30)])                       # 103
    detect_sl_acceptors(cfg)

    sites = load_sl_acceptors(tmp_path / "sl_acceptors.gff3")
    plus = [s for s in sites if s.strand == "+"]
    assert len(plus) == 1
    assert plus[0].support == 2
    assert set(plus[0].sources) == {"reads", "trinity-denovo"}


def test_detect_noop_without_signal(tmp_path):
    cfg = _config(tmp_path)  # no override, no verdict, no de novo signal
    _make_bam(tmp_path / "segemehl_Aligned.sortedByCoord.out.bam",
              [("r1", 0, 100, [(0, 30)], "A" * 30)])  # clean, no clip
    detect_sl_acceptors(cfg)

    out = tmp_path / "sl_acceptors.gff3"
    assert out.exists()
    assert load_sl_acceptors(out) == []


def test_acceptor_site_roundtrip(tmp_path):
    out = tmp_path / "acc.gff3"
    from eukan.assembly.sl_acceptors import _write_acceptors
    sites = [
        AcceptorSite("chr1", 500, "+", 12, ("reads", "rnaspades")),
        AcceptorSite("chr2", 30, "-", 3, ("trinity-denovo",)),
    ]
    _write_acceptors(sites, out)
    assert load_sl_acceptors(out) == sites
