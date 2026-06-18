"""Unit tests for eukan.assembly.jaccard (Trinity-style jaccard clipping)."""

from __future__ import annotations

import random

import pysam
import pytest

from eukan.assembly import jaccard as j
from eukan.assembly.jaccard import (
    Trough,
    _candidate_troughs,
    _chr_bin_nbits,
    _group_and_pick_best,
    _partition_exons,
    _require_hills,
    _sa_index_nbases,
    clip_gff3,
    coverage_array,
    find_clip_points,
    iter_fragment_spans,
    jaccard_array,
    run_jaccard,
    split_fasta_record,
)
from eukan.settings import AssemblyConfig

# A 120 bp non-palindromic genome (seeded) so reverse-complement is distinct —
# essential for the minus-strand split test to actually exercise the sign.
GENOME = "".join(random.Random(42).choice("ACGT") for _ in range(120))


def _read_fasta(path) -> dict[str, str]:
    recs, name, seq = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs[name] = "".join(seq)
            name, seq = line[1:].split()[0], []
        else:
            seq.append(line)
    if name is not None:
        recs[name] = "".join(seq)
    return recs


# --- jaccard math ----------------------------------------------------------


def test_jaccard_single_edge_fragments():
    # One fragment touches the left window edge, one the right, neither spans:
    # n_both=0, n_single=2 -> jaccard = 1/3.
    jac = jaccard_array([(1, 50), (110, 160)], 300)
    # window_lend = mid - 49; mid=60 -> window_lend=11, window_rend=110.
    assert jac[60] == round(1 / 3, 4)


def test_jaccard_no_fragments_yields_no_clip():
    # With no fragments the pseudocount makes empty regions read as high jaccard
    # (1.0), never a trough — so no false split on a contig with no read support.
    jac = jaccard_array([], 100)
    assert len(jac) == 101
    assert find_clip_points([], 100) == []


def test_jaccard_window_length_boundary():
    # A fragment exactly W long spans exactly one junction position fully.
    W = j._WINDOW
    jac = jaccard_array([(1, W)], 200)
    # at window_lend=1 (mid=1+(W-1)//2) this single frag spans both edges.
    mid = 1 + (W - 1) // 2
    assert jac[mid] == round((1 + 1) / (0 + 1 + 1), 4)  # n_both=1,n_single=0 -> 2/2...


def test_clean_break_produces_trough_and_one_clip():
    # Two well-covered clusters separated by a gap (> W): read pairs do not
    # bridge the gap, so jaccard troughs there and exactly one clip is called.
    left = [(i, i + 150) for i in range(1, 80)]       # covers ~1..230
    right = [(i, i + 150) for i in range(400, 480)]   # covers ~400..630
    frags = left + right
    L = 700
    jac = jaccard_array(frags, L)
    assert min(jac[231:400]) <= j._MAX_TROUGH_VAL  # a real trough in the gap
    clips = find_clip_points(frags, L)
    assert len(clips) == 1
    assert 230 < clips[0] < 400  # the clip lands inside the break


def test_well_supported_contig_has_no_clip():
    # A single dense cluster, every position bridged -> no trough, no clip.
    frags = [(i, i + 150) for i in range(1, 200)]
    assert find_clip_points(frags, 350) == []


# --- coverage --------------------------------------------------------------


def test_coverage_array_inclusive():
    cov = coverage_array([(1, 3), (2, 5)], 6)
    assert cov == [0, 1, 2, 2, 1, 1, 0]  # index 0 unused


# --- trough detection helpers ----------------------------------------------


def _flat_jac(length: int, dips: list[int]) -> list[float]:
    arr = [0.0] + [1.0] * length
    for d in dips:
        arr[d] = 0.0
    return arr


def test_require_hills_needs_both_sides():
    jac = _flat_jac(600, [300])
    kept = _require_hills([Trough(300, 0.0, 1.0)], jac, j._TROUGH_WIN, j._MIN_JACCARD_DELTA)
    assert len(kept) == 1

    # No hill on the left (everything below the trough+delta there) -> dropped.
    one_sided = [0.0] * 301 + [1.0] * 300  # index 1..300 = 0.0, 301..600 = 1.0
    dropped = _require_hills([Trough(300, 0.0, 1.0)], one_sided, j._TROUGH_WIN, j._MIN_JACCARD_DELTA)
    assert dropped == []


def test_group_picks_single_deepest_within_window():
    jac = _flat_jac(700, [300, 450])  # two dips 150 bp apart (<= trough_win)
    troughs = _candidate_troughs(jac, j._TROUGH_WIN, j._MAX_TROUGH_VAL)
    best = _group_and_pick_best(troughs, j._TROUGH_WIN)
    assert len(best) == 1  # merged into one group


# --- fasta splitting -------------------------------------------------------


def test_split_fasta_record():
    segs = split_fasta_record("N" * 400, [100, 250], 25)
    assert [len(s) for s in segs] == [100, 150, 150]


def test_split_fasta_record_drops_short_pieces():
    segs = split_fasta_record("N" * 400, [10, 250], 25)  # first piece 10 bp < 25
    assert [len(s) for s in segs] == [240, 150]  # [11..250]=240, [251..400]=150


# --- STAR index sizing -----------------------------------------------------


def test_sa_index_nbases_caps_and_scales():
    assert _sa_index_nbases(10**12) == "14"     # capped
    assert int(_sa_index_nbases(80)) < 8         # small for a tiny set
    assert _sa_index_nbases(0) == "2"            # degenerate guard


def test_chr_bin_nbits_scales_down_for_many_short_refs():
    # ~76k contigs averaging ~1.3 kb -> bins must drop well below the default 18.
    assert int(_chr_bin_nbits(97_000_000, 76_198)) <= 11
    assert _chr_bin_nbits(3_000_000_000, 25) == "18"  # few large refs -> capped
    assert _chr_bin_nbits(0, 0) == "18"               # degenerate guard


# --- GFF3 / GTF split path -------------------------------------------------


def test_partition_exons_plus_strand():
    # exons (1,20) and (31,50); spliced length 40; clip after spliced base 25.
    segs = _partition_exons([(1, 20), (31, 50)], [25], "+", 40)
    assert segs == [[(1, 20), (31, 35)], [(36, 50)]]


def test_partition_exons_minus_strand():
    # 5'->3' order is descending genomic for '-': exon (31,50) then (1,20).
    segs = _partition_exons([(31, 50), (1, 20)], [25], "-", 40)
    # segment1 = first 25 spliced bases from the high-genomic 5' end.
    assert segs == [[(31, 50), (16, 20)], [(1, 15)]]


# Two exons (1..40) + (51..90); spliced length 80 so a clip at 45 leaves both
# pieces above the 25 bp floor.
_GFF_TEMPLATE = (
    "##gff-version 3\n"
    "chr1\ttest\tgene\t1\t90\t.\t{s}\t.\tID=g1\n"
    "chr1\ttest\tmRNA\t1\t90\t.\t{s}\t.\tID=t1;Parent=g1\n"
    "chr1\ttest\texon\t1\t40\t.\t{s}\t.\tID=t1.e1;Parent=t1\n"
    "chr1\ttest\texon\t51\t90\t.\t{s}\t.\tID=t1.e2;Parent=t1\n"
)


def _orig_spliced(gff, genome):
    from eukan.gff import create_gff_db
    from eukan.gff.io import iter_assembled_sequences

    return dict(
        (m.id, s)
        for m, s in iter_assembled_sequences(
            create_gff_db(str(gff)), genome, child_featuretype="exon"
        )
    )["t1"]


@pytest.mark.parametrize("strand", ["+", "-"])
def test_clip_gff3_splits_and_reextracts(tmp_path, strand):
    (tmp_path / "genome.fa").write_text(f">chr1\n{GENOME}\n")
    gff = tmp_path / "tx.gff3"
    gff.write_text(_GFF_TEMPLATE.format(s=strand))

    orig = _orig_spliced(gff, tmp_path / "genome.fa")
    assert len(orig) == 80

    out_gff = tmp_path / "clipped.gff3"
    out_fa = tmp_path / "clipped.fasta"
    clip_gff3(gff, tmp_path / "genome.fa", {"t1": [45]}, out_gff, out_fa)

    recs = _read_fasta(out_fa)
    assert set(recs) == {"t1.j1", "t1.j2"}
    assert recs["t1.j1"] == orig[:45]      # guards the minus-strand sign
    assert recs["t1.j2"] == orig[45:]
    assert recs["t1.j1"] + recs["t1.j2"] == orig


def test_clip_gff3_passthrough_when_no_clip(tmp_path):
    (tmp_path / "genome.fa").write_text(f">chr1\n{GENOME}\n")
    gff = tmp_path / "tx.gff3"
    gff.write_text(_GFF_TEMPLATE.format(s="+"))
    out_gff = tmp_path / "out.gff3"
    out_fa = tmp_path / "out.fasta"
    clip_gff3(gff, tmp_path / "genome.fa", {}, out_gff, out_fa)  # no clips
    assert set(_read_fasta(out_fa)) == {"t1"}


# --- run_jaccard wiring ----------------------------------------------------


def test_run_jaccard_noop_on_single_end(tmp_path, monkeypatch, caplog):
    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("STAR must not run on single-end input")

    monkeypatch.setattr(j, "_clip_one_fasta", _boom)
    (tmp_path / "rnaspades.sl_depleted.fasta").write_text(">c1\nACGTACGT\n")
    config = AssemblyConfig(
        genome=tmp_path / "genome.fa", work_dir=tmp_path,
        single_reads=tmp_path / "reads.fq", num_cpu=1,
    )
    with caplog.at_level("WARNING"):
        run_jaccard(config)
    assert "paired reads" in caplog.text
    # No .jaccard.fasta written.
    assert not (tmp_path / "rnaspades.sl_depleted.jaccard.fasta").exists()


# --- fragment extraction from a BAM ----------------------------------------


def _write_bam(path, reads, ref="tx1", ref_len=800):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": ref, "LN": ref_len}]}
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for name, flag, start, length in reads:
            seg = pysam.AlignedSegment(out.header)
            seg.query_name = name
            seg.flag = flag
            seg.reference_id = 0
            seg.reference_start = start
            seg.mapping_quality = 60
            seg.cigartuples = [(0, length)]  # length M
            out.write(seg)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))


def test_iter_fragment_spans_filters_pairs(tmp_path):
    bam = tmp_path / "tx.bam"
    # flags: 99 = paired,proper,first,mate-reverse ; 147 = paired,proper,second,reverse
    _write_bam(bam, [
        ("good", 99, 100, 50), ("good", 147, 250, 50),     # insert 200 -> kept (101,300)
        ("short", 99, 100, 50), ("short", 147, 100, 50),   # insert 50  -> dropped
        ("long", 99, 100, 50), ("long", 147, 650, 50),     # insert 600 -> dropped
        ("sec", 99 | 0x100, 120, 50),                       # secondary -> skipped
    ])
    spans = dict(iter_fragment_spans(bam))
    assert spans["tx1"] == [(101, 300)]
