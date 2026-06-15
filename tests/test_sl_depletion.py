"""Unit tests for eukan.assembly.sl_depletion."""

from __future__ import annotations

import json

from Bio import SeqIO

from eukan.assembly.sl_depletion import (
    _cut,
    _deplete_fasta,
    _find_sites,
    _resolve_sl_motif,
    _revcomp,
    _variants,
    run_sl_depletion,
)
from eukan.infra.artifacts import Artifact
from eukan.settings import AssemblyConfig

SL = "GTACTTTATT"  # 10 bp, meets _MIN_MOTIF_LEN


def _config(tmp_path, **kw):
    return AssemblyConfig(genome=tmp_path / "g.fa", work_dir=tmp_path, **kw)


def _write_fasta(path, records):
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))


def _patterns(motif, k=0):
    return _variants(motif, k) | _variants(_revcomp(motif), k)


# --- low-level helpers -----------------------------------------------------


def test_revcomp():
    assert _revcomp("GTACTTTATT") == "AATAAAGTAC"


def test_variants_counts():
    assert _variants("ACGT", 0) == {"ACGT"}
    # 1 substitution: original + 3 alternatives per position
    assert len(_variants("ACGT", 1)) == 1 + 3 * 4


def test_find_sites_exact_and_merge():
    seq = "A" * 20 + SL + "C" * 20
    sites = _find_sites(seq, _patterns(SL, 0), len(SL))
    assert sites == [(20, 30)]


def test_find_sites_reverse_complement():
    rc = _revcomp(SL)
    seq = "G" * 15 + rc + "T" * 15
    sites = _find_sites(seq, _patterns(SL), len(SL))
    assert sites == [(15, 25)]


def test_find_sites_tolerates_one_mismatch():
    mutated = "G" + "A" + SL[2:]  # one substitution at position 1
    seq = "A" * 20 + mutated + "C" * 20
    assert _find_sites(seq, _patterns(SL, 1), len(SL)) == [(20, 30)]
    # exact-only search misses it
    assert _find_sites(seq, _patterns(SL, 0), len(SL)) == []


def test_cut_makes_n_plus_one_fragments():
    seq = "AAA" + "##" + "BBB" + "##" + "CCC"  # ## at 3-4 and 8-9
    sites = [(3, 5), (8, 10)]
    assert _cut(seq, sites) == ["AAA", "BBB", "CCC"]


# --- _deplete_fasta --------------------------------------------------------


def test_deplete_splits_fused_contig(tmp_path):
    src = tmp_path / "in.fasta"
    out = tmp_path / "out.fasta"
    _write_fasta(src, [("fused", "A" * 40 + SL + "C" * 40)])
    n_in, n_out, n_cut, n_sites = _deplete_fasta(src, out, _patterns(SL), len(SL), 25)
    assert (n_in, n_out, n_cut, n_sites) == (1, 2, 1, 1)
    recs = list(SeqIO.parse(str(out), "fasta"))
    assert [r.id for r in recs] == ["fused.sl1", "fused.sl2"]
    assert str(recs[0].seq) == "A" * 40 and str(recs[1].seq) == "C" * 40
    assert SL not in str(recs[0].seq) + str(recs[1].seq)


def test_deplete_filters_short_fragments(tmp_path):
    src = tmp_path / "in.fasta"
    out = tmp_path / "out.fasta"
    _write_fasta(src, [("t", "A" * 10 + SL + "C" * 40)])  # leading frag is 10 bp
    _deplete_fasta(src, out, _patterns(SL), len(SL), 25)
    recs = list(SeqIO.parse(str(out), "fasta"))
    assert [r.id for r in recs] == ["t.sl1"]  # short 10 bp fragment dropped
    assert str(recs[0].seq) == "C" * 40


def test_deplete_passes_through_clean_contig(tmp_path):
    src = tmp_path / "in.fasta"
    out = tmp_path / "out.fasta"
    _write_fasta(src, [("clean", "ACGT" * 30)])
    n_in, n_out, n_cut, n_sites = _deplete_fasta(src, out, _patterns(SL), len(SL), 25)
    assert (n_in, n_out, n_cut, n_sites) == (1, 1, 0, 0)
    recs = list(SeqIO.parse(str(out), "fasta"))
    assert recs[0].id == "clean" and str(recs[0].seq) == "ACGT" * 30


# --- _resolve_sl_motif -----------------------------------------------------


def _write_verdict(tmp_path, call, consensus=SL, key=SL):
    summary = {
        "verdict": {
            "trans_splicing": {
                "call": call,
                "top_non_trivial_cluster_consensus": consensus,
                "top_non_trivial_cluster_key": key,
            }
        }
    }
    (tmp_path / Artifact.SOFTCLIP_DIAGNOSTIC.value).write_text(json.dumps(summary))


def test_resolve_motif_override_wins(tmp_path):
    _write_verdict(tmp_path, "ABSENT")
    cfg = _config(tmp_path, sl_sequence="acgtacgtac")
    assert _resolve_sl_motif(cfg) == "ACGTACGTAC"


def test_resolve_motif_from_strong_verdict(tmp_path):
    _write_verdict(tmp_path, "STRONG")
    assert _resolve_sl_motif(_config(tmp_path)) == SL


def test_resolve_motif_absent_returns_none(tmp_path):
    _write_verdict(tmp_path, "ABSENT")
    assert _resolve_sl_motif(_config(tmp_path)) is None


def test_resolve_motif_too_short_returns_none(tmp_path):
    _write_verdict(tmp_path, "STRONG", consensus="GTAC", key="GTA")
    assert _resolve_sl_motif(_config(tmp_path)) is None


def test_resolve_motif_no_summary_returns_none(tmp_path):
    assert _resolve_sl_motif(_config(tmp_path)) is None


# --- run_sl_depletion ------------------------------------------------------


def test_run_depletes_de_novo_when_trans_spliced(tmp_path):
    _write_verdict(tmp_path, "STRONG")
    _write_fasta(
        tmp_path / "trinity-denovo.fasta",
        [("fused", "A" * 40 + SL + "C" * 40), ("clean", "ACGT" * 30)],
    )
    _write_fasta(tmp_path / "rnaspades.fasta", [("r", "G" * 40 + SL + "T" * 40)])

    run_sl_depletion(_config(tmp_path, min_sl_fragment=25))

    tri = list(SeqIO.parse(str(tmp_path / "trinity-denovo.sl_depleted.fasta"), "fasta"))
    assert [r.id for r in tri] == ["fused.sl1", "fused.sl2", "clean"]
    rna = list(SeqIO.parse(str(tmp_path / "rnaspades.sl_depleted.fasta"), "fasta"))
    assert [r.id for r in rna] == ["r.sl1", "r.sl2"]


def test_run_is_passthrough_when_absent(tmp_path):
    _write_verdict(tmp_path, "ABSENT")
    fused = "A" * 40 + SL + "C" * 40
    _write_fasta(tmp_path / "trinity-denovo.fasta", [("fused", fused)])

    run_sl_depletion(_config(tmp_path, min_sl_fragment=25))

    recs = list(SeqIO.parse(str(tmp_path / "trinity-denovo.sl_depleted.fasta"), "fasta"))
    assert [r.id for r in recs] == ["fused"]  # not cut
    assert str(recs[0].seq) == fused


def test_run_skips_missing_rnaspades(tmp_path):
    _write_verdict(tmp_path, "STRONG")
    _write_fasta(tmp_path / "trinity-denovo.fasta", [("c", "ACGT" * 30)])
    run_sl_depletion(_config(tmp_path))
    assert (tmp_path / "trinity-denovo.sl_depleted.fasta").exists()
    assert not (tmp_path / "rnaspades.sl_depleted.fasta").exists()
