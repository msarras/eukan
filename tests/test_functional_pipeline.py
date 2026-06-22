"""Tests for eukan.functional.pipeline — input-identity resume behaviour.

`eukan func-annot` shares one ``eukan-run.json`` across runs from the same
cwd. A second run on a *different* input file (by name or content) must
re-run on the new input without ``-f``; an *identical* input (same filename
and content) that was already computed is a friendly no-op telling the user
to pass ``-f`` to recompute. These tests stub the heavy search/annotate
steps with counters so the re-run vs skip decision is observable.
"""

from __future__ import annotations

import logging

import eukan.functional.pipeline as fp
from eukan.functional.pipeline import (
    _func_step_fingerprints,
    run_functional_annotation,
)
from eukan.settings import FunctionalConfig


def _install_stubs(monkeypatch) -> dict[str, int]:
    """Replace the search/annotate steps with counting stubs.

    Returns a dict of per-step invocation counters. The stubs write the same
    output files the real steps would (JSON caches, ``.mod.faa``, ``.mod.gff3``)
    so the manifest records valid, existing, integrity-checkable outputs.
    """
    counts = {"search": 0, "annotate_fasta": 0, "annotate_gff3": 0}

    def fake_search(config, homology_json, hmmscan_json):
        counts["search"] += 1
        homology_json.write_text("{}")
        hmmscan_json.write_text("{}")
        return homology_json

    def fake_annotate_fasta(proteins, homology_res, hmmscan_res, homology_db="uniprot"):
        counts["annotate_fasta"] += 1
        out = proteins.parent / f"{proteins.stem}.mod{proteins.suffix}"
        out.write_text(">s1 hypothetical protein\nMACE\n")
        return out

    def fake_annotate_gff3(
        gff3_path, homology_res, hmmscan_res, output_dir=None, homology_db="uniprot"
    ):
        counts["annotate_gff3"] += 1
        out = (output_dir or gff3_path.parent) / f"{gff3_path.stem}.mod{gff3_path.suffix}"
        out.write_text("##gff-version 3\nchr1\teukan\tgene\t1\t100\t.\t+\t.\tID=g1\n")
        return out

    monkeypatch.setattr(fp, "_search_and_cache", fake_search)
    monkeypatch.setattr(fp, "annotate_fasta", fake_annotate_fasta)
    monkeypatch.setattr(fp, "annotate_gff3", fake_annotate_gff3)
    return counts


def _config(tmp_path, name="proteins.faa", content=">s1\nMACE\n", gff3_path=None):
    proteins = tmp_path / name
    proteins.write_text(content)
    return FunctionalConfig(
        proteins=proteins, work_dir=tmp_path, num_cpu=2, gff3_path=gff3_path,
    )


class TestFuncAnnotInputIdentity:
    """Different inputs re-run by default; identical inputs are a no-op."""

    def test_different_input_reruns_without_force(self, tmp_path, monkeypatch):
        counts = _install_stubs(monkeypatch)
        cfg_a = _config(tmp_path, "prot_a.faa", ">a\nMAAA\n")
        cfg_b = _config(tmp_path, "prot_b.faa", ">b\nMBBB\n")

        run_functional_annotation(cfg_a)
        assert (counts["search"], counts["annotate_fasta"]) == (1, 1)

        # Same cwd / shared manifest, different proteins -> re-run, no -f.
        run_functional_annotation(cfg_b)
        assert (counts["search"], counts["annotate_fasta"]) == (2, 2)

    def test_same_name_different_content_reruns(self, tmp_path, monkeypatch):
        counts = _install_stubs(monkeypatch)
        proteins = tmp_path / "proteins.faa"
        proteins.write_text(">a\nMAAA\n")
        run_functional_annotation(
            FunctionalConfig(proteins=proteins, work_dir=tmp_path, num_cpu=2)
        )
        proteins.write_text(">a\nMCCC\n")  # same path, edited in place
        run_functional_annotation(
            FunctionalConfig(proteins=proteins, work_dir=tmp_path, num_cpu=2)
        )
        assert counts["search"] == 2  # content md5 flipped -> re-ran

    def test_identical_input_is_noop(self, tmp_path, monkeypatch, caplog):
        counts = _install_stubs(monkeypatch)
        cfg = _config(tmp_path)

        run_functional_annotation(cfg)
        with caplog.at_level(logging.INFO, logger="eukan.functional.pipeline"):
            run_functional_annotation(cfg)  # identical input -> friendly no-op

        assert counts["search"] == 1
        assert counts["annotate_fasta"] == 1
        assert "Already annotated" in caplog.text
        assert "-f" in caplog.text

    def test_identical_input_with_force_reruns(self, tmp_path, monkeypatch):
        counts = _install_stubs(monkeypatch)
        cfg = _config(tmp_path)
        run_functional_annotation(cfg)
        run_functional_annotation(cfg, force=True)
        assert counts["search"] == 2
        assert counts["annotate_fasta"] == 2

    def test_added_gff3_runs_without_force(self, tmp_path, monkeypatch):
        """Re-running the same proteins but newly asking for GFF3 annotation
        runs only the new step, no -f needed."""
        counts = _install_stubs(monkeypatch)
        proteins = tmp_path / "proteins.faa"
        proteins.write_text(">s1\nMACE\n")
        run_functional_annotation(
            FunctionalConfig(proteins=proteins, work_dir=tmp_path, num_cpu=2)
        )
        assert counts["annotate_gff3"] == 0

        gff3 = tmp_path / "genes.gff3"
        gff3.write_text("##gff-version 3\nchr1\teukan\tgene\t1\t100\t.\t+\t.\tID=g1\n")
        run_functional_annotation(
            FunctionalConfig(
                proteins=proteins, work_dir=tmp_path, num_cpu=2, gff3_path=gff3,
            )
        )
        # search + fasta cached (identical proteins), gff3 newly run.
        assert counts["search"] == 1
        assert counts["annotate_fasta"] == 1
        assert counts["annotate_gff3"] == 1


class TestFuncStepFingerprints:
    """``_func_step_fingerprints`` declares the right inputs/scalars per step."""

    def test_uniprot_scalars_and_inputs(self, tmp_path):
        proteins = tmp_path / "p.faa"
        proteins.write_text(">s\nM\n")
        cfg = FunctionalConfig(proteins=proteins, work_dir=tmp_path, num_cpu=8)
        hj, kj = tmp_path / "p.phmmer.json", tmp_path / "p.hmmscan.json"

        fp_map = _func_step_fingerprints(cfg, hj, kj)

        search_files, search_scalars = fp_map["search"]
        assert search_files == [proteins]
        assert "homology_db=uniprot" in search_scalars
        assert any(s.startswith("evalue=") for s in search_scalars)
        assert any(s.startswith("uniprot_db=") for s in search_scalars)
        # thread count must never enter the fingerprint (would force needless re-search)
        assert not any("num_cpu" in s for s in search_scalars)

        fasta_files, _ = fp_map["annotate_fasta"]
        assert fasta_files == [proteins, hj, kj]  # depends on both JSON caches
        assert "annotate_gff3" not in fp_map  # no gff3_path

    def test_gff3_entry_present_when_set(self, tmp_path):
        proteins = tmp_path / "p.faa"
        proteins.write_text(">s\nM\n")
        gff3 = tmp_path / "genes.gff3"
        gff3.write_text("##gff-version 3\n")
        cfg = FunctionalConfig(
            proteins=proteins, work_dir=tmp_path, gff3_path=gff3, num_cpu=2,
        )
        hj, kj = tmp_path / "p.phmmer.json", tmp_path / "p.hmmscan.json"

        gff3_files, _ = _func_step_fingerprints(cfg, hj, kj)["annotate_gff3"]
        assert gff3_files == [gff3, hj, kj]

    def test_kofam_mode_scalars(self, tmp_path):
        proteins = tmp_path / "p.faa"
        proteins.write_text(">s\nM\n")
        cfg = FunctionalConfig(
            proteins=proteins, work_dir=tmp_path, homology_db="kofam", num_cpu=2,
        )
        hj, kj = tmp_path / "p.kofam.json", tmp_path / "p.hmmscan.json"

        _, scalars = _func_step_fingerprints(cfg, hj, kj)["search"]
        assert any(s.startswith("kofam_db=") for s in scalars)
        assert any(s.startswith("ko_list=") for s in scalars)
        assert not any(s.startswith("uniprot_db=") for s in scalars)
