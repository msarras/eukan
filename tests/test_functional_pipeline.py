"""Tests for the functional annotation pipeline driver.

Focus: manifest step keys are scoped per proteome+mode so several
proteomes can be functionally annotated in one work_dir without the
second inheriting the first's "search complete" record (which used to
crash reading a cache file that was never written — see _step_scope).

The heavy homology search and FASTA annotation are monkeypatched: the
scoping/resume logic under test lives entirely in the driver, not in the
search itself, and the real steps need multi-GB databases.
"""

import json

import pytest

from eukan.functional import pipeline as fpipe
from eukan.settings import FunctionalConfig


def _fake_search(calls):
    """A _search_and_cache stand-in that records the proteome and writes caches."""
    def _search_and_cache(config, homology_json, hmmscan_json):
        calls.append(config.proteins.stem)
        homology_json.write_text(json.dumps({}))
        hmmscan_json.write_text(json.dumps({}))
        return homology_json
    return _search_and_cache


def _fake_annotate_fasta(proteins, homology_res, hmmscan_res, homology_db="uniprot"):
    out = proteins.parent / f"{proteins.stem}.mod{proteins.suffix}"
    out.write_text(">x\nMA\n")
    return out


@pytest.fixture
def calls(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(fpipe, "_search_and_cache", _fake_search(recorded))
    monkeypatch.setattr(fpipe, "annotate_fasta", _fake_annotate_fasta)
    return recorded


def _config(proteins_path, work_dir, homology_db="uniprot"):
    proteins_path.write_text(">x\nMA\n")
    return FunctionalConfig(
        proteins=proteins_path,
        work_dir=work_dir,
        manifest_dir=work_dir,
        homology_db=homology_db,
        num_cpu=1,
    )


class TestStepScope:
    def test_distinct_per_proteome_mode_and_dir(self, tmp_path):
        a = tmp_path / "a.faa"
        a.touch()
        b = tmp_path / "b.faa"
        b.touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        a_other_dir = sub / "a.faa"
        a_other_dir.touch()

        scope = fpipe._step_scope

        # Different proteomes → different scopes.
        assert scope(_config(a, tmp_path)) != scope(_config(b, tmp_path))
        # Same proteome, different homology DB → different scopes.
        assert scope(_config(a, tmp_path, "uniprot")) != scope(_config(a, tmp_path, "kofam"))
        # Same stem in a different directory → different scopes (distinct caches).
        assert scope(_config(a, tmp_path)) != scope(_config(a_other_dir, tmp_path))
        # Deterministic for the same inputs.
        assert scope(_config(a, tmp_path)) == scope(_config(a, tmp_path))


class TestMultiProteomeResume:
    def test_second_proteome_same_workdir_does_not_crash(self, tmp_path, calls):
        work_dir = tmp_path / "func-annot"

        prot_a = tmp_path / "speciesA.faa"
        fpipe.run_functional_annotation(_config(prot_a, work_dir))

        prot_b = tmp_path / "speciesB.faa"
        # Before the fix this raised FileNotFoundError: the search was
        # skipped (manifest said "complete") so speciesB's cache was never
        # written, yet the driver then tried to read it.
        fpipe.run_functional_annotation(_config(prot_b, work_dir))

        # Each proteome ran its own search; the second was NOT skipped.
        assert calls == ["speciesA", "speciesB"]
        assert (tmp_path / "speciesA.phmmer.json").exists()
        assert (tmp_path / "speciesB.phmmer.json").exists()

    def test_same_proteome_rerun_is_skipped(self, tmp_path, calls):
        work_dir = tmp_path / "func-annot"
        prot = tmp_path / "speciesA.faa"

        fpipe.run_functional_annotation(_config(prot, work_dir))
        fpipe.run_functional_annotation(_config(prot, work_dir))

        # Resume still works: search ran once, the re-run was skipped.
        assert calls == ["speciesA"]

    def test_mode_switch_reruns_search(self, tmp_path, calls):
        work_dir = tmp_path / "func-annot"
        prot = tmp_path / "speciesA.faa"

        fpipe.run_functional_annotation(_config(prot, work_dir, "uniprot"))
        fpipe.run_functional_annotation(_config(prot, work_dir, "kofam"))

        # Switching homology DB re-runs the search rather than reusing the
        # stale cache; each mode writes its own cache file.
        assert calls == ["speciesA", "speciesA"]
        assert (tmp_path / "speciesA.phmmer.json").exists()
        assert (tmp_path / "speciesA.kofam.json").exists()
