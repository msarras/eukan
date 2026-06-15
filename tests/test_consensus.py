"""Tests for eukan.annotation.consensus — PASA output resolution + engine dispatch."""

from __future__ import annotations

import os
import time
from pathlib import Path

from eukan.annotation import consensus as cons
from eukan.annotation.consensus import _resolve_consensus_path, build_consensus_models
from eukan.settings import PipelineConfig

_GENE_GFF = (
    "##gff-version 3\n"
    "chr1\tx\tgene\t1\t300\t.\t+\t.\tID=g1\n"
    "chr1\tx\tmRNA\t1\t300\t.\t+\t.\tID=g1.t1;Parent=g1\n"
    "chr1\tx\tCDS\t1\t300\t.\t+\t0\tID=c1;Parent=g1.t1\n"
)


class TestResolveConsensusPath:
    """``_resolve_consensus_path`` must return the *latest* PASA iteration.

    Regression: previously sorted file paths ascending and returned
    ``[0]`` — that's the *oldest* iteration. PASA suffixes can be
    variable-width process IDs, so even a "right index" lexicographic
    pick is fragile; we now sort by mtime instead.
    """

    def test_no_pasa_outputs_returns_evm_consensus(self, tmp_path):
        result = _resolve_consensus_path(tmp_path)
        assert result == tmp_path / "consensus_models.gff3"

    def test_picks_most_recently_modified_pasa_file(self, tmp_path):
        # Create three PASA iterations with monotonically-increasing mtime.
        old = tmp_path / "db.gene_structures_post_PASA_updates.99999.gff3"
        mid = tmp_path / "db.gene_structures_post_PASA_updates.55555.gff3"
        new = tmp_path / "db.gene_structures_post_PASA_updates.11111.gff3"
        for path in (old, mid, new):
            path.write_text("# placeholder\n")

        # Force ordered mtimes so the test doesn't depend on filesystem
        # tie-break order: old < mid < new.
        now = time.time()
        os.utime(old, (now - 200, now - 200))
        os.utime(mid, (now - 100, now - 100))
        os.utime(new, (now,       now))

        assert _resolve_consensus_path(tmp_path) == new

    def test_lex_sort_would_pick_wrong_file_with_variable_width_pids(self, tmp_path):
        """A 4-digit PID sorts lexicographically before a 5-digit one,
        so name-sorted ``[0]`` (or even ``[-1]``) can pick the wrong
        file. Mtime-based selection ignores filename ordering.
        """
        # Create the 5-digit (older) file first, then the 4-digit (newer).
        old_5digit = tmp_path / "db.gene_structures_post_PASA_updates.55555.gff3"
        new_4digit = tmp_path / "db.gene_structures_post_PASA_updates.9999.gff3"
        old_5digit.write_text("# old\n")
        new_4digit.write_text("# new\n")

        now = time.time()
        os.utime(old_5digit, (now - 100, now - 100))
        os.utime(new_4digit, (now,       now))

        # Lex sort would put 55555 last (after 9999), but the newer file
        # is 9999. The mtime-based selection picks 9999.
        assert _resolve_consensus_path(tmp_path) == new_4digit


class TestBuildConsensusDispatch:
    """``build_consensus_models`` routes to the engine named by config.

    combinr path: run_combinr_consensus, never PASA.
    evm path: run_evm (+ PASA only when utrs_db is set).
    """

    def _setup(self, tmp_path, monkeypatch) -> dict[str, int]:
        calls = {"combinr": 0, "evm": 0, "pasa": 0}
        sdir = tmp_path / "evm_consensus_models"

        def fake_combinr(config, s, evidence, *, transcripts=None):
            calls["combinr"] += 1
            (config.work_dir / "evm_consensus_models" / "consensus_models.gff3").write_text(_GENE_GFF)
            return s / "consensus_models.gff3"

        def fake_evm(config, evidence, *, transcripts=None):
            calls["evm"] += 1
            (config.work_dir / "evm_consensus_models" / "consensus_models.gff3").write_text(_GENE_GFF)
            return sdir / "consensus_models.gff3"

        def fake_pasa(*_a, **_k):
            calls["pasa"] += 1

        def fake_pretty(_consdb, _shortname, out):
            Path(out).write_text(_GENE_GFF)

        monkeypatch.setattr(cons, "run_combinr_consensus", fake_combinr)
        monkeypatch.setattr(cons, "run_evm", fake_evm)
        monkeypatch.setattr(cons, "add_utrs_from_pasa", fake_pasa)
        monkeypatch.setattr(cons, "_write_prettified_gff3", fake_pretty)
        return calls

    def _config(self, tmp_path, **kw) -> PipelineConfig:
        genome = tmp_path / "genome.fa"
        genome.write_text(">chr1\nACGT\n")
        prot = tmp_path / "proteins.faa"
        prot.write_text(">p\nM\n")
        return PipelineConfig(genome=genome, proteins=[prot], work_dir=tmp_path, **kw)

    def test_combinr_engine_skips_pasa(self, tmp_path, monkeypatch):
        calls = self._setup(tmp_path, monkeypatch)
        evidence = tmp_path / "prot.gff3"
        evidence.touch()
        cfg = self._config(tmp_path, consensus_engine="combinr")

        build_consensus_models(cfg, evidence, transcripts=None)

        assert calls == {"combinr": 1, "evm": 0, "pasa": 0}

    def test_evm_engine_is_default(self, tmp_path, monkeypatch):
        calls = self._setup(tmp_path, monkeypatch)
        evidence = tmp_path / "prot.gff3"
        evidence.touch()
        cfg = self._config(tmp_path)  # consensus_engine defaults to "evm"

        build_consensus_models(cfg, evidence, transcripts=None)

        assert calls["evm"] == 1 and calls["combinr"] == 0
        assert calls["pasa"] == 0  # no utrs_db set
