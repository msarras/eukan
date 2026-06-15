"""Unit tests for segemehl transcript→genome mapping (command construction)."""

from __future__ import annotations

from pathlib import Path

from eukan.assembly import segemehl
from eukan.settings import AssemblyConfig


def _config(tmp_path, **kw):
    return AssemblyConfig(genome=tmp_path / "genome.fa", work_dir=tmp_path, num_cpu=4, **kw)


def _mock(monkeypatch):
    """Capture run_cmd calls; stub run_piped (the coordinate sort)."""
    cmds: list[list[str]] = []
    monkeypatch.setattr(segemehl, "run_cmd", lambda cmd, **kw: cmds.append(cmd))
    monkeypatch.setattr(segemehl, "run_piped", lambda *a, **kw: None)
    return cmds


def _map_cmds(cmds):
    return [c for c in cmds if c and c[0] == "segemehl.x" and "-q" in c]


def test_maps_all_three_assemblies(tmp_path, monkeypatch):
    for name in ("trinity-gg.fasta", "trinity-denovo.sl_depleted.fasta",
                 "rnaspades.sl_depleted.fasta"):
        (tmp_path / name).write_text(">t\nACGTACGT\n")
    cmds = _mock(monkeypatch)

    segemehl.map_transcripts_segemehl(_config(tmp_path))

    # genome index built once
    assert sum(1 for c in cmds if c[:2] == ["segemehl.x", "-x"]) == 1
    maps = _map_cmds(cmds)
    assert len(maps) == 3
    outs = {c[c.index("-o") + 1] for c in maps}
    assert outs == {
        str(tmp_path / "trinity-gg.genome.unsorted.bam"),
        str(tmp_path / "trinity-denovo.genome.unsorted.bam"),
        str(tmp_path / "rnaspades.genome.unsorted.bam"),
    }


def test_map_command_uses_expected_flags(tmp_path, monkeypatch):
    (tmp_path / "trinity-gg.fasta").write_text(">t\nACGTACGT\n")
    cmds = _mock(monkeypatch)

    segemehl.map_transcripts_segemehl(_config(tmp_path))

    (cmd,) = _map_cmds(cmds)
    assert cmd[cmd.index("-H") + 1] == "0"
    assert "-e" in cmd  # brief CIGAR
    assert "-S" in cmd  # split/spliced
    assert cmd[cmd.index("-q") + 1] == str(tmp_path / "trinity-gg.fasta")
    assert cmd[cmd.index("-d") + 1] == str(tmp_path / "genome.fa")
    assert cmd[cmd.index("-t") + 1] == "4"
    # samtools index runs on the sorted genome BAM
    assert ["samtools", "index", "trinity-gg.genome.bam"] in cmds


def test_skips_missing_rnaspades(tmp_path, monkeypatch):
    (tmp_path / "trinity-gg.fasta").write_text(">t\nACGT\n")
    (tmp_path / "trinity-denovo.sl_depleted.fasta").write_text(">t\nACGT\n")
    cmds = _mock(monkeypatch)

    segemehl.map_transcripts_segemehl(_config(tmp_path))

    queries = {Path(c[c.index("-q") + 1]).name for c in _map_cmds(cmds)}
    assert queries == {"trinity-gg.fasta", "trinity-denovo.sl_depleted.fasta"}


def test_resumes_complete_bam(tmp_path, monkeypatch):
    (tmp_path / "trinity-gg.fasta").write_text(">t\nACGT\n")
    (tmp_path / "trinity-gg.genome.bam").write_text("bam")  # pretend mapped
    cmds = _mock(monkeypatch)  # mocked quickcheck → "complete"

    segemehl.map_transcripts_segemehl(_config(tmp_path))

    # no mapping command for the already-complete BAM
    assert _map_cmds(cmds) == []


def test_no_assemblies_is_noop(tmp_path, monkeypatch):
    cmds = _mock(monkeypatch)
    segemehl.map_transcripts_segemehl(_config(tmp_path))
    assert cmds == []
