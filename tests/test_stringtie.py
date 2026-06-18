"""Unit tests for eukan.assembly.stringtie (command construction)."""

from __future__ import annotations

from eukan.assembly import stringtie
from eukan.settings import AssemblyConfig


def _config(tmp_path, **kw):
    return AssemblyConfig(
        genome=tmp_path / "genome.fa", work_dir=tmp_path, num_cpu=4,
        aligner="segemehl", **kw,
    )


def test_run_stringtie_command(tmp_path, monkeypatch):
    cmds: list[list[str]] = []
    monkeypatch.setattr(stringtie, "run_cmd", lambda cmd, **kw: cmds.append(cmd))

    stringtie.run_stringtie(_config(tmp_path))

    (cmd,) = cmds
    assert cmd[0] == "stringtie"
    assert cmd[1] == str(tmp_path / "segemehl_Aligned.sortedByCoord.out.bam")
    assert cmd[cmd.index("-p") + 1] == "4"
    assert cmd[cmd.index("-o") + 1] == "stringtie.gtf"
    # unstranded: no strand flag
    assert "--rf" not in cmd and "--fr" not in cmd


def test_run_stringtie_skips_when_output_exists(tmp_path, monkeypatch):
    (tmp_path / "stringtie.gtf").write_text("# already done\n")
    cmds: list[list[str]] = []
    monkeypatch.setattr(stringtie, "run_cmd", lambda cmd, **kw: cmds.append(cmd))

    stringtie.run_stringtie(_config(tmp_path))

    assert cmds == []
