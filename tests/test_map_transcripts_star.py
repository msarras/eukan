"""Unit tests for STAR transcript->genome mapping (command construction)."""

from __future__ import annotations

from pathlib import Path

from eukan.assembly import star
from eukan.settings import AssemblyConfig


def _config(tmp_path, **kw):
    (tmp_path / "genome.fa").write_text(">chr1\n" + "ACGT" * 100 + "\n")
    return AssemblyConfig(genome=tmp_path / "genome.fa", work_dir=tmp_path, num_cpu=4, **kw)


def _mock(monkeypatch, tmp_path):
    """Record run_cmd calls; materialize STAR's sorted BAM so the rename succeeds."""
    cmds: list[list[str]] = []

    def fake(cmd, **kw):
        cmds.append(cmd)
        if cmd[0] in ("STAR", "STARlong") and "--outFileNamePrefix" in cmd:
            prefix = cmd[cmd.index("--outFileNamePrefix") + 1]
            (tmp_path / f"{prefix}Aligned.sortedByCoord.out.bam").write_text("bam")

    monkeypatch.setattr(star, "run_cmd", fake)
    return cmds


def _map_cmds(cmds):
    return [c for c in cmds if c and c[0] in ("STAR", "STARlong") and "--readFilesIn" in c]


def test_maps_de_novo_assembly(tmp_path, monkeypatch):
    (tmp_path / "rnaspades.fasta").write_text(">t\nACGTACGT\n")
    cmds = _mock(monkeypatch, tmp_path)

    star.map_transcripts_star(_config(tmp_path))

    assert sum(1 for c in cmds if "genomeGenerate" in c) == 1  # one genome index
    queries = {Path(c[c.index("--readFilesIn") + 1]).name for c in _map_cmds(cmds)}
    assert queries == {"rnaspades.fasta"}
    assert (tmp_path / "rnaspades.genome.bam").exists()


def test_star_flags_ungapped_local(tmp_path, monkeypatch):
    (tmp_path / "rnaspades.fasta").write_text(">t\nACGTACGT\n")
    cmds = _mock(monkeypatch, tmp_path)

    star.map_transcripts_star(_config(tmp_path))

    (cmd,) = _map_cmds(cmds)
    assert cmd[cmd.index("--alignEndsType") + 1] == "Local"   # soft-clip the SL
    assert cmd[cmd.index("--alignIntronMax") + 1] == "1"      # ungapped, no SL split
    assert cmd[cmd.index("--outReadsUnmapped") + 1] == "Fastx"
    assert ["samtools", "index", "rnaspades.genome.bam"] in cmds


def test_prefers_jaccard_clipped_query(tmp_path, monkeypatch):
    (tmp_path / "rnaspades.fasta").write_text(">t\nACGT\n")
    (tmp_path / "rnaspades.jaccard.fasta").write_text(">t\nACGTACGT\n")
    cmds = _mock(monkeypatch, tmp_path)

    star.map_transcripts_star(_config(tmp_path))

    (cmd,) = _map_cmds(cmds)
    assert Path(cmd[cmd.index("--readFilesIn") + 1]).name == "rnaspades.jaccard.fasta"
    assert (tmp_path / "rnaspades.genome.bam").exists()  # output keyed to the stem


def test_unmapped_captured(tmp_path, monkeypatch):
    (tmp_path / "rnaspades.fasta").write_text(">t\nACGT\n")

    def fake(cmd, **kw):
        if cmd[0] == "STAR" and "--outFileNamePrefix" in cmd:
            p = cmd[cmd.index("--outFileNamePrefix") + 1]
            (tmp_path / f"{p}Aligned.sortedByCoord.out.bam").write_text("bam")
            (tmp_path / f"{p}Unmapped.out.mate1").write_text(">u\nACGT\n")

    monkeypatch.setattr(star, "run_cmd", fake)
    star.map_transcripts_star(_config(tmp_path))

    assert (tmp_path / "rnaspades.unmapped_transcripts.fasta").exists()


def test_no_assemblies_is_noop(tmp_path, monkeypatch):
    cmds = _mock(monkeypatch, tmp_path)
    star.map_transcripts_star(_config(tmp_path))
    assert cmds == []
