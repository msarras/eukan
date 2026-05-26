"""Tests for EVM input staging and GeneMark source normalization.

Regression coverage for two bugs that silenced EVM evidence in the
no-transcripts + non-fungus/protist branch:

1. ``partition_EVM_inputs.pl`` was called with ``--transcript_alignments
   nr_transcripts.gff3`` but the file never existed — GeneMark predictions
   were passed under their own basename, ``genemark.gff3``.

2. GeneMark stamps column 2 of its GFF3 as ``GeneMark.hmm3`` (version-
   dependent), while the EVM weights map used the token ``genemark`` —
   EVM matches weights by source token, so the predictions were silently
   weighted zero.
"""

from __future__ import annotations

from pathlib import Path

import gffutils

from eukan.annotation import evm as evm_mod
from eukan.annotation.evm import _first_source_token, _stage_evm_inputs, run_evm
from eukan.annotation.genemark import _genemark_homogenize_source
from eukan.gff.normalize import normalize_to_gff3
from eukan.settings import PipelineConfig


def _make_gff(path: Path, source: str) -> Path:
    path.write_text(
        "##gff-version 3\n"
        f"chr1\t{source}\tgene\t1\t300\t.\t+\t.\tID=g1\n"
        f"chr1\t{source}\tmRNA\t1\t300\t.\t+\t.\tID=g1.t1;Parent=g1\n"
        f"chr1\t{source}\tCDS\t1\t300\t.\t+\t0\tID=cds1;Parent=g1.t1\n"
    )
    return path


class TestFirstSourceToken:
    def test_returns_column_two(self, tmp_path):
        gff = _make_gff(tmp_path / "x.gff3", "PASA-assembly")
        assert _first_source_token(gff) == "PASA-assembly"

    def test_skips_comments_and_blank_lines(self, tmp_path):
        path = tmp_path / "x.gff3"
        path.write_text(
            "##gff-version 3\n"
            "# some comment\n"
            "\n"
            "chr1\tgenemark\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        )
        assert _first_source_token(path) == "genemark"

    def test_returns_none_for_header_only(self, tmp_path):
        path = tmp_path / "x.gff3"
        path.write_text("##gff-version 3\n# no data\n")
        assert _first_source_token(path) is None


class TestStageEvmInputs:
    """``_stage_evm_inputs`` is what wires basename → role → weight.

    It must:
      - Write a weights.txt line per ab-initio whose basename is recognised.
      - Concatenate every ab-initio (i.e. everything except prot.gff3 and
        the transcripts file) into gene_predictions.gff3.
      - Symlink the transcripts file as nr_transcripts.gff3 *only* when one
        is supplied, and the TRANSCRIPT weights entry must use the source
        token actually present in the file.
    """

    def _run(
        self, sdir: Path, evidence: list[Path],
        transcripts: Path | None = None,
        weights: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        weights = weights or ["2", "1", "3"]
        sdir.mkdir(parents=True, exist_ok=True)
        _stage_evm_inputs(sdir, evidence, transcripts, weights)
        weights_text = (sdir / "weights.txt").read_text()
        preds = (sdir / "gene_predictions.gff3").read_bytes().decode()
        return weights_text, [line for line in preds.splitlines() if line.strip()]

    def _src_dir(self, tmp_path: Path) -> Path:
        """Sibling dir for source GFF3s so symlinks aren't self-referential."""
        d = tmp_path / "src"
        d.mkdir(exist_ok=True)
        return d

    def test_protein_only_evidence(self, tmp_path):
        src = self._src_dir(tmp_path)
        prot = _make_gff(src / "prot.gff3", "prot_align")
        weights_text, pred_lines = self._run(tmp_path / "evm", [prot])
        assert "PROTEIN\tprot_align\t2" in weights_text
        # prot.gff3 must not be concatenated into gene_predictions.gff3
        assert pred_lines == []

    def test_ab_initios_get_concatenated(self, tmp_path):
        src = self._src_dir(tmp_path)
        prot = _make_gff(src / "prot.gff3", "prot_align")
        aug = _make_gff(src / "augustus.gff3", "augustus")
        snap = _make_gff(src / "snap.gff3", "snap")

        weights_text, pred_lines = self._run(tmp_path / "evm", [prot, aug, snap])

        for token, _type, weight in [
            ("prot_align", "PROTEIN", "2"),
            ("augustus", "ABINITIO_PREDICTION", "1"),
            ("snap", "ABINITIO_PREDICTION", "1"),
        ]:
            assert f"{_type}\t{token}\t{weight}" in weights_text

        # Every ab initio is in gene_predictions.gff3.
        joined = "\n".join(pred_lines)
        assert "augustus\tgene" in joined
        assert "snap\tgene" in joined
        assert "prot_align\tgene" not in joined

    def test_transcripts_pasa_source(self, tmp_path):
        """PASA-assembled transcripts: weights entry must read PASA-assembly."""
        src = self._src_dir(tmp_path)
        prot = _make_gff(src / "prot.gff3", "prot_align")
        aug = _make_gff(src / "augustus.gff3", "augustus")
        trans = _make_gff(src / "pasa_in.gff3", "PASA-assembly")

        sdir = tmp_path / "evm"
        weights_text, _ = self._run(sdir, [prot, aug], transcripts=trans)

        assert (sdir / "nr_transcripts.gff3").is_symlink()
        assert "TRANSCRIPT\tPASA-assembly\t3" in weights_text

    def test_transcripts_genemark_source(self, tmp_path):
        """GeneMark-as-transcripts: weights entry tracks the file's source.

        This is the regression case — without source-aware staging EVM
        sees TRANSCRIPT predictions whose source doesn't match any
        weights entry and silently ignores them.
        """
        src = self._src_dir(tmp_path)
        prot = _make_gff(src / "prot.gff3", "prot_align")
        aug = _make_gff(src / "augustus.gff3", "augustus")
        snap = _make_gff(src / "snap.gff3", "snap")
        gm = _make_gff(src / "genemark.gff3", "genemark")

        sdir = tmp_path / "evm"
        weights_text, pred_lines = self._run(
            sdir, [prot, aug, snap], transcripts=gm,
        )

        # GeneMark is staged as nr_transcripts.gff3, not concat'd into
        # gene_predictions.gff3 (it's TRANSCRIPT evidence here).
        assert (sdir / "nr_transcripts.gff3").is_symlink()
        assert "TRANSCRIPT\tgenemark\t3" in weights_text
        # Ab-initio entry is absent because genemark.gff3 wasn't in evidence.
        assert "ABINITIO_PREDICTION\tgenemark" not in weights_text
        # And the file isn't concatenated through the ab-initio path.
        joined = "\n".join(pred_lines)
        assert "genemark\tgene" not in joined

    def test_unknown_basename_skips_weights_but_still_concats(self, tmp_path):
        """Files with non-canonical basenames still feed gene_predictions.gff3."""
        src = self._src_dir(tmp_path)
        prot = _make_gff(src / "prot.gff3", "prot_align")
        unknown = _make_gff(src / "mystery.gff3", "mystery")
        weights_text, pred_lines = self._run(tmp_path / "evm", [prot, unknown])

        assert "mystery" not in weights_text
        assert any("mystery\tgene" in line for line in pred_lines)


class TestGenemarkHomogenizeSource:
    """The transform must rewrite source regardless of GeneMark's value."""

    def _feature(self, source: str) -> gffutils.Feature:
        return gffutils.Feature(
            seqid="chr1", source=source, featuretype="gene",
            start=1, end=300, strand="+", attributes={"ID": ["g1"]},
        )

    def test_overrides_genemark_hmm3(self):
        f = self._feature("GeneMark.hmm3")
        assert _genemark_homogenize_source(f).source == "genemark"

    def test_overrides_genemark_hmm(self):
        f = self._feature("GeneMark.hmm")
        assert _genemark_homogenize_source(f).source == "genemark"

    def test_normalize_pipeline_applies_transform(self, tmp_path):
        """Integration: normalize_to_gff3 must wire the post_transform in.

        Mirror how run_genemark calls normalize_to_gff3 — with both the
        source-homogenize post_transform and fix_contig_names — and
        verify column 2 in the output is ``genemark`` end-to-end.
        """
        src = tmp_path / "in.gff"
        src.write_text(
            "##gff-version 3\n"
            "chr1\tGeneMark.hmm3\tgene\t100\t200\t.\t+\t.\tID=g1\n"
            "chr1\tGeneMark.hmm3\tmRNA\t100\t200\t.\t+\t.\tID=g1.t1;Parent=g1\n"
            "chr1\tGeneMark.hmm3\tCDS\t100\t200\t.\t+\t0\tID=cds1;Parent=g1.t1\n"
        )
        out = tmp_path / "out.gff3"

        normalize_to_gff3(
            src, out,
            post_transform=_genemark_homogenize_source,
            fix_contig_names=True,
        )

        for line in out.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split("\t")
            assert cols[1] == "genemark", f"source not normalized: {line!r}"


class TestRunEvmArgv:
    """``run_evm`` must include ``--transcript_alignments`` in both perl
    invocations when ``transcripts`` is set and omit it entirely when not.

    The previously-shipping bug crashed at ``partition_EVM_inputs.pl`` —
    a future refactor that splits one perl call could regress the other
    independently, so we pin the flag's presence on **both** scripts.
    """

    def _stub_externals(self, monkeypatch) -> list[list[str]]:
        """Replace every external entry point used by run_evm.

        Returns the captured-argv list, mutated by the fake ``run_cmd``.
        """
        captured: list[list[str]] = []

        def fake_run_cmd(
            cmd: list[str], *, cwd: Path,
            out_file: str | None = None,
            err_file: str | None = None,
            **_: object,
        ) -> None:
            captured.append(list(cmd))
            # write_EVM_commands.pl uses out_file="commands.list" — the
            # downstream code does (sdir / "commands.list").read_text(),
            # so the file must exist or run_evm crashes before our asserts.
            if out_file is not None:
                (cwd / out_file).write_text("")

        def fake_run_shell(*_args, **_kwargs) -> None:
            pass

        def fake_parallel_map(*_args, **_kwargs) -> None:
            pass

        def fake_concat_files(_srcs, dest: Path) -> None:
            Path(dest).write_bytes(b"")

        monkeypatch.setattr(evm_mod, "run_cmd", fake_run_cmd)
        monkeypatch.setattr(evm_mod, "run_shell", fake_run_shell)
        monkeypatch.setattr(evm_mod, "parallel_map", fake_parallel_map)
        monkeypatch.setattr(evm_mod, "concat_files", fake_concat_files)
        return captured

    def _make_config(self, tmp_path: Path) -> PipelineConfig:
        genome = tmp_path / "genome.fa"
        genome.touch()
        proteins = tmp_path / "proteins.faa"
        proteins.touch()
        return PipelineConfig(
            genome=genome, proteins=[proteins], work_dir=tmp_path,
        )

    def _gff(self, path: Path, source: str) -> Path:
        path.write_text(
            "##gff-version 3\n"
            f"chr1\t{source}\tgene\t1\t300\t.\t+\t.\tID=g1\n"
        )
        return path

    def _argv_starting_with(
        self, captured: list[list[str]], leading: str,
    ) -> list[str]:
        for argv in captured:
            if argv and argv[0] == leading:
                return argv
        raise AssertionError(
            f"no captured argv starting with {leading!r}; "
            f"got {[a[0] for a in captured if a]!r}"
        )

    def test_transcript_args_omitted_when_none(self, tmp_path, monkeypatch):
        captured = self._stub_externals(monkeypatch)
        src = tmp_path / "src"
        src.mkdir()
        prot = self._gff(src / "prot.gff3", "prot_align")

        run_evm(self._make_config(tmp_path), [prot], transcripts=None)

        for leading in ("partition_EVM_inputs.pl", "write_EVM_commands.pl"):
            argv = self._argv_starting_with(captured, leading)
            assert "--transcript_alignments" not in argv, (
                f"{leading}: flag should be omitted when transcripts is None"
            )
            assert "nr_transcripts.gff3" not in argv, (
                f"{leading}: filename should be omitted when transcripts is None"
            )

    def test_transcript_args_present_when_supplied(self, tmp_path, monkeypatch):
        captured = self._stub_externals(monkeypatch)
        src = tmp_path / "src"
        src.mkdir()
        prot = self._gff(src / "prot.gff3", "prot_align")
        trans = self._gff(src / "trans.gff3", "PASA-assembly")

        run_evm(self._make_config(tmp_path), [prot], transcripts=trans)

        for leading in ("partition_EVM_inputs.pl", "write_EVM_commands.pl"):
            argv = self._argv_starting_with(captured, leading)
            idx = argv.index("--transcript_alignments")
            assert argv[idx + 1] == "nr_transcripts.gff3", (
                f"{leading}: flag value should be the staged basename"
            )

        # And the file is actually staged as nr_transcripts.gff3.
        sdir = tmp_path / "evm_consensus_models"
        assert (sdir / "nr_transcripts.gff3").is_symlink()

    def test_transcript_args_appear_in_both_perl_calls(
        self, tmp_path, monkeypatch,
    ):
        """Asymmetry guard: previously only partition was passed the flag.

        If a future refactor adds the flag to one script but not the other,
        EVM partitions and command generation will disagree on what
        transcripts are in play. This test enforces the pair.
        """
        captured = self._stub_externals(monkeypatch)
        src = tmp_path / "src"
        src.mkdir()
        prot = self._gff(src / "prot.gff3", "prot_align")
        trans = self._gff(src / "trans.gff3", "genemark")

        run_evm(self._make_config(tmp_path), [prot], transcripts=trans)

        partition = self._argv_starting_with(captured, "partition_EVM_inputs.pl")
        write = self._argv_starting_with(captured, "write_EVM_commands.pl")
        assert "--transcript_alignments" in partition
        assert "--transcript_alignments" in write
