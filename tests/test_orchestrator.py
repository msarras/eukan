"""Tests for eukan.{annotation,assembly}.pipeline — CLI flag → step translation."""

from __future__ import annotations

from typing import ClassVar

from eukan.annotation.pipeline import force_steps_from_run_flags
from eukan.assembly.pipeline import (
    force_steps_from_run_flags as assembly_force_steps_from_run_flags,
)
from eukan.infra.manifest import RunManifest
from eukan.infra.pipeline import run_orchestrated_step


class TestForceStepsFromRunFlags:
    """``--run-*`` CLI flags translate to the right manifest keys.

    The CLI surface is the dict of booleans accepted as kwargs here;
    the pipeline surface is the list of ``annotation/<step>`` keys
    that get popped from the manifest before re-execution.
    """

    def test_no_flags_returns_empty(self):
        assert force_steps_from_run_flags() == []

    def test_run_genemark_groups_orf_finder(self):
        """--run-genemark forces both genemark and orf_finder (shared flag)."""
        result = force_steps_from_run_flags(run_genemark=True)
        assert set(result) == {"annotation/genemark", "annotation/orf_finder"}

    def test_run_snap_groups_codingquarry(self):
        """--run-snap forces both snap and codingquarry (shared flag)."""
        result = force_steps_from_run_flags(run_snap=True)
        assert set(result) == {"annotation/snap", "annotation/codingquarry"}

    def test_run_augustus_alone(self):
        assert force_steps_from_run_flags(run_augustus=True) == ["annotation/augustus"]

    def test_run_consensus_alone(self):
        assert force_steps_from_run_flags(run_consensus=True) == [
            "annotation/evm_consensus_models"
        ]

    def test_run_prot_align_default_picks_non_ssp(self):
        """Without spaln_ssp, --run-prot-align forces prot_align only."""
        result = force_steps_from_run_flags(run_prot_align=True)
        assert result == ["annotation/prot_align"]

    def test_run_prot_align_with_spsp_picks_ssp(self):
        """With spaln_ssp=True, --run-prot-align forces prot_align_ssp only."""
        result = force_steps_from_run_flags(run_prot_align=True, spaln_ssp=True)
        assert result == ["annotation/prot_align_ssp"]

    def test_spsp_alone_does_not_force(self):
        """spaln_ssp gates which prot-align step gets forced; alone it's a no-op."""
        assert force_steps_from_run_flags(spaln_ssp=True) == []

    def test_all_flags_together(self):
        """Every --run-* flag set forces every step exactly once."""
        result = force_steps_from_run_flags(
            spaln_ssp=False,
            run_genemark=True,
            run_prot_align=True,
            run_augustus=True,
            run_snap=True,
            run_consensus=True,
        )
        assert set(result) == {
            "annotation/genemark",
            "annotation/orf_finder",
            "annotation/prot_align",
            "annotation/augustus",
            "annotation/snap",
            "annotation/codingquarry",
            "annotation/evm_consensus_models",
        }
        assert "annotation/prot_align_ssp" not in result

    def test_all_flags_with_spsp(self):
        """Same as above but spaln_ssp swaps prot_align → prot_align_ssp."""
        result = force_steps_from_run_flags(
            spaln_ssp=True,
            run_genemark=True,
            run_prot_align=True,
            run_augustus=True,
            run_snap=True,
            run_consensus=True,
        )
        assert "annotation/prot_align_ssp" in result
        assert "annotation/prot_align" not in result

    def test_returned_keys_are_prefixed(self):
        """Every returned key carries the ``annotation/`` prefix."""
        result = force_steps_from_run_flags(
            run_genemark=True, run_augustus=True, run_snap=True,
        )
        assert all(k.startswith("annotation/") for k in result)


class TestAssemblyForceStepsFromRunFlags:
    """``--run-X`` on assemble narrows the step list AND forces re-run.

    Returns full ``assembly/<step>`` keys, harmonized with the annotation
    pipeline. Empty list = "run all pending, force nothing".
    """

    _ALL_KEYS: ClassVar[list[str]] = [
        "assembly/star", "assembly/stringtie",
        "assembly/rnaspades", "assembly/jaccard", "assembly/map_transcripts",
        "assembly/sl_detect", "assembly/sl_cut", "assembly/combinr",
    ]

    def test_no_flags_returns_empty(self):
        """No flags → empty list → run all pending, force nothing."""
        assert assembly_force_steps_from_run_flags() == []

    def test_force_alone_returns_all_keys(self):
        """--force alone → re-run every step from scratch."""
        assert assembly_force_steps_from_run_flags(force=True) == self._ALL_KEYS

    def test_run_star_alone_forces_star_only(self):
        assert assembly_force_steps_from_run_flags(run_star=True) == ["assembly/star"]

    def test_run_combinr_alone_forces_combinr_only(self):
        assert assembly_force_steps_from_run_flags(run_combinr=True) == ["assembly/combinr"]

    def test_run_rnaspades_alone_forces_rnaspades_only(self):
        assert assembly_force_steps_from_run_flags(run_rnaspades=True) == ["assembly/rnaspades"]

    def test_run_stringtie_alone_forces_stringtie_only(self):
        assert assembly_force_steps_from_run_flags(run_stringtie=True) == ["assembly/stringtie"]

    def test_run_sl_steps_alone(self):
        assert assembly_force_steps_from_run_flags(run_sl_detect=True) == ["assembly/sl_detect"]
        assert assembly_force_steps_from_run_flags(run_sl_cut=True) == ["assembly/sl_cut"]

    def test_run_star_with_force_takes_run_flag(self):
        """--run-star --force scopes to star; --run-X takes precedence over --force."""
        assert assembly_force_steps_from_run_flags(run_star=True, force=True) == [
            "assembly/star"
        ]

    def test_multiple_run_flags(self):
        result = assembly_force_steps_from_run_flags(run_star=True, run_combinr=True)
        assert result == ["assembly/star", "assembly/combinr"]

    def test_step_order_is_pipeline_order(self):
        """Returned keys follow pipeline order regardless of kwarg order."""
        result = assembly_force_steps_from_run_flags(
            run_combinr=True, run_sl_cut=True, run_sl_detect=True,
            run_map_transcripts=True, run_jaccard=True, run_rnaspades=True,
            run_stringtie=True, run_star=True,
        )
        assert result == self._ALL_KEYS

    def test_returned_keys_are_prefixed(self):
        result = assembly_force_steps_from_run_flags(run_star=True, run_combinr=True)
        assert all(k.startswith("assembly/") for k in result)


class TestRunOrchestratedStepOutput:
    """A declared output_file is recorded in the manifest even when missing,
    so validate_step_outputs can flag it on resume instead of it being lost."""

    def test_missing_declared_output_is_still_recorded(self, tmp_path):
        manifest = RunManifest()
        out = tmp_path / "step" / "expected.gff3"

        def writes_nothing():
            return None

        result = run_orchestrated_step(
            tmp_path, manifest, "annotation/thing",
            writes_nothing,
            step_dir=tmp_path / "step",
            output_file=out,
        )
        assert result == out
        record = manifest.steps["annotation/thing"]
        assert record.output_file == str(out)
        assert record.output_md5 is None  # nothing to checksum

    def test_existing_declared_output_is_recorded_and_checksummed(self, tmp_path):
        manifest = RunManifest()
        out = tmp_path / "step" / "expected.gff3"

        def writes_output():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("data")
            return None

        result = run_orchestrated_step(
            tmp_path, manifest, "annotation/thing2",
            writes_output,
            step_dir=tmp_path / "step",
            output_file=out,
        )
        assert result == out
        record = manifest.steps["annotation/thing2"]
        assert record.output_file == str(out)
        assert record.output_md5 is not None
