"""Unit tests for pipeline state machine."""

from __future__ import annotations

from pathlib import Path

from pipeline.pipeline_state import STAGE_ORDER, PipelineStage, PipelineState


def test_stage_order_complete():
    assert PipelineStage.RAW_DATA_ACQUIRED in STAGE_ORDER
    assert PipelineStage.FRAMEWORK_VALIDATED in STAGE_ORDER
    assert STAGE_ORDER[0] == PipelineStage.RAW_DATA_ACQUIRED
    assert STAGE_ORDER[-1] == PipelineStage.FRAMEWORK_VALIDATED


def test_state_mark_and_persist(tmp_path: Path):
    state = PipelineState(project_root=tmp_path)
    state.mark_completed(PipelineStage.RAW_DATA_ACQUIRED)
    assert state.is_completed(PipelineStage.RAW_DATA_ACQUIRED)
    path = tmp_path / "outputs" / "pipeline_state.json"
    assert path.exists()

    loaded = PipelineState.load(tmp_path)
    assert loaded.is_completed(PipelineStage.RAW_DATA_ACQUIRED)


def test_reset_from_clears_later_stages(tmp_path: Path):
    state = PipelineState(project_root=tmp_path)
    state.mark_completed(PipelineStage.RAW_DATA_ACQUIRED)
    state.mark_completed(PipelineStage.DATA_VALIDATED)
    state.mark_completed(PipelineStage.DATA_PROCESSED)
    state.reset_from(PipelineStage.DATA_VALIDATED)
    assert state.is_completed(PipelineStage.RAW_DATA_ACQUIRED)
    assert not state.is_completed(PipelineStage.DATA_VALIDATED)
    assert not state.is_completed(PipelineStage.DATA_PROCESSED)