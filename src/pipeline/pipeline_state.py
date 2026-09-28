"""Persistent pipeline stage tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.io import ensure_dir, read_json, write_json


class PipelineStage(str, Enum):
    """Ordered research pipeline stages."""

    RAW_DATA_ACQUIRED = "RAW_DATA_ACQUIRED"
    DATA_VALIDATED = "DATA_VALIDATED"
    DATA_PROCESSED = "DATA_PROCESSED"
    INDEX_CREATED = "INDEX_CREATED"
    STATISTICS_COMPLETED = "STATISTICS_COMPLETED"
    ECONOMETRICS_COMPLETED = "ECONOMETRICS_COMPLETED"
    ML_COMPLETED = "ML_COMPLETED"
    XAI_COMPLETED = "XAI_COMPLETED"
    CONVERGENCE_COMPLETED = "CONVERGENCE_COMPLETED"
    VISUALIZATION_COMPLETED = "VISUALIZATION_COMPLETED"
    REPORTING_COMPLETED = "REPORTING_COMPLETED"
    FRAMEWORK_VALIDATED = "FRAMEWORK_VALIDATED"


STAGE_ORDER: List[PipelineStage] = list(PipelineStage)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_index(stage: PipelineStage | str) -> int:
    stage = PipelineStage(stage)
    return STAGE_ORDER.index(stage)


@dataclass
class PipelineState:
    """Track completed/failed stages and persist to outputs/pipeline_state.json."""

    project_root: Path
    completed_stages: List[str] = field(default_factory=list)
    failed_stage: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamps: Dict[str, str] = field(default_factory=dict)
    updated_at: Optional[str] = None

    @property
    def state_path(self) -> Path:
        return Path(self.project_root) / "outputs" / "pipeline_state.json"

    def mark_completed(self, stage: PipelineStage | str) -> None:
        stage_name = PipelineStage(stage).value
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
        # Keep completion order aligned with STAGE_ORDER
        order = {s.value: i for i, s in enumerate(STAGE_ORDER)}
        self.completed_stages = sorted(
            set(self.completed_stages), key=lambda s: order.get(s, 999)
        )
        self.timestamps[stage_name] = _utcnow_iso()
        if self.failed_stage == stage_name:
            self.failed_stage = None
        self.updated_at = _utcnow_iso()
        self.save()

    def mark_failed(
        self,
        stage: PipelineStage | str,
        error: Exception | str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        stage_name = PipelineStage(stage).value
        self.failed_stage = stage_name
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "message": str(error),
            "timestamp": _utcnow_iso(),
        }
        if details:
            entry["details"] = details
        if hasattr(error, "to_dict"):
            try:
                entry["error"] = error.to_dict()  # type: ignore[union-attr]
            except Exception:
                pass
        self.errors.append(entry)
        self.timestamps[f"{stage_name}_failed"] = entry["timestamp"]
        self.updated_at = entry["timestamp"]
        self.save()

    def is_completed(self, stage: PipelineStage | str) -> bool:
        return PipelineStage(stage).value in self.completed_stages

    def reset_from(self, stage: PipelineStage | str) -> None:
        """Clear completion status from ``stage`` inclusive onward."""
        stage = PipelineStage(stage)
        cut = _stage_index(stage)
        keep = {
            s.value
            for i, s in enumerate(STAGE_ORDER)
            if i < cut and s.value in self.completed_stages
        }
        self.completed_stages = [
            s.value for s in STAGE_ORDER if s.value in keep
        ]
        # Drop timestamps for cleared stages
        cleared = {s.value for s in STAGE_ORDER[cut:]}
        self.timestamps = {
            k: v
            for k, v in self.timestamps.items()
            if k not in cleared and not any(k.startswith(f"{c}_") for c in cleared)
        }
        if self.failed_stage and _stage_index(self.failed_stage) >= cut:
            self.failed_stage = None
        self.updated_at = _utcnow_iso()
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_stages": list(self.completed_stages),
            "failed_stage": self.failed_stage,
            "errors": list(self.errors),
            "timestamps": dict(self.timestamps),
            "updated_at": self.updated_at,
        }

    def save(self) -> Path:
        ensure_dir(self.state_path.parent)
        return write_json(self.state_path, self.to_dict())

    @classmethod
    def load(cls, project_root: Path) -> "PipelineState":
        project_root = Path(project_root).resolve()
        path = project_root / "outputs" / "pipeline_state.json"
        state = cls(project_root=project_root)
        if not path.exists():
            return state
        data = read_json(path)
        state.completed_stages = list(data.get("completed_stages") or [])
        state.failed_stage = data.get("failed_stage")
        state.errors = list(data.get("errors") or [])
        state.timestamps = dict(data.get("timestamps") or {})
        state.updated_at = data.get("updated_at")
        return state
