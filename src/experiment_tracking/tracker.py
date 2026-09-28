"""Lightweight experiment tracker writing experiments/{id}/meta.json."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.io import ensure_dir, read_json, write_json


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentTracker:
    """Create and update experiment directories under ``experiments/``."""

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.root = ensure_dir(self.project_root / "experiments")

    def _experiment_dir(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def _meta_path(self, experiment_id: str) -> Path:
        return self._experiment_dir(experiment_id) / "meta.json"

    def create(
        self,
        name: Optional[str] = None,
        *,
        experiment_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        experiment_id = experiment_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ) + "_" + uuid.uuid4().hex[:8]
        exp_dir = ensure_dir(self._experiment_dir(experiment_id))
        record = {
            "experiment_id": experiment_id,
            "name": name or experiment_id,
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
            "status": "created",
            "params": params or {},
            "metrics": {},
            "artifacts": [],
            "tags": tags or [],
            "meta": meta or {},
            "path": str(exp_dir.relative_to(self.project_root)),
        }
        write_json(self._meta_path(experiment_id), record)
        return record

    def update(
        self,
        experiment_id: str,
        *,
        status: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        record = self.get(experiment_id)
        if status is not None:
            record["status"] = status
        if params:
            record.setdefault("params", {}).update(params)
        if metrics:
            record.setdefault("metrics", {}).update(metrics)
        if artifacts:
            existing = list(record.get("artifacts") or [])
            for art in artifacts:
                if art not in existing:
                    existing.append(art)
            record["artifacts"] = existing
        if meta:
            record.setdefault("meta", {}).update(meta)
        if tags:
            existing_tags = list(record.get("tags") or [])
            for tag in tags:
                if tag not in existing_tags:
                    existing_tags.append(tag)
            record["tags"] = existing_tags
        record["updated_at"] = _utcnow_iso()
        write_json(self._meta_path(experiment_id), record)
        return record

    def log_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return self.update(experiment_id, metrics=metrics)

    def add_artifact(
        self, experiment_id: str, artifact_path: str | Path
    ) -> Dict[str, Any]:
        return self.update(experiment_id, artifacts=[str(artifact_path)])

    def get(self, experiment_id: str) -> Dict[str, Any]:
        path = self._meta_path(experiment_id)
        if not path.exists():
            raise KeyError(f"Unknown experiment_id: {experiment_id}")
        return read_json(path)

    def list(self) -> List[Dict[str, Any]]:
        experiments: List[Dict[str, Any]] = []
        if not self.root.exists():
            return experiments
        for child in sorted(self.root.iterdir()):
            meta = child / "meta.json"
            if child.is_dir() and meta.exists():
                experiments.append(read_json(meta))
        return experiments
