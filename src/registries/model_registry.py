"""Registry for fitted models and metadata under outputs/models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from common.io import ensure_dir, read_json, write_json


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """Persist model artifacts and an index at outputs/models/registry.json."""

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.root = ensure_dir(self.project_root / "outputs" / "models")
        self.index_path = self.root / "registry.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            data = read_json(self.index_path)
            self._index = dict(data.get("models") or {})

    def register(
        self,
        model_id: str,
        model: Any = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        artifact_name: str = "model.joblib",
    ) -> Dict[str, Any]:
        model_dir = ensure_dir(self.root / model_id)
        artifact_path = model_dir / artifact_name
        if model is not None:
            joblib.dump(model, artifact_path)

        entry = {
            "model_id": model_id,
            "created_at": _utcnow_iso(),
            "artifact_path": str(artifact_path.relative_to(self.project_root))
            if artifact_path.exists()
            else None,
            "meta": meta or {},
            "metrics": metrics or {},
            "tags": tags or [],
        }
        write_json(model_dir / "meta.json", entry)
        self._index[model_id] = entry
        self._save_index()
        return entry

    def get(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._index:
            raise KeyError(f"Unknown model_id: {model_id}")
        return dict(self._index[model_id])

    def load_model(self, model_id: str) -> Any:
        entry = self.get(model_id)
        rel = entry.get("artifact_path")
        if not rel:
            raise FileNotFoundError(f"No artifact registered for {model_id}")
        path = self.project_root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        return joblib.load(path)

    def list(self) -> List[Dict[str, Any]]:
        return [dict(v) for v in self._index.values()]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for entry in self._index.values():
            flat = {
                "model_id": entry.get("model_id"),
                "created_at": entry.get("created_at"),
                "artifact_path": entry.get("artifact_path"),
                "tags": ",".join(entry.get("tags") or []),
            }
            metrics = entry.get("metrics") or {}
            for k, v in metrics.items():
                flat[f"metric_{k}"] = v
            rows.append(flat)
        return pd.DataFrame(rows)

    def save_csv(self, path: Optional[Path | str] = None) -> Path:
        path = Path(path) if path else self.root / "registry.csv"
        df = self.to_dataframe()
        ensure_dir(path.parent)
        df.to_csv(path, index=False)
        return path

    def _save_index(self) -> Path:
        payload = {
            "updated_at": _utcnow_iso(),
            "models": self._index,
        }
        write_json(self.index_path, payload)
        self.save_csv()
        return self.index_path
