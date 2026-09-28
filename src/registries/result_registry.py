"""Registry for analysis results under results/ and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from common.io import ensure_dir, read_json, write_df, write_json


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultRegistry:
    """
    Persist tabular/JSON research results.

    Default root: ``results/`` (with index at ``results/registry.json``).
    Also mirrors a CSV index under ``outputs/`` when useful.
    """

    def __init__(
        self,
        project_root: Path | str,
        *,
        results_root: Optional[Path | str] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.root = ensure_dir(
            Path(results_root)
            if results_root
            else self.project_root / "results"
        )
        self.index_path = self.root / "registry.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            data = read_json(self.index_path)
            self._index = dict(data.get("results") or {})

    def register(
        self,
        result_id: str,
        payload: Union[Dict[str, Any], List[Any], pd.DataFrame, None] = None,
        *,
        category: str = "general",
        meta: Optional[Dict[str, Any]] = None,
        fmt: str = "json",
    ) -> Dict[str, Any]:
        category_dir = ensure_dir(self.root / category)
        if isinstance(payload, pd.DataFrame):
            artifact = category_dir / f"{result_id}.csv"
            write_df(payload, artifact)
            fmt = "csv"
        elif fmt == "csv" and payload is not None:
            artifact = category_dir / f"{result_id}.csv"
            if isinstance(payload, list):
                write_df(pd.DataFrame(payload), artifact)
            else:
                write_df(pd.DataFrame([payload]), artifact)
        else:
            artifact = category_dir / f"{result_id}.json"
            write_json(
                artifact,
                {
                    "result_id": result_id,
                    "category": category,
                    "meta": meta or {},
                    "data": payload,
                    "created_at": _utcnow_iso(),
                },
            )
            fmt = "json"

        entry = {
            "result_id": result_id,
            "category": category,
            "format": fmt,
            "created_at": _utcnow_iso(),
            "artifact_path": str(artifact.relative_to(self.project_root)),
            "meta": meta or {},
        }
        write_json(category_dir / f"{result_id}.meta.json", entry)
        self._index[result_id] = entry
        self._save_index()
        return entry

    def get(self, result_id: str) -> Dict[str, Any]:
        if result_id not in self._index:
            raise KeyError(f"Unknown result_id: {result_id}")
        return dict(self._index[result_id])

    def list(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        items = [dict(v) for v in self._index.values()]
        if category is not None:
            items = [i for i in items if i.get("category") == category]
        return items

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.list())

    def save_csv(self, path: Optional[Path | str] = None) -> Path:
        path = Path(path) if path else self.root / "registry.csv"
        ensure_dir(path.parent)
        self.to_dataframe().to_csv(path, index=False)
        # Mirror under outputs for audit convenience
        mirror = self.project_root / "outputs" / "results_registry.csv"
        ensure_dir(mirror.parent)
        self.to_dataframe().to_csv(mirror, index=False)
        return path

    def _save_index(self) -> Path:
        payload = {
            "updated_at": _utcnow_iso(),
            "results": self._index,
        }
        write_json(self.index_path, payload)
        self.save_csv()
        return self.index_path
