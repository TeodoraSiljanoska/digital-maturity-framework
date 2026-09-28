"""Data catalog built from indicators.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from common.io import ensure_dir, write_json


class DataCatalog:
    """In-memory catalog of research indicators keyed by indicator_id."""

    def __init__(self, indicators: Optional[Iterable[Dict[str, Any]]] = None):
        self._records: Dict[str, Dict[str, Any]] = {}
        if indicators:
            for item in indicators:
                self._add(item)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DataCatalog":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        indicators = data.get("indicators", []) if isinstance(data, dict) else []
        return cls(indicators)

    @classmethod
    def from_project(cls, project_root: Path | str) -> "DataCatalog":
        project_root = Path(project_root)
        return cls.from_yaml(project_root / "config" / "indicators.yaml")

    def _add(self, item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        key = item.get("indicator_id") or item.get("research_variable")
        if not key:
            raise ValueError(f"Indicator missing indicator_id: {item}")
        self._records[str(key)] = dict(item)

    def get(self, indicator_id: str) -> Dict[str, Any]:
        if indicator_id not in self._records:
            raise KeyError(f"Unknown indicator_id: {indicator_id}")
        return dict(self._records[indicator_id])

    def list(self) -> List[Dict[str, Any]]:
        return [dict(v) for v in self._records.values()]

    def by_source(self, source: str) -> List[Dict[str, Any]]:
        return [
            dict(v)
            for v in self._records.values()
            if str(v.get("source", "")).lower() == source.lower()
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = self.list()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def save_json(self, path: Path | str) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        payload = {"indicators": self.list()}
        return write_json(path, payload)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, indicator_id: object) -> bool:
        return str(indicator_id) in self._records

    def ids(self) -> List[str]:
        return list(self._records.keys())

    def acquired_ids(self) -> List[str]:
        """External indicators only (exclude constructed DMI)."""
        return [
            iid
            for iid, rec in self._records.items()
            if str(rec.get("role")) != "dependent"
            and str(rec.get("retrieval_method") or "") != "constructed"
        ]

    def uri(self, indicator_id: str) -> str:
        rec = self.get(indicator_id)
        return str(rec.get("uri") or f"http://purl.org/dmi-framework#Ind_{indicator_id}")

    def by_collection_method(self, method: str) -> List[Dict[str, Any]]:
        key = method.lower()
        return [
            dict(v)
            for v in self._records.values()
            if str(v.get("collection_method") or v.get("retrieval_method") or "").lower()
            == key
        ]

    def by_phase(self, phase: str) -> List[Dict[str, Any]]:
        key = phase.lower()
        return [
            dict(v)
            for v in self._records.values()
            if key in [str(p).lower() for p in (v.get("process_phases") or [])]
        ]

    def by_pillar(self, pillar: str) -> List[Dict[str, Any]]:
        return [
            dict(v)
            for v in self._records.values()
            if str(v.get("pillar", "")).lower() == pillar.lower()
        ]

    def reusable_frame(self) -> pd.DataFrame:
        """One row per indicator with URI, phase, country-independent facets."""
        rows = []
        for rec in self.list():
            rows.append(
                {
                    "indicator_id": rec.get("indicator_id"),
                    "uri": rec.get("uri"),
                    "name": rec.get("name"),
                    "role": rec.get("role"),
                    "pillar": rec.get("pillar"),
                    "source": rec.get("source"),
                    "collection_method": rec.get("collection_method")
                    or rec.get("retrieval_method"),
                    "process_phases": ",".join(rec.get("process_phases") or []),
                    "unit": rec.get("unit"),
                    "polarity": rec.get("polarity"),
                    "same_as": rec.get("same_as") or rec.get("source_url"),
                    "indicator_code": rec.get("indicator_code"),
                }
            )
        return pd.DataFrame(rows)
