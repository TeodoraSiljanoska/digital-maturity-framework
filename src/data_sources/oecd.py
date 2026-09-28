"""OECD adapter (snapshot optional; otherwise indicator unavailable)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from common.errors import DataSourceError, IndicatorUnavailableError
from data_sources.base import DataSourceAdapter, ensure_standard_columns, utc_now_iso


class OECDAdapter(DataSourceAdapter):
    name = "oecd"

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Any] = None):
        super().__init__(config=config, project_root=project_root)

    def get_data(
        self,
        indicator_codes: Sequence[str],
        countries: Sequence[str],
        start_year: int,
        end_year: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        path = self._resolve_snapshot_path()
        if path is not None and path.exists():
            wide = pd.read_csv(path)
            codes = [str(c) for c in indicator_codes]
            id_map: Dict[str, str] = dict(kwargs.get("indicator_id_map") or {})
            frames = []
            retrieved_at = utc_now_iso()
            for code in codes:
                if code not in wide.columns:
                    raise IndicatorUnavailableError(
                        f"OECD indicator not present in snapshot: {code}",
                        details={"indicator_code": code, "path": str(path)},
                    )
                part = wide.loc[
                    wide["country_iso3"].astype(str).str.upper().isin(
                        {c.upper() for c in countries}
                    )
                    & wide["year"].between(int(start_year), int(end_year)),
                    ["country_iso3", "year", code],
                ].copy()
                part = part.rename(columns={code: "value"})
                part["indicator_code"] = code
                part["indicator_id"] = id_map.get(code, kwargs.get("indicator_id", code))
                part["source"] = self.name
                part["retrieved_at"] = retrieved_at
                frames.append(part)
            if not frames:
                raise DataSourceError("OECD: snapshot produced no rows")
            return ensure_standard_columns(pd.concat(frames, ignore_index=True))

        raise IndicatorUnavailableError(
            "OECD adapter is not configured for live retrieval of the requested indicators",
            details={
                "indicator_codes": list(indicator_codes),
                "hint": "Provide snapshot_file in sources.yaml or curated OECD extract",
            },
        )

    def _resolve_snapshot_path(self) -> Optional[Path]:
        rel = self.config.get("snapshot_file")
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            root = Path(self.project_root) if self.project_root else Path.cwd()
            path = root / path
        return path
