"""Abstract data-source adapter interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from common.errors import DataSourceError, ValidationError


STANDARD_COLUMNS: List[str] = [
    "country_iso3",
    "year",
    "indicator_id",
    "indicator_code",
    "value",
    "source",
    "retrieved_at",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_standard_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[STANDARD_COLUMNS]


class DataSourceAdapter(ABC):
    """Adapter that returns long-format indicator panels."""

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Any] = None):
        self.config = dict(config or {})
        self.project_root = project_root

    @abstractmethod
    def get_data(
        self,
        indicator_codes: Sequence[str],
        countries: Sequence[str],
        start_year: int,
        end_year: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return long-format panel with STANDARD_COLUMNS."""

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Soft validation of adapter output; raises on empty/hard schema issues."""
        if df is None:
            raise ValidationError(f"{self.name}: validate received None")
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"{self.name}: validate expected DataFrame")
        if df.empty:
            raise ValidationError(
                f"{self.name}: empty dataset",
                details={"source": self.name},
            )
        missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
        if missing:
            raise ValidationError(
                f"{self.name}: missing required columns",
                details={"missing": missing},
            )
        non_null_ratio = float(df["value"].notna().mean()) if len(df) else 0.0
        return {
            "source": self.name,
            "rows": int(len(df)),
            "countries": int(df["country_iso3"].nunique()),
            "years": sorted(int(y) for y in df["year"].dropna().unique().tolist()),
            "indicators": sorted(df["indicator_code"].dropna().astype(str).unique().tolist()),
            "non_null_ratio": non_null_ratio,
            "ok": True,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.__class__.__name__,
            "config_keys": sorted(self.config.keys()),
            "standard_columns": list(STANDARD_COLUMNS),
        }


class SnapshotCsvAdapter(DataSourceAdapter):
    """Load curated official snapshot CSVs into the standard long format."""

    name = "snapshot"
    value_columns: Dict[str, str] = {}
    default_snapshot_key = "snapshot_file"

    def snapshot_path(self) -> Any:
        from pathlib import Path

        rel = self.config.get(self.default_snapshot_key)
        if not rel:
            raise DataSourceError(
                f"{self.name}: snapshot_file not configured",
                details={"config": self.config},
            )
        path = Path(rel)
        if not path.is_absolute():
            if self.project_root is None:
                path = Path.cwd() / path
            else:
                path = Path(self.project_root) / path
        return path

    def load_snapshot(self) -> pd.DataFrame:
        path = self.snapshot_path()
        if not path.exists():
            raise DataSourceError(
                f"{self.name}: snapshot missing at {path}",
                details={"path": str(path)},
            )
        df = pd.read_csv(path)
        if df.empty:
            raise DataSourceError(
                f"{self.name}: snapshot is empty",
                details={"path": str(path)},
            )
        return df

    def write_snapshot(self, df: pd.DataFrame) -> Any:
        from pathlib import Path

        from common.io import ensure_dir, write_df

        path = Path(self.snapshot_path())
        ensure_dir(path.parent)
        return write_df(df, path)

    def get_data(
        self,
        indicator_codes: Sequence[str],
        countries: Sequence[str],
        start_year: int,
        end_year: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        wide = self.load_snapshot()
        if "country_iso3" not in wide.columns or "year" not in wide.columns:
            raise DataSourceError(
                f"{self.name}: snapshot must include country_iso3 and year",
                details={"columns": list(wide.columns)},
            )

        codes = [str(c) for c in indicator_codes]
        countries_u = {str(c).upper() for c in countries}
        id_map: Dict[str, str] = dict(kwargs.get("indicator_id_map") or {})

        frames: List[pd.DataFrame] = []
        retrieved_at = utc_now_iso()
        for code in codes:
            value_col = self.value_columns.get(code, code)
            if value_col not in wide.columns:
                continue
            part = wide.loc[
                wide["country_iso3"].astype(str).str.upper().isin(countries_u)
                & wide["year"].between(int(start_year), int(end_year)),
                ["country_iso3", "year", value_col],
            ].copy()
            part = part.rename(columns={value_col: "value"})
            part["indicator_code"] = code
            part["indicator_id"] = id_map.get(code, kwargs.get("indicator_id", code))
            part["source"] = self.name
            part["retrieved_at"] = retrieved_at
            frames.append(part)

        if not frames:
            raise DataSourceError(
                f"{self.name}: no requested indicators found in snapshot",
                details={
                    "requested": codes,
                    "available_value_columns": list(self.value_columns.values()),
                    "snapshot_columns": list(wide.columns),
                },
            )
        out = ensure_standard_columns(pd.concat(frames, ignore_index=True))
        out["country_iso3"] = out["country_iso3"].astype(str).str.upper()
        out["year"] = out["year"].astype(int)
        return out

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        try:
            path = self.snapshot_path()
            meta["snapshot_file"] = str(path)
            meta["snapshot_exists"] = bool(path.exists())
        except Exception as exc:  # noqa: BLE001
            meta["snapshot_error"] = repr(exc)
        meta["value_columns"] = dict(self.value_columns)
        return meta


def filter_years(
    df: pd.DataFrame, start_year: int, end_year: int, year_col: str = "year"
) -> pd.DataFrame:
    return df.loc[df[year_col].between(int(start_year), int(end_year))].copy()


def normalize_iso3_list(countries: Iterable[str]) -> List[str]:
    return [str(c).strip().upper() for c in countries if str(c).strip()]
