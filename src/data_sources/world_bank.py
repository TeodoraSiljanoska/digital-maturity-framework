"""World Bank API v2 adapter with pagination and retries.

With ``use_cached_raw: true`` the adapter serves a previously retrieved WDI
vintage from disk instead of calling the API, so a re-run reproduces the same
official values even after the World Bank revises its series.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

from common.errors import DataSourceError
from data_sources.base import (
    DataSourceAdapter,
    ensure_standard_columns,
    normalize_iso3_list,
    utc_now_iso,
)


class WorldBankAdapter(DataSourceAdapter):
    name = "world_bank"

    def __init__(self, config: Optional[Dict[str, Any]] = None, project_root: Optional[Any] = None):
        super().__init__(config=config, project_root=project_root)
        self.base_url = str(self.config.get("base_url", "https://api.worldbank.org/v2")).rstrip("/")
        self.timeout = int(self.config.get("timeout_seconds", 60))
        self.retry = int(self.config.get("retry", 3))
        self.per_page = int(self.config.get("per_page", 20000))
        self.use_cached_raw = bool(self.config.get("use_cached_raw", False))
        self.cached_raw_path = str(
            self.config.get("cached_raw_path", "data/raw/world_bank/wdi_vintage.parquet")
        )

    def _cached_path(self) -> Path:
        path = Path(self.cached_raw_path)
        if not path.is_absolute():
            root = Path(self.project_root) if self.project_root is not None else Path.cwd()
            path = root / path
        return path

    def _from_cache(
        self,
        codes: Sequence[str],
        countries: Sequence[str],
        start_year: int,
        end_year: int,
        id_map: Dict[str, str],
    ) -> pd.DataFrame:
        """Serve the stored vintage; its retrieved_at stamps are kept as recorded."""
        path = self._cached_path()
        if not path.exists():
            raise DataSourceError(
                "World Bank cached vintage not found",
                details={"path": str(path)},
            )
        cached = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        wanted = set(normalize_iso3_list(countries))
        out = cached.loc[
            cached["indicator_code"].astype(str).isin([str(c) for c in codes])
            & cached["country_iso3"].astype(str).str.upper().isin(wanted)
            & pd.to_numeric(cached["year"], errors="coerce").between(int(start_year), int(end_year))
        ].copy()
        if out.empty:
            raise DataSourceError(
                "World Bank cached vintage holds none of the requested cells",
                details={"path": str(path), "codes": list(codes)},
            )
        out["indicator_id"] = out["indicator_code"].map(id_map).fillna(out.get("indicator_id"))
        out["source"] = self.name
        out = ensure_standard_columns(out)
        out["country_iso3"] = out["country_iso3"].astype(str).str.upper()
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
        return out

    def _request_json(self, url: str) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retry + 1):
            try:
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code >= 500:
                    raise DataSourceError(
                        f"World Bank server error {resp.status_code}",
                        details={"url": url, "attempt": attempt},
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retry:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        raise DataSourceError(
            f"World Bank request failed after {self.retry} retries",
            details={"url": url, "cause": repr(last_err)},
        )

    def _fetch_indicator(
        self,
        code: str,
        countries: Sequence[str],
        start_year: int,
        end_year: int,
    ) -> List[Dict[str, Any]]:
        iso_list = ";".join(normalize_iso3_list(countries))
        if not iso_list:
            raise DataSourceError("World Bank: empty country list")

        page = 1
        pages = 1
        rows: List[Dict[str, Any]] = []
        while page <= pages:
            url = (
                f"{self.base_url}/country/{iso_list}/indicator/{code}"
                f"?date={int(start_year)}:{int(end_year)}"
                f"&format=json&per_page={self.per_page}&page={page}"
            )
            payload = self._request_json(url)
            if not isinstance(payload, list) or len(payload) < 2:
                raise DataSourceError(
                    "World Bank: unexpected response shape",
                    details={"url": url, "payload_type": type(payload).__name__},
                )
            meta, data = payload[0], payload[1]
            if isinstance(meta, dict):
                pages = int(meta.get("pages") or 1)
            if not data:
                break
            for item in data:
                if not isinstance(item, dict):
                    continue
                iso3 = item.get("countryiso3code") or ""
                year_raw = item.get("date")
                value = item.get("value")
                if not iso3 or year_raw is None:
                    continue
                try:
                    year = int(year_raw)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "country_iso3": str(iso3).upper(),
                        "year": year,
                        "indicator_code": code,
                        "value": value,
                    }
                )
            page += 1
        return rows

    def get_data(
        self,
        indicator_codes: Sequence[str],
        countries: Sequence[str],
        start_year: int,
        end_year: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        codes = [str(c) for c in indicator_codes]
        id_map: Dict[str, str] = dict(kwargs.get("indicator_id_map") or {})
        if self.use_cached_raw:
            return self._from_cache(codes, countries, start_year, end_year, id_map)
        retrieved_at = utc_now_iso()
        frames: List[pd.DataFrame] = []
        errors: List[str] = []

        for code in codes:
            try:
                rows = self._fetch_indicator(code, countries, start_year, end_year)
                if not rows:
                    errors.append(f"{code}: no rows")
                    continue
                part = pd.DataFrame(rows)
                part["indicator_id"] = id_map.get(code, kwargs.get("indicator_id", code))
                part["source"] = self.name
                part["retrieved_at"] = retrieved_at
                frames.append(part)
            except DataSourceError as exc:
                errors.append(f"{code}: {exc.message}")

        if not frames:
            raise DataSourceError(
                "World Bank: total failure fetching indicators",
                details={"errors": errors, "codes": codes},
            )
        out = ensure_standard_columns(pd.concat(frames, ignore_index=True))
        out["country_iso3"] = out["country_iso3"].astype(str).str.upper()
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["year"]).copy()
        out["year"] = out["year"].astype(int)
        return out

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update(
            {
                "base_url": self.base_url,
                "timeout_seconds": self.timeout,
                "retry": self.retry,
                "api": "World Bank API v2",
                "mode": "cached_vintage" if self.use_cached_raw else "live_api",
            }
        )
        if self.use_cached_raw:
            meta["cached_raw_path"] = self.cached_raw_path
            path = self._cached_path()
            if path.exists():
                cached = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                meta["vintage_retrieved_at"] = sorted(
                    str(v) for v in cached["retrieved_at"].dropna().unique()
                )
        return meta
