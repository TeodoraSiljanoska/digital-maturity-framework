"""Context indicators Z1–Z6 from the World Bank WDI API.

Each value keeps the request URL, the retrieval time, the SHA-256 of the raw
JSON response and the WDI ``lastupdated`` date of the series. Cells are
labelled ``official``, ``carried_forward`` (same country's nearest publication
at most ``carry_max_years`` away, as for X1–X10) or ``missing``; no context
cell is ever reconstructed by a model.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import yaml

from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger

logger = get_logger("dmf.knowledge.context")

COUNTRIES = ["MKD", "SRB", "SVN", "FRA", "GBR", "DEU", "USA", "CAN", "AUS", "BRA", "ARG", "IND"]


def load_config(root: Path) -> Dict[str, Any]:
    with (root / "config" / "context_indicators.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get(url: str, timeout: int, retry: int) -> Tuple[bytes, Any]:
    last = None
    for attempt in range(1, retry + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content, resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"WDI request failed: {url}: {last!r}")


def fetch_vintage(root: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Download every Z series once and store the vintage with per-value provenance."""
    v = cfg["vintage"]
    base = str(v["base_url"]).rstrip("/")
    y0, y1 = cfg["years"]
    rows: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    for zid, spec in cfg["indicators"].items():
        code = spec["code"]
        url = (
            f"{base}/country/{';'.join(COUNTRIES)}/indicator/{code}"
            f"?date={y0}:{y1}&format=json&per_page=1000"
        )
        raw, payload = _get(url, int(v["timeout_seconds"]), int(v["retry"]))
        retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sha = hashlib.sha256(raw).hexdigest()
        info, data = payload[0], payload[1] or []
        meta_url = f"{base}/indicator/{code}?format=json"
        _, mpayload = _get(meta_url, int(v["timeout_seconds"]), int(v["retry"]))
        m = (mpayload[1] or [{}])[0]
        meta.append(
            {
                "indicator_id": zid,
                "indicator_code": code,
                "wdi_name": m.get("name"),
                "wdi_source": (m.get("source") or {}).get("value"),
                "source_organization": m.get("sourceOrganization"),
                "lastupdated": info.get("lastupdated"),
                "request_url": url,
                "retrieved_at": retrieved,
                "response_sha256": sha,
                "n_records": len(data),
            }
        )
        for item in data:
            rows.append(
                {
                    "indicator_id": zid,
                    "indicator_code": code,
                    "country_iso3": str(item.get("countryiso3code") or "").upper(),
                    "year": int(item["date"]),
                    "value": item.get("value"),
                    "wdi_lastupdated": info.get("lastupdated"),
                    "source_url": url,
                    "retrieved_at": retrieved,
                    "response_sha256": sha,
                    "method": "api",
                }
            )
        logger.info("WDI %s (%s): %d records", zid, code, len(data))
    long = pd.DataFrame(rows)
    out = Path(v["cached_raw_path"])
    out = out if out.is_absolute() else root / out
    ensure_dir(out.parent)
    write_df(long, out)
    write_df(long, out.with_suffix(".csv"))
    write_json(out.with_name(out.stem + "_metadata.json"), meta)
    return long


def load_vintage(root: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    v = cfg["vintage"]
    path = Path(v["cached_raw_path"])
    path = path if path.is_absolute() else root / path
    if v.get("use_cached_raw", True) and path.exists():
        return pd.read_parquet(path)
    return fetch_vintage(root, cfg)


def cell_panel(long: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wide panel of Z values and the matching provenance grid (12 x 14 grid)."""
    y0, y1 = cfg["years"]
    k = int(cfg.get("carry_max_years", 2))
    grid = pd.MultiIndex.from_product([COUNTRIES, range(y0, y1 + 1)], names=["country_iso3", "year"])
    official = (
        long.dropna(subset=["value"])
        .pivot_table(index=["country_iso3", "year"], columns="indicator_id", values="value", aggfunc="first")
        .reindex(grid)
    )
    zids = list(cfg["indicators"].keys())
    official = official.reindex(columns=zids)
    filled = official.copy()
    prov = pd.DataFrame("missing", index=grid, columns=zids)
    for z in zids:
        s = official[z]
        prov.loc[s.notna(), z] = "official"
        f = s.groupby(level="country_iso3").transform(lambda x: x.ffill(limit=k).bfill(limit=k))
        carried = s.isna() & f.notna()
        prov.loc[carried, z] = "carried_forward"
        filled[z] = f
    return filled.reset_index(), prov.reset_index()


def series_table(prov: pd.DataFrame, filled: pd.DataFrame, ids: List[str], vintage_year: int) -> pd.DataFrame:
    """One row per indicator x country series: coverage, first/last official year, longest gap."""
    rows = []
    for iid in ids:
        for iso, g in prov.groupby("country_iso3"):
            g = g.sort_values("year")
            off = g.loc[g[iid] == "official", "year"].tolist()
            states = g[iid].tolist()
            longest, run = 0, 0
            for s in states:
                run = run + 1 if s != "official" else 0
                longest = max(longest, run)
            rows.append(
                {
                    "indicator_id": iid,
                    "country_iso3": iso,
                    "n_years": len(states),
                    "n_official": len(off),
                    "n_carried": int(sum(s == "carried_forward" for s in states)),
                    "n_reconstructed": int(sum(s == "mice_imputed" for s in states)),
                    "n_missing": int(sum(s == "missing" for s in states)),
                    "official_share": len(off) / len(states) if states else 0.0,
                    "first_official_year": min(off) if off else None,
                    "last_official_year": max(off) if off else None,
                    "publication_lag_years": (vintage_year - max(off)) if off else None,
                    "longest_non_official_run": longest,
                }
            )
    return pd.DataFrame(rows)


def run(project_root: Path | str, refresh: bool = False) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    cfg = load_config(root)
    long = fetch_vintage(root, cfg) if refresh else load_vintage(root, cfg)
    filled, prov = cell_panel(long, cfg)
    out = ensure_dir(root / "outputs" / "knowledge_layer" / "context")
    write_df(filled, out / "context_panel.csv")
    write_df(prov, out / "context_provenance.csv")
    zids = list(cfg["indicators"].keys())
    series = series_table(prov, filled, zids, vintage_year=2025)
    write_df(series, out / "context_series.csv")
    counts = {z: prov[z].value_counts().to_dict() for z in zids}
    write_json(out / "context_provenance_counts.json", counts)
    return {"n_values": int(long["value"].notna().sum()), "provenance": counts}
