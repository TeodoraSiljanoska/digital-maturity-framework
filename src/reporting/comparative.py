"""Compare the four country groups: mean DMI, growth, and top factors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.io import ensure_dir, read_df, write_df, write_json
from common.logging_utils import get_logger
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.reporting.comparative")


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "countries"):
        if hasattr(config, key):
            out[key] = getattr(config, key) or {}
    return out


def _load_panel(root: Path) -> pd.DataFrame:
    for path in (
        root / "data" / "processed" / "analysis_panel.parquet",
        root / "outputs" / "data" / "dmi_panel.parquet",
        root / "data" / "processed" / "dmi_panel.parquet",
    ):
        if path.exists():
            return read_df(path)
    raise FileNotFoundError("No DMI/analysis panel available for comparative analysis")


def _configured_groups(config: Any) -> List[str]:
    countries = _as_mapping(config).get("countries") or {}
    groups = countries.get("groups") or {}
    if groups:
        return list(groups.keys())
    return ["balkan", "developed_eu", "developed_non_eu", "developing"]


def _growth_by_group(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in panel.groupby("group_id"):
        yearly = g.groupby("year")["DMI"].mean().sort_index()
        if yearly.empty:
            continue
        start_y, end_y = int(yearly.index.min()), int(yearly.index.max())
        start_v, end_v = float(yearly.iloc[0]), float(yearly.iloc[-1])
        n_years = max(end_y - start_y, 1)
        abs_growth = end_v - start_v
        pct_growth = (end_v / start_v - 1.0) if start_v else np.nan
        cagr = (end_v / start_v) ** (1.0 / n_years) - 1.0 if start_v > 0 else np.nan
        rows.append(
            {
                "group_id": gid,
                "start_year": start_y,
                "end_year": end_y,
                "mean_dmi_start": start_v,
                "mean_dmi_end": end_v,
                "abs_growth": abs_growth,
                "pct_growth": pct_growth,
                "cagr": cagr,
                "n_obs": int(len(g)),
                "n_countries": int(g["country_iso3"].nunique())
                if "country_iso3" in g.columns
                else None,
            }
        )
    return pd.DataFrame(rows)


def _top_factors_by_group(panel: pd.DataFrame, factor_cols: List[str], top_k: int = 5) -> pd.DataFrame:
    rows = []
    for gid, g in panel.groupby("group_id"):
        present = [c for c in factor_cols if c in g.columns]
        if not present or "DMI" not in g.columns:
            continue
        corrs = g[present + ["DMI"]].corr()["DMI"].drop(labels=["DMI"], errors="ignore")
        corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
        for rank, (feat, val) in enumerate(corrs.head(top_k).items(), start=1):
            rows.append(
                {
                    "group_id": gid,
                    "rank": rank,
                    "factor": feat,
                    "corr_with_dmi": float(val) if pd.notna(val) else np.nan,
                    "abs_corr": float(abs(val)) if pd.notna(val) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Compare configured country groups on mean DMI, growth, and top correlated factors.

    Writes ``results/comparative/``.
    """
    root = Path(project_root).resolve()
    panel = _load_panel(root)
    if "group_id" not in panel.columns or "DMI" not in panel.columns:
        raise ValueError("Comparative analysis requires group_id and DMI columns")

    configured = _configured_groups(config)
    present_groups = sorted(panel["group_id"].dropna().unique().tolist())

    mean_dmi = (
        panel.groupby("group_id")["DMI"]
        .agg(n="count", mean_dmi="mean", std_dmi="std", min_dmi="min", max_dmi="max")
        .reset_index()
    )
    growth = _growth_by_group(panel)

    factor_cols = [c for c in panel.columns if c.startswith("X") or c.startswith("C")]
    # Prefer contemporaneous X/C over lags for interpretability
    factor_cols = [c for c in factor_cols if "_lag" not in c] or factor_cols
    top_factors = _top_factors_by_group(panel, factor_cols)

    out_dir = ensure_dir(root / "results" / "comparative")
    write_df(mean_dmi, out_dir / "mean_dmi_by_group.csv")
    write_df(growth, out_dir / "growth_by_group.csv")
    write_df(top_factors, out_dir / "top_factors_by_group.csv")

    # Wide comparison table
    merged = mean_dmi.merge(growth, on="group_id", how="outer")
    write_df(merged, out_dir / "group_comparison.csv")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configured_groups": configured,
        "present_groups": present_groups,
        "missing_groups": [g for g in configured if g not in present_groups],
        "n_groups_present": len(present_groups),
        "artifacts": {
            "mean_dmi_by_group": "results/comparative/mean_dmi_by_group.csv",
            "growth_by_group": "results/comparative/growth_by_group.csv",
            "top_factors_by_group": "results/comparative/top_factors_by_group.csv",
            "group_comparison": "results/comparative/group_comparison.csv",
        },
    }
    write_json(out_dir / "comparative_summary.json", summary)

    registry = ResultRegistry(root)
    registry.register(
        "comparative_group_summary",
        summary,
        category="comparative",
        fmt="json",
    )
    registry.register(
        "comparative_group_comparison",
        merged,
        category="comparative",
        fmt="csv",
    )

    logger.info(
        "Comparative analysis for %d groups (configured %d)",
        len(present_groups),
        len(configured),
    )
    return summary
