"""Create lagged features, group dummies, and growth rates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from common.io import ensure_dir, read_df, write_df
from common.logging_utils import get_logger
from lineage.tracker import LineageTracker
from pipeline.pipeline_config import PipelineConfig

BASE_FEATURES: List[str] = [
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "X8",
    "X9",
    "X10",
    "C1",
    "C2",
]

GROWTH_VARS: List[str] = ["X1", "X2", "X3", "X4", "X8", "X9", "X10"]

logger = get_logger("dmf.feature_engineering")


def _feature_lag(config: PipelineConfig) -> int:
    research_lag = config.research.get("feature_lag")
    if research_lag is not None:
        return int(research_lag)
    ml = config.machine_learning or {}
    if ml.get("use_lagged_features"):
        return 1
    return 1


def _add_lags(
    df: pd.DataFrame,
    cols: Sequence[str],
    lag: int,
) -> pd.DataFrame:
    out = df.sort_values(["country_iso3", "year"]).copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[f"{col}_lag{lag}"] = out.groupby("country_iso3")[col].shift(lag)
    return out


def _add_group_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "group_id" not in out.columns:
        return out
    dummies = pd.get_dummies(out["group_id"], prefix="group", dtype=float)
    # Avoid column collisions
    for col in dummies.columns:
        out[col] = dummies[col].values
    return out


def _add_yoy_growth(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.sort_values(["country_iso3", "year"]).copy()
    for col in cols:
        if col not in out.columns:
            continue
        prev = out.groupby("country_iso3")[col].shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = (out[col] - prev) / prev.abs()
        growth = growth.replace([np.inf, -np.inf], np.nan)
        out[f"{col}_yoy"] = growth
    return out


def run(project_root: Path | str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Engineer features from the processed wide panel.

    Saves ``data/processed/panel_features.parquet``.
    """
    project_root = Path(project_root)
    wide_path = project_root / "data" / "processed" / "panel_wide.parquet"
    logger.info("Loading wide panel from %s", wide_path)
    panel = read_df(wide_path)

    lag = _feature_lag(config)
    feature_cols = [c for c in BASE_FEATURES if c in panel.columns]

    feats = panel.copy()
    feats = _add_lags(feats, feature_cols, lag=lag)
    feats = _add_group_dummies(feats)

    # Year as numeric feature (already present; ensure numeric dtype)
    feats["year"] = pd.to_numeric(feats["year"], errors="coerce")
    include_year = bool(
        (config.machine_learning or {}).get("include_year_feature", True)
    )
    if include_year and "year_num" not in feats.columns:
        feats["year_num"] = feats["year"].astype(float)

    feats = _add_yoy_growth(feats, [c for c in GROWTH_VARS if c in feats.columns])
    feats = feats.sort_values(["country_iso3", "year"]).reset_index(drop=True)

    out_path = write_df(
        feats,
        ensure_dir(project_root / "data" / "processed") / "panel_features.parquet",
    )

    tracker = LineageTracker(project_root)
    tracker.record(
        node_id="processed.panel_features",
        node_type="dataset",
        parent_ids=["processed.panel_wide"],
        transform="feature_engineering",
        path=out_path.relative_to(project_root),
    )

    lag_cols = [c for c in feats.columns if "_lag" in c]
    growth_cols = [c for c in feats.columns if c.endswith("_yoy")]
    group_cols = [
        c
        for c in feats.columns
        if str(c).startswith("group_") and c not in {"group_id", "group_name"}
    ]

    result = {
        "path": str(out_path),
        "rows": int(len(feats)),
        "columns": list(feats.columns),
        "feature_lag": lag,
        "lag_columns": lag_cols,
        "growth_columns": growth_cols,
        "group_dummy_columns": group_cols,
        "include_year_feature": include_year,
        "shape": [int(feats.shape[0]), int(feats.shape[1])],
    }
    logger.info("Feature engineering complete: path=%s rows=%s", out_path, result["rows"])
    return result
