"""Construct the Digital Maturity Index (DMI) and sensitivity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.errors import InsufficientSampleError
from common.io import ensure_dir, read_df, write_df, write_json
from common.logging_utils import get_logger
from lineage.tracker import LineageTracker
from pipeline.pipeline_config import PipelineConfig

logger = get_logger("dmf.index")

MIN_DMI_OBS = 30


def _minmax_to_scale(
    series: pd.Series,
    feature_range: Tuple[float, float] = (0.0, 100.0),
) -> pd.Series:
    """Min-max normalize a series to ``feature_range`` using full-sample min/max."""
    lo, hi = feature_range
    values = pd.to_numeric(series, errors="coerce")
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        # Constant / empty → map observed values to mid-scale, keep NaN
        out = pd.Series(np.nan, index=series.index, dtype=float)
        observed = values.notna()
        if observed.any():
            out.loc[observed] = (lo + hi) / 2.0
        return out
    scaled = (values - vmin) / (vmax - vmin)
    return lo + scaled * (hi - lo)


def _pillar_scores(
    df: pd.DataFrame,
    pillars: Dict[str, Any],
    feature_range: Tuple[float, float] = (0.0, 100.0),
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Normalize constituent indicators to [0,100] on the full sample, then
    compute each pillar as the mean of available (non-NaN) constituents.
    """
    out = df.copy()
    pillar_indicator_map: Dict[str, List[str]] = {}
    normalized_cols: List[str] = []

    for pillar_name, spec in pillars.items():
        indicators = list((spec or {}).get("indicators") or [])
        pillar_indicator_map[pillar_name] = indicators
        present = [c for c in indicators if c in out.columns]
        norm_names = []
        for col in present:
            ncol = f"{col}_norm"
            out[ncol] = _minmax_to_scale(out[col], feature_range=feature_range)
            norm_names.append(ncol)
            normalized_cols.append(ncol)
        if norm_names:
            out[f"pillar_{pillar_name}"] = out[norm_names].mean(axis=1, skipna=True)
            # If all constituents NaN, mean is NaN — ensure explicit
            all_nan = out[norm_names].isna().all(axis=1)
            out.loc[all_nan, f"pillar_{pillar_name}"] = np.nan
        else:
            out[f"pillar_{pillar_name}"] = np.nan

    return out, pillar_indicator_map


def _weighted_dmi(
    df: pd.DataFrame,
    weights: Dict[str, float],
    pillar_names: Sequence[str],
) -> pd.Series:
    """Weighted sum of pillars; renormalize weights over available pillars per row."""
    scores = np.full(len(df), np.nan, dtype=float)
    pillar_cols = [f"pillar_{p}" for p in pillar_names]
    weight_vec = np.array([float(weights.get(p, 0.0)) for p in pillar_names], dtype=float)

    values = df[pillar_cols].to_numpy(dtype=float)
    for i in range(len(df)):
        row = values[i]
        available = ~np.isnan(row)
        if not available.any():
            continue
        w = weight_vec.copy()
        w[~available] = 0.0
        wsum = w.sum()
        if wsum <= 0:
            continue
        w = w / wsum
        scores[i] = float(np.nansum(row * w))
    return pd.Series(scores, index=df.index, name="DMI")


def _pca_score(df: pd.DataFrame, indicator_cols: Sequence[str]) -> pd.Series:
    """PCA alternative: first PC of standardized available indicators, scaled 0–100."""
    cols = [c for c in indicator_cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, name="DMI_pca")

    mat = df[cols].to_numpy(dtype=float)
    # Require at least 2 non-null indicators per row for a score
    row_ok = np.sum(~np.isnan(mat), axis=1) >= 2
    if row_ok.sum() < 2:
        return pd.Series(np.nan, index=df.index, name="DMI_pca")

    # Column-wise standardize using observed values
    means = np.nanmean(mat, axis=0)
    stds = np.nanstd(mat, axis=0, ddof=1)
    stds = np.where(stds == 0, 1.0, stds)
    standardized = (mat - means) / stds
    # Impute remaining NaN with 0 (mean) for PCA only
    standardized = np.where(np.isnan(standardized), 0.0, standardized)

    try:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(standardized)[:, 0]
    except Exception:  # noqa: BLE001
        # Fallback: SVD
        centered = standardized - standardized.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vt[0]

    score = pd.Series(np.nan, index=df.index, name="DMI_pca", dtype=float)
    score.loc[df.index[row_ok]] = pc1[row_ok]
    # Rescale observed PC1 to [0, 100]
    observed = score.dropna()
    if len(observed) == 0:
        return score
    vmin, vmax = float(observed.min()), float(observed.max())
    if vmax == vmin:
        score.loc[observed.index] = 50.0
    else:
        score.loc[observed.index] = 100.0 * (observed - vmin) / (vmax - vmin)
    # Rows with insufficient indicators stay NaN
    score.loc[~row_ok] = np.nan
    return score


def _weight_shock_sensitivity(
    df: pd.DataFrame,
    base_weights: Dict[str, float],
    pillar_names: Sequence[str],
    shock_pct: float,
    n_shocks: int,
    seed: int,
) -> pd.DataFrame:
    """Monte Carlo weight shocks (±shock_pct); return mean/std of DMI per row."""
    rng = np.random.default_rng(seed)
    pillar_cols = [f"pillar_{p}" for p in pillar_names]
    base_w = np.array([float(base_weights.get(p, 0.0)) for p in pillar_names], dtype=float)
    values = df[pillar_cols].to_numpy(dtype=float)

    sims = np.full((n_shocks, len(df)), np.nan, dtype=float)
    for s in range(n_shocks):
        shocks = rng.uniform(-shock_pct, shock_pct, size=len(base_w))
        w = base_w * (1.0 + shocks)
        w = np.clip(w, 0.0, None)
        for i in range(len(df)):
            row = values[i]
            available = ~np.isnan(row)
            if not available.any():
                continue
            ww = w.copy()
            ww[~available] = 0.0
            wsum = ww.sum()
            if wsum <= 0:
                continue
            ww = ww / wsum
            sims[s, i] = float(np.nansum(row * ww))

    return pd.DataFrame(
        {
            "DMI_shock_mean": np.nanmean(sims, axis=0),
            "DMI_shock_std": np.nanstd(sims, axis=0, ddof=1),
        },
        index=df.index,
    )


def _cronbach_alpha(items: pd.DataFrame) -> Optional[float]:
    """Cronbach's alpha on complete cases; None if insufficient items/rows."""
    clean = items.dropna()
    k = clean.shape[1]
    n = clean.shape[0]
    if k < 2 or n < 3:
        return None
    item_vars = clean.var(axis=0, ddof=1)
    total_var = clean.sum(axis=1).var(ddof=1)
    if total_var == 0 or pd.isna(total_var):
        return None
    alpha = (k / (k - 1.0)) * (1.0 - float(item_vars.sum()) / float(total_var))
    return float(alpha)


def _reliability(
    df: pd.DataFrame,
    indicator_cols: Sequence[str],
    pillar_cols: Sequence[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    ind_present = [c for c in indicator_cols if c in df.columns]
    pillar_present = [c for c in pillar_cols if c in df.columns]

    if cfg.get("item_correlations", True) and len(ind_present) >= 2:
        corr = df[ind_present].corr(method="pearson")
        report["indicator_pairwise_correlations"] = json.loads(
            corr.round(6).to_json(orient="split")
        )
        # Flatten upper triangle for convenience
        pairs = []
        for i, a in enumerate(ind_present):
            for b in ind_present[i + 1 :]:
                val = corr.loc[a, b]
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "correlation": None if pd.isna(val) else float(val),
                    }
                )
        report["indicator_correlation_pairs"] = pairs

    if cfg.get("item_correlations", True) and len(pillar_present) >= 2:
        pcorr = df[pillar_present].corr(method="pearson")
        report["pillar_pairwise_correlations"] = json.loads(
            pcorr.round(6).to_json(orient="split")
        )

    if cfg.get("cronbach_alpha", True):
        # Prefer normalized indicator items when present
        norm_cols = [f"{c}_norm" for c in ind_present if f"{c}_norm" in df.columns]
        items = df[norm_cols] if norm_cols else df[ind_present]
        report["cronbach_alpha_indicators"] = _cronbach_alpha(items)
        if pillar_present:
            report["cronbach_alpha_pillars"] = _cronbach_alpha(df[pillar_present])

    report["n_indicators"] = len(ind_present)
    report["n_pillars"] = len(pillar_present)
    return report


def build_dmi_panel(
    panel: pd.DataFrame,
    index_cfg: Dict[str, Any],
    *,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct DMI pillars/scores and sensitivity columns on an in-memory panel.

    Returns ``(scored_frame, meta)`` where meta includes ``n_dmi`` and reliability.
    """
    pillars = index_cfg.get("pillars", {}) or {}
    weights = dict(index_cfg.get("default_weights", {}) or {})
    scale = index_cfg.get("scale", [0, 100])
    feature_range = (float(scale[0]), float(scale[1]))
    sens_cfg = index_cfg.get("sensitivity", {}) or {}
    rel_cfg = index_cfg.get("reliability", {}) or {}

    all_indicators: List[str] = []
    for spec in pillars.values():
        all_indicators.extend(list((spec or {}).get("indicators") or []))
    all_indicators = list(dict.fromkeys(all_indicators))

    scored, pillar_map = _pillar_scores(panel, pillars, feature_range=feature_range)
    pillar_names = list(pillars.keys())
    if not weights:
        weights = {p: 1.0 / max(len(pillar_names), 1) for p in pillar_names}

    scored["DMI"] = _weighted_dmi(scored, weights, pillar_names)
    n_dmi = int(scored["DMI"].notna().sum())
    if n_dmi < MIN_DMI_OBS:
        raise InsufficientSampleError(
            f"Fewer than {MIN_DMI_OBS} non-null DMI observations (got {n_dmi})",
            details={"n_dmi": n_dmi, "min_required": MIN_DMI_OBS},
        )

    if sens_cfg.get("pca", True):
        scored["DMI_pca"] = _pca_score(scored, all_indicators)

    shock_pct = float(sens_cfg.get("weight_shock_pct", 0.20))
    n_shocks = int(sens_cfg.get("n_shocks", 20))
    shock_df = _weight_shock_sensitivity(
        scored,
        weights,
        pillar_names,
        shock_pct=shock_pct,
        n_shocks=n_shocks,
        seed=seed,
    )
    scored["DMI_shock_mean"] = shock_df["DMI_shock_mean"]
    scored["DMI_shock_std"] = shock_df["DMI_shock_std"]

    pillar_cols = [f"pillar_{p}" for p in pillar_names]
    reliability = _reliability(scored, all_indicators, pillar_cols, rel_cfg)
    meta = {
        "n_dmi": n_dmi,
        "pillars": pillar_names,
        "pillar_map": pillar_map,
        "weights": weights,
        "reliability": reliability,
        "indicators": all_indicators,
    }
    return scored, meta


def run(project_root: Path | str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Build DMI from the processed wide / features panel.

    Saves DMI panel, summary, sensitivity, reliability, and analysis panel.
    """
    project_root = Path(project_root)
    index_cfg = config.index or {}

    # Prefer features panel (has group dummies etc.) but fall back to wide
    features_path = project_root / "data" / "processed" / "panel_features.parquet"
    wide_path = project_root / "data" / "processed" / "panel_wide.parquet"
    source_path = features_path if features_path.exists() else wide_path
    logger.info("Building DMI from %s", source_path)
    panel = read_df(source_path)

    scored, dmi_meta = build_dmi_panel(panel, index_cfg, seed=config.random_seed())
    all_indicators = list(dmi_meta["indicators"])
    pillar_names = list(dmi_meta["pillars"])
    weights = dict(dmi_meta["weights"])
    n_dmi = int(dmi_meta["n_dmi"])
    reliability = dmi_meta["reliability"]
    pillar_cols = [f"pillar_{p}" for p in pillar_names]

    # Lag of DMI for analysis
    lag = int(config.research.get("feature_lag", 1))
    scored = scored.sort_values(["country_iso3", "year"]).reset_index(drop=True)
    scored[f"DMI_lag{lag}"] = scored.groupby("country_iso3")["DMI"].shift(lag)

    # Persist DMI panel (core columns + group)
    core_cols = [
        "country_iso3",
        "year",
        "group_id",
        "group_name",
        *all_indicators,
        *pillar_cols,
        "DMI",
        "DMI_pca",
        "DMI_shock_mean",
        "DMI_shock_std",
        f"DMI_lag{lag}",
    ]
    core_cols = [c for c in core_cols if c in scored.columns]
    dmi_panel = scored[core_cols].copy()

    out_data = ensure_dir(project_root / "outputs" / "data")
    processed = ensure_dir(project_root / "data" / "processed")
    path_outputs = write_df(dmi_panel, out_data / "dmi_panel.parquet")
    path_processed = write_df(dmi_panel, processed / "dmi_panel.parquet")

    # Descriptive summary
    summary_rows = []
    summary_rows.append(
        {
            "metric": "DMI",
            "n": n_dmi,
            "mean": float(scored["DMI"].mean()),
            "std": float(scored["DMI"].std(ddof=1)),
            "min": float(scored["DMI"].min()),
            "p25": float(scored["DMI"].quantile(0.25)),
            "median": float(scored["DMI"].median()),
            "p75": float(scored["DMI"].quantile(0.75)),
            "max": float(scored["DMI"].max()),
        }
    )
    if "group_id" in scored.columns:
        for gid, gdf in scored.groupby("group_id", dropna=False):
            series = gdf["DMI"].dropna()
            if series.empty:
                continue
            summary_rows.append(
                {
                    "metric": f"DMI|group={gid}",
                    "n": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "p25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "p75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                }
            )
    for pcol in pillar_cols:
        series = scored[pcol].dropna()
        if series.empty:
            continue
        summary_rows.append(
            {
                "metric": pcol,
                "n": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )

    desc_dir = ensure_dir(project_root / "results" / "descriptive")
    summary_path = desc_dir / "dmi_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # Sensitivity table
    sens_table = scored[
        ["country_iso3", "year", "DMI", "DMI_pca", "DMI_shock_mean", "DMI_shock_std"]
    ].copy()
    # Keep only columns that exist
    sens_table = sens_table[[c for c in sens_table.columns if c in scored.columns]]
    tables_dir = ensure_dir(project_root / "outputs" / "tables")
    sens_path = tables_dir / "dmi_sensitivity.csv"
    sens_table.to_csv(sens_path, index=False)

    reliability_path = write_json(tables_dir / "dmi_reliability.json", reliability)

    # Analysis panel = features + DMI columns
    analysis = scored.copy()
    analysis_path = write_df(analysis, processed / "analysis_panel.parquet")

    tracker = LineageTracker(project_root)
    parent = (
        "processed.panel_features"
        if source_path == features_path
        else "processed.panel_wide"
    )
    tracker.record(
        node_id="processed.dmi_panel",
        node_type="dataset",
        parent_ids=[parent],
        transform="build_dmi",
        path=path_processed.relative_to(project_root),
    )
    tracker.record(
        node_id="processed.analysis_panel",
        node_type="dataset",
        parent_ids=["processed.dmi_panel", parent],
        transform="merge_dmi_features",
        path=analysis_path.relative_to(project_root),
    )

    result = {
        "dmi_panel_outputs": str(path_outputs),
        "dmi_panel_processed": str(path_processed),
        "analysis_panel": str(analysis_path),
        "summary_path": str(summary_path),
        "sensitivity_path": str(sens_path),
        "reliability_path": str(reliability_path),
        "n_dmi": n_dmi,
        "rows": int(len(dmi_panel)),
        "pillar_names": pillar_names,
        "weights": weights,
        "cronbach_alpha_indicators": reliability.get("cronbach_alpha_indicators"),
        "shape": [int(dmi_panel.shape[0]), int(dmi_panel.shape[1])],
    }
    logger.info(
        "DMI complete: n_dmi=%s mean=%.3f path=%s",
        n_dmi,
        float(scored["DMI"].mean()),
        path_processed,
    )
    return result
