"""Descriptive statistics, correlations, moments, and VIF diagnostics."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger
from common.seeds import set_global_seed
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.statistics")

DEFAULT_X = [f"X{i}" for i in range(1, 11)]
DEFAULT_C = ["C1", "C2"]
ID_COLS = ["country_iso3", "year", "group_id"]


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in (
        "research",
        "econometrics",
        "machine_learning",
        "preprocessing",
        "index",
    ):
        if hasattr(config, key):
            out[key] = getattr(config, key) or {}
    return out


def _seed(config: Any) -> int:
    if hasattr(config, "random_seed"):
        try:
            return int(config.random_seed())
        except Exception:
            pass
    mapping = _as_mapping(config)
    research = mapping.get("research") or {}
    return int(research.get("random_seed", 42))


def load_analysis_panel(project_root: Path) -> pd.DataFrame:
    """
    Load analysis panel from processed path, with fallback merge of DMI + features.
    """
    root = Path(project_root).resolve()
    primary = root / "data" / "processed" / "analysis_panel.parquet"
    if primary.exists():
        logger.info("Loading analysis panel from %s", primary)
        return pd.read_parquet(primary)

    dmi_path = root / "outputs" / "data" / "dmi_panel.parquet"
    feature_candidates = [
        root / "data" / "processed" / "features.parquet",
        root / "outputs" / "data" / "features.parquet",
        root / "data" / "processed" / "feature_panel.parquet",
        root / "outputs" / "data" / "feature_panel.parquet",
    ]
    if not dmi_path.exists():
        raise FileNotFoundError(
            "No analysis panel found. Expected "
            f"{primary} or {dmi_path} (+ features)."
        )

    dmi = pd.read_parquet(dmi_path)
    features = None
    for cand in feature_candidates:
        if cand.exists():
            features = pd.read_parquet(cand)
            logger.info("Merging DMI panel with features from %s", cand)
            break

    if features is None:
        logger.warning("Features file not found; using DMI panel alone")
        return dmi

    keys = [c for c in ("country_iso3", "year") if c in dmi.columns and c in features.columns]
    if not keys:
        raise ValueError("Cannot merge DMI and features: missing country_iso3/year keys")
    # Prefer feature columns when overlapping (except keys / DMI)
    overlap = [c for c in features.columns if c in dmi.columns and c not in keys and c != "DMI"]
    dmi_keep = dmi.drop(columns=overlap, errors="ignore")
    merged = dmi_keep.merge(features, on=keys, how="left")
    return merged


def ensure_lagged_features(
    df: pd.DataFrame,
    base_cols: Sequence[str],
    *,
    entity_col: str = "country_iso3",
    time_col: str = "year",
    lag: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create ``{col}_lag{lag}`` columns when missing; return df and lag names."""
    out = df.copy()
    if entity_col not in out.columns or time_col not in out.columns:
        raise ValueError(f"Panel must contain {entity_col} and {time_col}")
    out = out.sort_values([entity_col, time_col])
    lag_names: List[str] = []
    for col in base_cols:
        if col not in out.columns:
            continue
        lag_name = f"{col}_lag{lag}"
        lag_names.append(lag_name)
        if lag_name not in out.columns:
            out[lag_name] = out.groupby(entity_col, sort=False)[col].shift(lag)
    return out, lag_names


def predictor_columns(
    df: pd.DataFrame,
    *,
    independent: Optional[Sequence[str]] = None,
    controls: Optional[Sequence[str]] = None,
    use_lagged: bool = True,
    lag: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Resolve predictor column names (lagged when configured) and ensure they exist."""
    indep = list(independent or DEFAULT_X)
    ctrls = list(controls or DEFAULT_C)
    base = [c for c in indep + ctrls if c in df.columns or use_lagged]
    # Keep bases that exist as raw columns (needed to build lags)
    raw_needed = [c for c in indep + ctrls if c in df.columns]
    work = df.copy()
    if use_lagged:
        work, lag_names = ensure_lagged_features(work, raw_needed, lag=lag)
        # Prefer lag columns; fall back to raw if lag somehow unavailable
        preds: List[str] = []
        for col in indep + ctrls:
            lag_name = f"{col}_lag{lag}"
            if lag_name in work.columns:
                preds.append(lag_name)
            elif col in work.columns:
                preds.append(col)
        return work, preds
    preds = [c for c in indep + ctrls if c in work.columns]
    return work, preds


def _numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _describe_block(df: pd.DataFrame, cols: Sequence[str], label: str) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    desc = df[list(cols)].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    desc = desc.rename(columns={"index": "variable"})
    desc.insert(0, "sample", label)
    # Extra moments
    extra = []
    for c in cols:
        s = df[c].dropna()
        extra.append(
            {
                "variable": c,
                "skew": float(s.skew()) if len(s) else np.nan,
                "kurtosis": float(s.kurtosis()) if len(s) else np.nan,
                "n_missing": int(df[c].isna().sum()),
            }
        )
    extra_df = pd.DataFrame(extra)
    return desc.merge(extra_df, on="variable", how="left")


def _correlation_matrix(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return pd.DataFrame()
    return df[use].corr(method="pearson")


def _dmi_moments_and_shapiro(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna().astype(float)
    payload: Dict[str, Any] = {
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "variance": float(s.var(ddof=1)) if len(s) > 1 else np.nan,
        "skewness": float(s.skew()) if len(s) else np.nan,
        "kurtosis": float(s.kurtosis()) if len(s) else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "shapiro_stat": None,
        "shapiro_pvalue": None,
        "shapiro_applied": False,
        "shapiro_note": None,
    }
    # Shapiro requires 3 <= n <= 5000
    if 3 <= len(s) <= 5000:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval = stats.shapiro(s.values)
        payload["shapiro_stat"] = float(stat)
        payload["shapiro_pvalue"] = float(pval)
        payload["shapiro_applied"] = True
    elif len(s) > 5000:
        payload["shapiro_note"] = "n>5000; Shapiro skipped (sample too large)"
    else:
        payload["shapiro_note"] = "n<3; Shapiro not applicable"
    return payload


def _vif_table(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in predictors if c in df.columns]
    clean = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    if clean.shape[0] < len(cols) + 2 or clean.shape[1] < 2:
        return pd.DataFrame(
            [{"variable": c, "vif": np.nan, "note": "insufficient_sample"} for c in cols]
        )
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return pd.DataFrame(
            [{"variable": c, "vif": np.nan, "note": "statsmodels_unavailable"} for c in cols]
        )

    x = clean.to_numpy(dtype=float)
    # Drop zero-variance columns for VIF stability
    keep_idx = [i for i in range(x.shape[1]) if np.nanstd(x[:, i]) > 0]
    if len(keep_idx) < 2:
        return pd.DataFrame(
            [{"variable": c, "vif": np.nan, "note": "zero_variance"} for c in cols]
        )
    x = x[:, keep_idx]
    kept_cols = [cols[i] for i in keep_idx]
    # statsmodels regresses column i on the remaining columns without adding an
    # intercept, so an explicit constant is required; without it every
    # non-centred predictor reports a wildly inflated VIF.
    design = np.column_stack([np.ones(len(x)), x])
    for i, name in enumerate(kept_cols):
        try:
            vif = float(variance_inflation_factor(design, i + 1))
        except Exception as exc:  # pragma: no cover
            rows.append({"variable": name, "vif": np.nan, "note": repr(exc)})
            continue
        rows.append({"variable": name, "vif": vif, "note": None})
    present = {r["variable"] for r in rows}
    for c in cols:
        if c not in present:
            rows.append({"variable": c, "vif": np.nan, "note": "excluded_zero_variance"})
    return pd.DataFrame(rows)


def _copy_to_outputs_tables(src: Path, tables_dir: Path) -> Path:
    ensure_dir(tables_dir)
    dest = tables_dir / src.name
    shutil.copy2(src, dest)
    return dest


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Compute descriptive statistics, correlations, DMI moments/Shapiro, and VIF.

    Saves under ``results/descriptive/`` and copies key tables to ``outputs/tables/``.
    """
    root = Path(project_root).resolve()
    set_global_seed(_seed(config))
    mapping = _as_mapping(config)
    eco = mapping.get("econometrics") or {}
    research = mapping.get("research") or {}

    independent = list(eco.get("independent") or DEFAULT_X)
    controls = list(eco.get("controls") or DEFAULT_C)
    use_lagged = bool(eco.get("use_lagged_features", True))
    lag = int((research.get("feature_lag") or 1))

    df = load_analysis_panel(root)
    df, predictors = predictor_columns(
        df,
        independent=independent,
        controls=controls,
        use_lagged=use_lagged,
        lag=lag,
    )

    target = str(eco.get("dependent") or research.get("dependent_variable") or "DMI")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' missing from analysis panel")

    numeric_interest = _numeric_cols(
        df, [target] + independent + controls + predictors
    )
    # Deduplicate preserving order
    seen = set()
    numeric_interest = [c for c in numeric_interest if not (c in seen or seen.add(c))]

    results_dir = ensure_dir(root / "results" / "descriptive")
    tables_dir = ensure_dir(root / "outputs" / "tables")
    registry = ResultRegistry(root)

    # --- Full-sample descriptive ---
    full_desc = _describe_block(df, numeric_interest, "full_sample")
    full_path = results_dir / "descriptive_full_sample.csv"
    write_df(full_desc, full_path)
    _copy_to_outputs_tables(full_path, tables_dir)
    registry.register(
        "descriptive_full_sample",
        full_desc,
        category="descriptive",
        meta={"sample": "full_sample"},
    )

    # --- By group ---
    group_frames: List[pd.DataFrame] = []
    if "group_id" in df.columns:
        for gid, gdf in df.groupby("group_id", dropna=False):
            block = _describe_block(gdf, numeric_interest, str(gid))
            if not block.empty:
                group_frames.append(block)
    group_desc = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    group_path = results_dir / "descriptive_by_group.csv"
    write_df(group_desc, group_path)
    _copy_to_outputs_tables(group_path, tables_dir)
    registry.register(
        "descriptive_by_group",
        group_desc,
        category="descriptive",
        meta={"sample": "by_group"},
    )

    # --- Correlation ---
    corr_cols = [c for c in [target] + predictors if c in df.columns]
    corr = _correlation_matrix(df, corr_cols)
    corr_path = results_dir / "correlation_matrix.csv"
    if not corr.empty:
        corr_out = corr.reset_index().rename(columns={"index": "variable"})
        write_df(corr_out, corr_path)
        _copy_to_outputs_tables(corr_path, tables_dir)
        registry.register(
            "correlation_matrix",
            corr_out,
            category="descriptive",
            meta={"variables": corr_cols},
        )
    else:
        write_json(corr_path.with_suffix(".json"), {"note": "insufficient columns"})

    # --- DMI moments / Shapiro ---
    moments = _dmi_moments_and_shapiro(df[target])
    moments_path = results_dir / "dmi_distribution_moments.json"
    write_json(moments_path, moments)
    moments_csv = pd.DataFrame([moments])
    moments_csv_path = results_dir / "dmi_distribution_moments.csv"
    write_df(moments_csv, moments_csv_path)
    _copy_to_outputs_tables(moments_csv_path, tables_dir)
    registry.register(
        "dmi_distribution_moments_result",
        moments,
        category="descriptive",
        meta={"target": target, "path": str(moments_path.relative_to(root))},
        fmt="json",
    )

    # Group-level Shapiro summary
    group_moments_rows = []
    if "group_id" in df.columns:
        for gid, gdf in df.groupby("group_id", dropna=False):
            m = _dmi_moments_and_shapiro(gdf[target])
            m["group_id"] = gid
            group_moments_rows.append(m)
    if group_moments_rows:
        gm_df = pd.DataFrame(group_moments_rows)
        gm_path = results_dir / "dmi_moments_by_group.csv"
        write_df(gm_df, gm_path)
        _copy_to_outputs_tables(gm_path, tables_dir)
        registry.register("dmi_moments_by_group", gm_df, category="descriptive")

    # --- VIF ---
    vif_preds = [c for c in predictors if c in df.columns]
    if len(df.dropna(subset=vif_preds)) < 10:
        logger.warning("Small sample for VIF; proceeding with available rows")
    vif_df = _vif_table(df, vif_preds)
    vif_path = results_dir / "vif_predictors.csv"
    write_df(vif_df, vif_path)
    _copy_to_outputs_tables(vif_path, tables_dir)
    registry.register(
        "vif_predictors",
        vif_df,
        category="descriptive",
        meta={"predictors": vif_preds, "use_lagged": use_lagged},
    )

    summary = {
        "n_rows": int(len(df)),
        "n_countries": int(df["country_iso3"].nunique()) if "country_iso3" in df.columns else None,
        "years": sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
        "target": target,
        "predictors": predictors,
        "use_lagged_features": use_lagged,
        "artifacts": {
            "descriptive_full_sample": str(full_path.relative_to(root)),
            "descriptive_by_group": str(group_path.relative_to(root)),
            "correlation_matrix": str(corr_path.relative_to(root)) if corr_path.exists() else None,
            "dmi_moments": str(moments_path.relative_to(root)),
            "vif": str(vif_path.relative_to(root)),
        },
    }
    summary_path = results_dir / "descriptive_summary.json"
    write_json(summary_path, summary)
    registry.register(
        "descriptive_summary_result",
        summary,
        category="descriptive",
        fmt="json",
        meta={"path": str(summary_path.relative_to(root))},
    )
    logger.info("Descriptive statistics complete (%s rows)", summary["n_rows"])
    return summary
