"""MICE-style multivariate imputation via sklearn IterativeImputer + RandomForest.

Selected as the v2 filling method based on recent empirical research (2024–2026):
Multiple Imputation by Chained Equations (MICE) remains the dominant multivariate
approach; tree-based MICE variants are competitive with classical MICE-PMM.

The module exposes three levels of use:

``apply_mice_imputation``
    Single completed panel (conditional-mean or predictive-mean-matching draw)
    used as the point dataset for all downstream pipeline stages.
``multiple_imputation``
    ``m`` completed panels drawn with predictive mean matching and bootstrap
    resampling of the donor model, for Rubin's-rules inference.
``validate_imputation``
    Masked-cell benchmark that hides a fraction of *observed* official cells,
    re-imputes them, and scores reconstruction error against naive baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger
from common.seeds import set_global_seed

logger = get_logger("dmf.imputation")

DEFAULT_INDICATORS: List[str] = [
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


def _as_imputation_cfg(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "imputation"):
        return dict(getattr(config, "imputation") or {})
    if isinstance(config, dict):
        return dict(config.get("imputation") or config)
    return {}


def imputation_kwargs_from_config(config: Any) -> Dict[str, Any]:
    """Translate ``config/imputation.yaml`` into ``apply_mice_imputation`` kwargs."""
    cfg = _as_imputation_cfg(config)
    mice_cfg = cfg.get("mice") or {}
    pmm_cfg = cfg.get("pmm") or {}
    return {
        "seed": int(cfg.get("random_seed", 42)),
        "max_iter": int(mice_cfg.get("max_iter", 15)),
        "n_estimators": int(mice_cfg.get("n_estimators", 50)),
        "min_samples_leaf": int(mice_cfg.get("min_samples_leaf", 2)),
        "aux_features": list(cfg.get("aux_features") or ["year"]),
        "include_group_dummies": bool(cfg.get("include_group_dummies", True)),
        "include_country_dummies": bool(cfg.get("include_country_dummies", True)),
        "clip_to_observed_range": bool(cfg.get("clip_to_observed_range", True)),
        "donors": int(pmm_cfg.get("donors", 5)),
    }


def _build_design(
    df: pd.DataFrame,
    cols: Sequence[str],
    aux_features: Optional[Sequence[str]],
    include_group_dummies: bool,
    include_country_dummies: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """Assemble the numeric design matrix used by the chained-equation imputer."""
    frames: List[pd.DataFrame] = [df[list(cols)].apply(pd.to_numeric, errors="coerce")]
    aux_used: List[str] = []
    for aux in aux_features or []:
        if aux in df.columns and aux not in cols:
            frames.append(pd.to_numeric(df[aux], errors="coerce").to_frame(aux))
            aux_used.append(aux)
    if include_group_dummies and "group_id" in df.columns:
        dummies = pd.get_dummies(df["group_id"], prefix="grp", dtype=float)
        frames.append(dummies)
        aux_used.extend(list(dummies.columns))
    if include_country_dummies and "country_iso3" in df.columns:
        dummies = pd.get_dummies(df["country_iso3"], prefix="cty", dtype=float)
        frames.append(dummies)
        aux_used.extend(list(dummies.columns))
    design = pd.concat(frames, axis=1)
    return design, aux_used


def _pmm_draw(
    y_observed: np.ndarray,
    pred_observed: np.ndarray,
    pred_missing: np.ndarray,
    *,
    donors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Predictive mean matching: replace each missing cell with a randomly chosen
    donor drawn from the ``donors`` observed cases with closest predicted value.

    Keeps imputed values inside the observed support (no impossible index
    scores) and injects the between-draw variability required by Rubin's rules.
    """
    if len(y_observed) == 0:
        return pred_missing
    k = int(min(max(donors, 1), len(y_observed)))
    out = np.empty(len(pred_missing), dtype=float)
    for i, p in enumerate(pred_missing):
        distances = np.abs(pred_observed - p)
        candidate_idx = np.argpartition(distances, k - 1)[:k]
        out[i] = float(y_observed[rng.choice(candidate_idx)])
    return out


def apply_mice_imputation(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    seed: int = 42,
    max_iter: int = 15,
    n_estimators: int = 50,
    min_samples_leaf: int = 2,
    aux_features: Optional[Sequence[str]] = None,
    include_group_dummies: bool = True,
    include_country_dummies: bool = True,
    clip_to_observed_range: bool = True,
    pmm: bool = False,
    donors: int = 5,
    bootstrap_donor_model: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Impute ``value_cols`` with IterativeImputer (MICE-style) + RandomForest.

    With ``pmm=False`` the filled value is the chained-equation conditional
    expectation (the point panel used by downstream stages). With ``pmm=True``
    the conditional expectation is only a matching key: the stored value is a
    real observed value from a nearby donor, which is what makes repeated calls
    with different seeds a valid set of multiple imputations.

    Returns ``(imputed_df, mask_df, report)`` where mask is True for cells
    that were missing before imputation and filled.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = [c for c in value_cols if c in out.columns]
    if not cols:
        empty_mask = pd.DataFrame(index=out.index)
        return out, empty_mask, {"status": "skipped", "reason": "no_value_cols"}

    before_missing = {c: int(out[c].isna().sum()) for c in cols}
    mask = out[cols].isna().copy()

    ranges = {
        c: (
            float(pd.to_numeric(out[c], errors="coerce").min(skipna=True)),
            float(pd.to_numeric(out[c], errors="coerce").max(skipna=True)),
        )
        for c in cols
        if out[c].notna().any()
    }

    design, aux_used = _build_design(
        out, cols, aux_features, include_group_dummies, include_country_dummies
    )
    design_cols = list(design.columns)

    estimator = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=-1,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=int(max_iter),
        random_state=int(seed),
        sample_posterior=False,
        skip_complete=True,
    )

    arr = imputer.fit_transform(design.to_numpy(dtype=float))
    imputed_design = pd.DataFrame(arr, columns=design_cols, index=out.index)

    for c in cols:
        if not bool(mask[c].any()):
            continue
        filled = imputed_design[c]
        if pmm:
            # Re-fit a donor model on the completed design to score observed and
            # missing cells on a common scale, then match on predicted values.
            feature_cols = [x for x in design_cols if x != c]
            X_all = imputed_design[feature_cols].to_numpy(dtype=float)
            obs_idx = np.flatnonzero(~mask[c].to_numpy())
            mis_idx = np.flatnonzero(mask[c].to_numpy())
            y_obs = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)[obs_idx]
            fit_rows = obs_idx
            if bootstrap_donor_model and len(obs_idx) > 1:
                fit_rows = rng.choice(obs_idx, size=len(obs_idx), replace=True)
            donor_model = RandomForestRegressor(
                n_estimators=int(n_estimators),
                min_samples_leaf=int(min_samples_leaf),
                random_state=int(rng.integers(0, 2**31 - 1)),
                n_jobs=-1,
            )
            y_fit = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)[fit_rows]
            donor_model.fit(X_all[fit_rows], y_fit)
            pred_obs = donor_model.predict(X_all[obs_idx])
            pred_mis = donor_model.predict(X_all[mis_idx])
            drawn = _pmm_draw(y_obs, pred_obs, pred_mis, donors=donors, rng=rng)
            filled = filled.copy()
            filled.iloc[mis_idx] = drawn
        elif clip_to_observed_range and c in ranges:
            lo, hi = ranges[c]
            if np.isfinite(lo) and np.isfinite(hi):
                filled = filled.clip(lo, hi)
        # Only fill originally missing cells; keep observed official values intact
        out.loc[mask[c], c] = filled.loc[mask[c]]

    after_missing = {c: int(out[c].isna().sum()) for c in cols}
    report = {
        "method": "mice_iterative_rf_pmm" if pmm else "mice_iterative_rf",
        "sklearn_class": "IterativeImputer",
        "estimator": "RandomForestRegressor",
        "draw_type": "predictive_mean_matching" if pmm else "conditional_mean",
        "donors": int(donors) if pmm else None,
        "max_iter": int(max_iter),
        "n_estimators": int(n_estimators),
        "seed": int(seed),
        "aux_features": aux_used,
        "include_country_dummies": bool(include_country_dummies),
        "clip_to_observed_range": bool(clip_to_observed_range),
        "missing_before": before_missing,
        "missing_after": after_missing,
        "cells_imputed": {c: int(mask[c].sum()) for c in cols},
        "total_cells_imputed": int(mask.to_numpy().sum()),
        "n_rows": int(len(out)),
        "note": (
            "Observed official values are preserved; only originally missing "
            "cells are replaced. Imputed values are statistical reconstructions, "
            "not new official publications."
        ),
        "references": [
            "Pereira et al., In-Database Data Imputation (MICE), PACMMOD/SIGMOD 2024",
            "Tree-based imputation vs MICE PMM, arXiv:2401.09602 (2024)",
        ],
    }
    mask_out = mask.copy()
    if "country_iso3" in out.columns:
        mask_out.insert(0, "country_iso3", out["country_iso3"].values)
    if "year" in out.columns:
        mask_out.insert(1, "year", out["year"].values)
    return out, mask_out, report


def multiple_imputation(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    n_imputations: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Draw ``n_imputations`` completed panels for Rubin's-rules inference.

    Variability across draws comes from (a) bootstrap resampling of the donor
    model's training rows and (b) random selection among the matched donors,
    which is the standard tree-based approximation to proper multiple
    imputation when the conditional model has no closed-form posterior.
    """
    draws: List[pd.DataFrame] = []
    kwargs.pop("pmm", None)
    kwargs.pop("bootstrap_donor_model", None)
    kwargs.pop("seed", None)
    for d in range(int(n_imputations)):
        imputed, _mask, _report = apply_mice_imputation(
            df,
            value_cols,
            seed=int(seed) + d + 1,
            pmm=True,
            bootstrap_donor_model=True,
            **kwargs,
        )
        draws.append(imputed)
    meta = {
        "n_imputations": int(n_imputations),
        "draw_type": "predictive_mean_matching_bootstrap",
        "base_seed": int(seed),
    }
    return draws, meta


def impute_train_test(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    train_mask: np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 15,
    n_estimators: int = 50,
    min_samples_leaf: int = 2,
    aux_features: Optional[Sequence[str]] = None,
    include_group_dummies: bool = True,
    include_country_dummies: bool = True,
    clip_to_observed_range: bool = True,
    **_ignored: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Leakage-safe completion: the chained-equation models see training rows only.

    Test-period cells are still filled, but every conditional model behind those
    fills was estimated without any test-period information, which is what a
    hold-out evaluation of predictive accuracy requires.
    """
    set_global_seed(seed)
    out = df.reset_index(drop=True).copy()
    train_mask = np.asarray(train_mask, dtype=bool)
    cols = [c for c in value_cols if c in out.columns]
    mask = out[cols].isna().copy()

    design, aux_used = _build_design(
        out, cols, aux_features, include_group_dummies, include_country_dummies
    )
    estimator = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=-1,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=int(max_iter),
        random_state=int(seed),
        sample_posterior=False,
        skip_complete=True,
    )
    matrix = design.to_numpy(dtype=float)
    imputer.fit(matrix[train_mask])
    filled = pd.DataFrame(
        imputer.transform(matrix), columns=list(design.columns), index=out.index
    )

    for c in cols:
        if not bool(mask[c].any()):
            continue
        values = filled[c]
        if clip_to_observed_range:
            train_values = pd.to_numeric(out.loc[train_mask, c], errors="coerce")
            lo, hi = float(train_values.min()), float(train_values.max())
            if np.isfinite(lo) and np.isfinite(hi):
                values = values.clip(lo, hi)
        out.loc[mask[c], c] = values.loc[mask[c]]

    report = {
        "method": "mice_iterative_rf_train_only",
        "n_train_rows": int(train_mask.sum()),
        "n_rows": int(len(out)),
        "aux_features": aux_used,
        "cells_imputed": {c: int(mask[c].sum()) for c in cols},
    }
    return out, report


def _baseline_fills(
    df: pd.DataFrame, col: str, target_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """Naive reference fills used to benchmark MICE reconstruction quality."""
    work = df.copy()
    series = pd.to_numeric(work[col], errors="coerce")
    out: Dict[str, np.ndarray] = {}
    if "country_iso3" in work.columns:
        country_mean = series.groupby(work["country_iso3"]).transform("mean")
        out["country_mean"] = country_mean.to_numpy(dtype=float)[target_idx]
    if {"group_id", "year"}.issubset(work.columns):
        group_year_mean = series.groupby([work["group_id"], work["year"]]).transform("mean")
        out["group_year_mean"] = group_year_mean.to_numpy(dtype=float)[target_idx]
    out["global_median"] = np.full(len(target_idx), float(series.median(skipna=True)))
    return out


def validate_imputation(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    mask_fraction: float = 0.15,
    n_repeats: int = 5,
    seed: int = 42,
    configurations: Optional[Dict[str, Dict[str, Any]]] = None,
    base_kwargs: Optional[Dict[str, Any]] = None,
    eligible: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Masked-cell benchmark: hide observed cells, re-impute, score error.

    Returns a tidy frame with one row per (configuration, variable, repeat)
    holding RMSE / MAE / normalised RMSE against the hidden true values, plus
    the same metrics for naive baselines evaluated on identical cells.

    By default any non-missing cell of ``df`` may be hidden, which on the
    pre-imputation panel includes carried-forward copies of neighbouring
    publications. ``eligible`` (boolean per indicator, same row order as
    ``df``) narrows the pool; passing the official-cell mask scores the
    reconstruction on published values only.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    cols = [c for c in value_cols if c in df.columns]
    if eligible is not None:
        eligible = eligible.reset_index(drop=True)
    base_kwargs = dict(base_kwargs or {})
    base_kwargs.pop("seed", None)
    configurations = configurations or {
        "mice_group_only": {"include_country_dummies": False},
        "mice_country_aware": {"include_country_dummies": True},
    }

    rows: List[Dict[str, Any]] = []
    for repeat in range(int(n_repeats)):
        holdout: Dict[str, np.ndarray] = {}
        masked = df.copy()
        for c in cols:
            observed = pd.to_numeric(df[c], errors="coerce").notna().to_numpy()
            if eligible is not None and c in eligible.columns:
                observed &= eligible[c].fillna(False).to_numpy(dtype=bool)
            observed_idx = np.flatnonzero(observed)
            # Never blind a column so heavily that the imputer loses its anchor
            n_hide = int(np.floor(len(observed_idx) * float(mask_fraction)))
            if n_hide < 3 or len(observed_idx) - n_hide < 10:
                continue
            hidden = rng.choice(observed_idx, size=n_hide, replace=False)
            holdout[c] = hidden
            masked.iloc[hidden, masked.columns.get_loc(c)] = np.nan
        if not holdout:
            continue

        for config_name, overrides in configurations.items():
            kwargs = {**base_kwargs, **overrides}
            imputed, _mask, _report = apply_mice_imputation(
                masked, cols, seed=int(seed) + repeat, **kwargs
            )
            for c, hidden in holdout.items():
                truth = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[hidden]
                pred = pd.to_numeric(imputed[c], errors="coerce").to_numpy(dtype=float)[hidden]
                sd = float(np.nanstd(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)))
                err = pred - truth
                rows.append(
                    {
                        "configuration": config_name,
                        "variable": c,
                        "repeat": repeat,
                        "n_masked": int(len(hidden)),
                        "rmse": float(np.sqrt(np.nanmean(err**2))),
                        "mae": float(np.nanmean(np.abs(err))),
                        "nrmse_sd": float(np.sqrt(np.nanmean(err**2)) / sd) if sd else np.nan,
                    }
                )

        # Baselines on the identical hidden cells
        for c, hidden in holdout.items():
            truth = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[hidden]
            sd = float(np.nanstd(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)))
            for name, pred in _baseline_fills(masked, c, hidden).items():
                err = pred - truth
                rows.append(
                    {
                        "configuration": f"baseline_{name}",
                        "variable": c,
                        "repeat": repeat,
                        "n_masked": int(len(hidden)),
                        "rmse": float(np.sqrt(np.nanmean(err**2))),
                        "mae": float(np.nanmean(np.abs(err))),
                        "nrmse_sd": float(np.sqrt(np.nanmean(err**2)) / sd) if sd else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """Standalone entry: impute ``data/processed/panel_wide.parquet`` if present."""
    root = Path(project_root).resolve()
    cfg = _as_imputation_cfg(config)
    if not bool(cfg.get("enabled", True)):
        return {"status": "disabled"}

    panel_path = root / "data" / "processed" / "panel_wide.parquet"
    df = pd.read_parquet(panel_path)
    imputed, mask, report = apply_mice_imputation(
        df, DEFAULT_INDICATORS, **imputation_kwargs_from_config(config)
    )

    out_cfg = cfg.get("output") or {}
    mask_path = write_df(mask, root / out_cfg.get("mask_path", "data/processed/imputation_mask.parquet"))
    panel_imp = write_df(
        imputed,
        root / out_cfg.get("panel_path", "data/processed/panel_wide_imputed.parquet"),
    )
    # Overwrite panel_wide used by downstream stages
    wide_path = write_df(imputed, root / "data" / "processed" / "panel_wide.parquet")
    report_path = write_json(
        root / out_cfg.get("report_path", "outputs/audit/imputation_report.json"),
        {**report, "panel_wide": str(wide_path), "mask_path": str(mask_path), "panel_imputed": str(panel_imp)},
    )
    logger.info(
        "MICE imputation complete: filled=%s report=%s",
        report["total_cells_imputed"],
        report_path,
    )
    return {"report": report, "report_path": str(report_path), "mask_path": str(mask_path)}
