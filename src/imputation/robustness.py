"""Robustness layer for the imputed (v2) research track.

A single completed panel is convenient for descriptives, plots and dashboards,
but it is not sufficient evidence. This module produces the diagnostics a
reviewer needs before trusting any result estimated on reconstructed cells:

1. ``imputation_shares``    — how much of each variable, group and country is reconstructed.
2. ``imputation_validation``— masked-cell benchmark of reconstruction error vs naive fills.
3. ``mi_pooled_fe``         — fixed-effects inference pooled over m imputations (Rubin's rules).
4. ``complete_case_fe``     — the same specification on observed cells only.
5. ``leakage_safe_ml``      — predictive horse-race with the imputer fitted on training years only.

When imputation is disabled (the v1 track) the module writes a
``not_applicable`` marker so both versions expose the same artifact contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from common.io import ensure_dir, read_df, write_df, write_json
from common.logging_utils import get_logger
from imputation.mice import (
    DEFAULT_INDICATORS,
    impute_train_test,
    imputation_kwargs_from_config,
    multiple_imputation,
    validate_imputation,
)
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.imputation.robustness")


def _cfg(config: Any, name: str) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, name):
        return dict(getattr(config, name) or {})
    if isinstance(config, dict):
        return dict(config.get(name) or {})
    return {}


def _prepare_draw(
    draw: pd.DataFrame,
    *,
    index_cfg: Dict[str, Any],
    log_controls: Sequence[str],
    independent: Sequence[str],
    controls: Sequence[str],
    lag: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply the downstream transform chain (log → DMI → lags) to one panel."""
    from index.build import build_dmi_panel
    from preprocessing.pipeline import _apply_log_transforms
    from statistics.descriptive import predictor_columns

    work = _apply_log_transforms(draw, list(log_controls))
    scored, _meta = build_dmi_panel(work, index_cfg, seed=seed)
    scored = scored.sort_values(["country_iso3", "year"]).reset_index(drop=True)
    scored, predictors = predictor_columns(
        scored,
        independent=list(independent),
        controls=list(controls),
        use_lagged=True,
        lag=lag,
    )
    return scored, predictors


def _fit_fe(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    *,
    target: str = "DMI",
    entity_col: str = "country_iso3",
    time_col: str = "year",
) -> Optional[pd.DataFrame]:
    from econometrics.models import _fit_fe_linearmodels, _prepare_estimation_frame

    panel = _prepare_estimation_frame(frame, target, predictors, entity_col, time_col)
    if panel.empty or panel[entity_col].nunique() < 2:
        return None
    _res, coef, metrics = _fit_fe_linearmodels(
        panel, target, list(predictors), entity_col, time_col
    )
    coef = coef.copy()
    coef["nobs"] = metrics.get("nobs")
    return coef


def _rubin_pool(per_draw: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-imputation FE estimates with Rubin's (1987) rules."""
    m = len(per_draw)
    stacked = pd.concat(per_draw, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for term, grp in stacked.groupby("term"):
        q = grp["coefficient"].to_numpy(dtype=float)
        u = grp["std_error"].to_numpy(dtype=float) ** 2
        if len(q) < 2 or not np.isfinite(q).all():
            continue
        q_bar = float(np.mean(q))
        u_bar = float(np.mean(u))
        b = float(np.var(q, ddof=1))
        total = u_bar + (1.0 + 1.0 / m) * b
        se = float(np.sqrt(total)) if total > 0 else np.nan
        between_share = (1.0 + 1.0 / m) * b
        fmi = float(between_share / total) if total > 0 else np.nan
        if between_share > 0:
            df = (m - 1) * (1.0 + u_bar / between_share) ** 2
        else:
            df = float("inf")
        t_stat = q_bar / se if se and np.isfinite(se) and se > 0 else np.nan
        pval = (
            float(2.0 * stats.t.sf(abs(t_stat), df=max(df, 1.0)))
            if np.isfinite(t_stat)
            else np.nan
        )
        rows.append(
            {
                "term": term,
                "coefficient": q_bar,
                "std_error": se,
                "within_variance": u_bar,
                "between_variance": b,
                "pvalue": pval,
                "t_statistic": t_stat,
                "df": float(df),
                "fraction_missing_information": fmi,
                "n_imputations": m,
            }
        )
    return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)


def _imputation_shares(mask: pd.DataFrame, cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    present = [c for c in cols if c in mask.columns]
    by_variable = pd.DataFrame(
        {
            "variable": present,
            "n_imputed": [int(mask[c].sum()) for c in present],
            "n_rows": [int(len(mask))] * len(present),
        }
    )
    by_variable["share_imputed"] = by_variable["n_imputed"] / by_variable["n_rows"]
    by_variable = by_variable.sort_values("share_imputed", ascending=False).reset_index(drop=True)

    frames = {"by_variable": by_variable}
    if "country_iso3" in mask.columns:
        by_country = (
            mask.groupby("country_iso3")[present]
            .mean()
            .mean(axis=1)
            .rename("share_imputed")
            .reset_index()
            .sort_values("share_imputed", ascending=False)
        )
        frames["by_country"] = by_country
    if "year" in mask.columns:
        by_year = (
            mask.groupby("year")[present]
            .mean()
            .mean(axis=1)
            .rename("share_imputed")
            .reset_index()
        )
        frames["by_year"] = by_year
    return frames


def _leakage_safe_ml(
    preimp: pd.DataFrame,
    *,
    config: Any,
    imp_kwargs: Dict[str, Any],
    index_cfg: Dict[str, Any],
    log_controls: Sequence[str],
    independent: Sequence[str],
    controls: Sequence[str],
    lag: int,
    seed: int,
    test_years: Sequence[int],
    model_types: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Re-run the H1.6 horse-race with a training-years-only imputer."""
    from machine_learning.train import (
        _build_estimator,
        _fit_with_search,
        regression_metrics,
        temporal_split,
    )

    min_test = min(int(y) for y in test_years)
    train_mask = preimp["year"].astype(int).to_numpy() < min_test
    completed, imp_report = impute_train_test(
        preimp, DEFAULT_INDICATORS, train_mask, **imp_kwargs
    )
    scored, predictors = _prepare_draw(
        completed,
        index_cfg=index_cfg,
        log_controls=log_controls,
        independent=independent,
        controls=controls,
        lag=lag,
        seed=seed,
    )
    work = scored.dropna(subset=["DMI"]).copy()
    train_df, test_df = temporal_split(work, time_col="year", test_years=list(test_years))
    usable = [c for c in predictors if c in work.columns]
    train_df = train_df.dropna(subset=usable)
    test_df = test_df.dropna(subset=usable)

    rows: List[Dict[str, Any]] = []
    X_train, y_train = train_df[usable], train_df["DMI"]
    X_test, y_test = test_df[usable], test_df["DMI"]
    for model_type in model_types:
        estimator, warn = _build_estimator(model_type, seed)
        if estimator is None:
            rows.append({"model_type": model_type, "status": "skipped", "warning": warn})
            continue
        try:
            model, _meta = _fit_with_search(
                model_type,
                estimator,
                X_train,
                y_train,
                seed=seed,
                n_iter=8,
                cv_folds=3,
                scoring="neg_root_mean_squared_error",
            )
            metrics = regression_metrics(y_test, model.predict(X_test))
            rows.append({"model_type": model_type, "status": "ok", **metrics})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Leakage-safe training failed for %s: %s", model_type, exc)
            rows.append({"model_type": model_type, "status": "failed", "warning": repr(exc)})

    table = pd.DataFrame(rows)
    ok = table[table["status"] == "ok"] if "status" in table.columns else table
    verdict: Dict[str, Any] = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_years": [int(y) for y in test_years],
        "imputer": imp_report,
    }
    baselines = {"linear_regression", "ridge"}
    if not ok.empty:
        ml = ok[~ok["model_type"].isin(baselines)]
        base = ok[ok["model_type"].isin(baselines)]
        if not ml.empty and not base.empty:
            best_ml = ml.sort_values("rmse").iloc[0]
            best_base = base.sort_values("rmse").iloc[0]
            verdict.update(
                {
                    "best_ml": {"model_type": str(best_ml["model_type"]), "rmse": float(best_ml["rmse"])},
                    "best_baseline": {
                        "model_type": str(best_base["model_type"]),
                        "rmse": float(best_base["rmse"]),
                    },
                    "status": "supported"
                    if float(best_ml["rmse"]) < float(best_base["rmse"])
                    else "not_supported",
                    "relative_improvement": float(
                        (float(best_base["rmse"]) - float(best_ml["rmse"])) / float(best_base["rmse"])
                    ),
                }
            )
        else:
            verdict["status"] = "inconclusive"
    else:
        verdict["status"] = "inconclusive"
    return table, verdict


def _diagnostics_figure(
    root: Path,
    preimp: pd.DataFrame,
    point_panel: pd.DataFrame,
    mask: pd.DataFrame,
    shares: pd.DataFrame,
    validation: pd.DataFrame,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None

    top_vars = [
        v
        for v in shares.sort_values("share_imputed", ascending=False)["variable"].tolist()
        if v in mask.columns and mask[v].sum() > 0
    ][:2]

    n_panels = 2 + len(top_vars)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    plot_shares = shares[shares["share_imputed"] > 0]
    ax.bar(plot_shares["variable"], plot_shares["share_imputed"] * 100, color="#4C72B0")
    ax.set_title("Share of reconstructed cells")
    ax.set_ylabel("% of country-year cells")
    ax.set_xlabel("Indicator")

    for i, var in enumerate(top_vars, start=1):
        ax = axes[i]
        flags = mask[var].to_numpy(dtype=bool)
        observed = pd.to_numeric(preimp[var], errors="coerce").to_numpy(dtype=float)[~flags]
        imputed = pd.to_numeric(point_panel[var], errors="coerce").to_numpy(dtype=float)[flags]
        ax.hist(observed[~np.isnan(observed)], bins=12, alpha=0.65, label="observed", color="#55A868")
        ax.hist(imputed[~np.isnan(imputed)], bins=12, alpha=0.65, label="imputed", color="#C44E52")
        ax.set_title(f"{var}: observed vs imputed")
        ax.set_xlabel(var)
        ax.legend()

    ax = axes[-1]
    if not validation.empty:
        agg = (
            validation.groupby("configuration")["nrmse_sd"].mean().sort_values()
        )
        ax.barh(agg.index, agg.to_numpy(), color="#8172B2")
        ax.set_title("Masked-cell benchmark")
        ax.set_xlabel("Normalised RMSE (lower is better)")
    else:
        ax.axis("off")

    fig.tight_layout()
    path = ensure_dir(root / "outputs" / "figures") / "imputation_diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """Produce the imputation robustness package under ``results/imputation/``."""
    root = Path(project_root).resolve()
    results_dir = ensure_dir(root / "results" / "imputation")
    tables_dir = ensure_dir(root / "outputs" / "tables")
    registry = ResultRegistry(root)

    imp_cfg = _cfg(config, "imputation")
    out_cfg = imp_cfg.get("output") or {}
    preimp_path = root / out_cfg.get(
        "preimputation_path", "data/processed/panel_wide_preimputation.parquet"
    )
    mask_path = root / out_cfg.get("mask_path", "data/processed/imputation_mask.parquet")

    if not bool(imp_cfg.get("enabled", False)) or not preimp_path.exists() or not mask_path.exists():
        payload = {
            "status": "not_applicable",
            "reason": "imputation disabled or pre-imputation artifacts unavailable",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(results_dir / "imputation_robustness.json", payload)
        logger.info("Imputation robustness skipped: %s", payload["reason"])
        return payload

    research = _cfg(config, "research")
    index_cfg = _cfg(config, "index")
    eco_cfg = _cfg(config, "econometrics")
    ml_cfg = _cfg(config, "machine_learning")
    pre_cfg = _cfg(config, "preprocessing")

    seed = int(imp_cfg.get("random_seed", research.get("random_seed", 42)))
    lag = int(research.get("feature_lag", 1) or 1)
    independent = list(eco_cfg.get("independent") or [f"X{i}" for i in range(1, 11)])
    controls = list(eco_cfg.get("controls") or ["C1", "C2"])
    log_controls = list(((pre_cfg.get("transforms") or {}).get("log_controls")) or [])
    imp_kwargs = imputation_kwargs_from_config(config)
    imp_kwargs.pop("seed", None)
    donors = imp_kwargs.pop("donors", 5)

    preimp = read_df(preimp_path).reset_index(drop=True)
    mask = read_df(mask_path).reset_index(drop=True)
    point_panel = read_df(
        root / out_cfg.get("panel_path", "data/processed/panel_wide_imputed.parquet")
    ).reset_index(drop=True)

    summary: Dict[str, Any] = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "artifacts": {},
    }

    # --- 0. Cell provenance ---------------------------------------------------
    provenance_path = root / (
        (_cfg(config, "preprocessing").get("provenance") or {}).get(
            "path", "data/processed/cell_provenance.parquet"
        )
    )
    if provenance_path.exists():
        provenance = read_df(provenance_path)
        present = [c for c in DEFAULT_INDICATORS if c in provenance.columns]
        rows = []
        for var in present:
            counts = provenance[var].value_counts()
            total = int(counts.sum())
            rows.append(
                {
                    "variable": var,
                    "official": int(counts.get("official", 0)),
                    "carried_forward": int(counts.get("carried_forward", 0)),
                    "mice_imputed": int(counts.get("mice_imputed", 0)),
                    "missing": int(counts.get("missing", 0)),
                    "share_official": float(counts.get("official", 0) / total) if total else np.nan,
                }
            )
        prov_table = pd.DataFrame(rows).sort_values("share_official").reset_index(drop=True)
        prov_path = write_df(prov_table, results_dir / "cell_provenance_by_variable.csv")
        write_df(prov_table, tables_dir / "cell_provenance_by_variable.csv")
        registry.register("cell_provenance", prov_table, category="imputation")
        totals = prov_table[["official", "carried_forward", "mice_imputed", "missing"]].sum()
        grand = float(totals.sum())
        summary["provenance"] = {
            "by_state_share": {k: float(v / grand) for k, v in totals.items()} if grand else {},
            "n_cells": int(grand),
        }
        summary["artifacts"]["provenance"] = str(prov_path.relative_to(root))

    # --- 1. Reconstruction shares --------------------------------------------
    share_frames = _imputation_shares(mask, DEFAULT_INDICATORS)
    for name, frame in share_frames.items():
        path = write_df(frame, results_dir / f"imputation_share_{name}.csv")
        summary["artifacts"][f"share_{name}"] = str(path.relative_to(root))
    write_df(share_frames["by_variable"], tables_dir / "imputation_share_by_variable.csv")
    summary["most_reconstructed"] = (
        share_frames["by_variable"].head(3).to_dict("records")
    )

    # --- 2. Masked-cell benchmark --------------------------------------------
    val_cfg = imp_cfg.get("validation") or {}
    validation = pd.DataFrame()
    if bool(val_cfg.get("enabled", True)):
        logger.info("Running masked-cell imputation benchmark")
        validation = validate_imputation(
            preimp,
            DEFAULT_INDICATORS,
            mask_fraction=float(val_cfg.get("mask_fraction", 0.15)),
            n_repeats=int(val_cfg.get("n_repeats", 5)),
            seed=seed,
            base_kwargs=imp_kwargs,
        )
        if not validation.empty:
            path = write_df(validation, results_dir / "imputation_validation_raw.csv")
            agg = (
                validation.groupby(["configuration", "variable"])
                .agg(rmse=("rmse", "mean"), mae=("mae", "mean"), nrmse_sd=("nrmse_sd", "mean"))
                .reset_index()
            )
            agg_path = write_df(agg, results_dir / "imputation_validation.csv")
            write_df(agg, tables_dir / "imputation_validation.csv")
            overall = (
                validation.groupby("configuration")["nrmse_sd"].mean().sort_values()
            )
            summary["validation"] = {
                "overall_nrmse_by_configuration": {k: float(v) for k, v in overall.items()},
                "best_configuration": str(overall.index[0]),
                "mask_fraction": float(val_cfg.get("mask_fraction", 0.15)),
                "n_repeats": int(val_cfg.get("n_repeats", 5)),
            }
            summary["artifacts"]["validation_raw"] = str(path.relative_to(root))
            summary["artifacts"]["validation"] = str(agg_path.relative_to(root))
            registry.register("imputation_validation", agg, category="imputation")

    # --- 3. Multiple imputation + Rubin pooling ------------------------------
    mi_cfg = imp_cfg.get("multiple_imputation") or {}
    if bool(mi_cfg.get("enabled", True)):
        m = int(mi_cfg.get("n_imputations", 10))
        logger.info("Drawing %s multiple imputations for Rubin pooling", m)
        draws, mi_meta = multiple_imputation(
            preimp,
            DEFAULT_INDICATORS,
            n_imputations=m,
            seed=seed,
            donors=donors,
            **imp_kwargs,
        )
        per_draw: List[pd.DataFrame] = []
        for i, draw in enumerate(draws):
            try:
                scored, predictors = _prepare_draw(
                    draw,
                    index_cfg=index_cfg,
                    log_controls=log_controls,
                    independent=independent,
                    controls=controls,
                    lag=lag,
                    seed=seed,
                )
                coef = _fit_fe(scored, predictors)
                if coef is not None:
                    coef["draw"] = i
                    per_draw.append(coef)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("FE estimation failed on imputation draw %s: %s", i, exc)
        if len(per_draw) >= 2:
            pooled = _rubin_pool(per_draw)
            pooled_path = write_df(pooled, results_dir / "mi_pooled_fe.csv")
            write_df(pooled, tables_dir / "mi_pooled_fe.csv")
            write_df(
                pd.concat(per_draw, ignore_index=True),
                results_dir / "mi_per_draw_fe.csv",
            )
            registry.register("mi_pooled_fe", pooled, category="imputation")
            summary["multiple_imputation"] = {
                **mi_meta,
                "n_draws_estimated": len(per_draw),
                "mean_fraction_missing_information": float(
                    pooled["fraction_missing_information"].mean()
                ),
                "significant_terms": pooled.loc[
                    pooled["pvalue"] < 0.05, "term"
                ].tolist(),
            }
            summary["artifacts"]["mi_pooled_fe"] = str(pooled_path.relative_to(root))

    # --- 4. Complete-case (observed-only) fixed effects -----------------------
    try:
        scored_cc, preds_cc = _prepare_draw(
            preimp,
            index_cfg=index_cfg,
            log_controls=log_controls,
            independent=independent,
            controls=controls,
            lag=lag,
            seed=seed,
        )
        cc = _fit_fe(scored_cc, preds_cc)
        if cc is not None:
            cc_path = write_df(cc, results_dir / "complete_case_fe.csv")
            write_df(cc, tables_dir / "complete_case_fe.csv")
            registry.register("complete_case_fe", cc, category="imputation")
            summary["complete_case"] = {
                "nobs": int(cc["nobs"].iloc[0]) if "nobs" in cc.columns else None,
                "significant_terms": cc.loc[cc["pvalue"] < 0.05, "term"].tolist(),
            }
            summary["artifacts"]["complete_case_fe"] = str(cc_path.relative_to(root))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Complete-case FE failed: %s", exc)
        summary["complete_case"] = {"status": "failed", "note": repr(exc)}

    # --- 5. Leakage-safe predictive comparison -------------------------------
    ls_cfg = imp_cfg.get("leakage_safe_ml") or {}
    if bool(ls_cfg.get("enabled", True)):
        try:
            test_years = list(
                (ml_cfg.get("validation") or {}).get("test_years") or [2024, 2025]
            )
            model_types = list(
                ls_cfg.get("models")
                or ["random_forest", "xgboost", "catboost", "svr", "linear_regression", "ridge"]
            )
            logger.info("Running leakage-safe ML comparison on test years %s", test_years)
            table, verdict = _leakage_safe_ml(
                preimp,
                config=config,
                imp_kwargs=imp_kwargs,
                index_cfg=index_cfg,
                log_controls=log_controls,
                independent=independent,
                controls=controls,
                lag=lag,
                seed=seed,
                test_years=test_years,
                model_types=model_types,
            )
            ls_path = write_df(table, results_dir / "leakage_safe_ml.csv")
            write_df(table, tables_dir / "leakage_safe_ml.csv")
            write_json(results_dir / "leakage_safe_ml_verdict.json", verdict)
            registry.register("leakage_safe_ml", table, category="imputation")
            summary["leakage_safe_ml"] = verdict
            summary["artifacts"]["leakage_safe_ml"] = str(ls_path.relative_to(root))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Leakage-safe ML comparison failed: %s", exc)
            summary["leakage_safe_ml"] = {"status": "failed", "note": repr(exc)}

    # --- 6. Diagnostics figure ------------------------------------------------
    fig_path = _diagnostics_figure(
        root, preimp, point_panel, mask, share_frames["by_variable"], validation
    )
    if fig_path is not None:
        summary["artifacts"]["figure"] = str(fig_path.relative_to(root))

    payload_path = results_dir / "imputation_robustness.json"
    write_json(payload_path, summary)
    write_json(root / "outputs" / "audit" / "imputation_robustness.json", summary)
    registry.register(
        "imputation_robustness", summary, category="imputation", fmt="json"
    )
    logger.info("Imputation robustness package written to %s", payload_path)
    return summary
