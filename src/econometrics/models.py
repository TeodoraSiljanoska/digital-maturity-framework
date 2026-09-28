"""Panel fixed/random effects models, Hausman test, and diagnostics."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from common.errors import InsufficientSampleError, ModelNotApplicableError
from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger
from common.seeds import set_global_seed
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.econometrics")

try:
    from linearmodels.panel import PanelOLS, RandomEffects

    HAS_LINEARMODELS = True
except ImportError:  # pragma: no cover
    HAS_LINEARMODELS = False
    PanelOLS = RandomEffects = None  # type: ignore

try:
    from linearmodels.panel.compare import compare as lm_compare
except ImportError:  # pragma: no cover
    lm_compare = None  # type: ignore

try:
    from statsmodels.tsa.stattools import adfuller

    HAS_ADF = True
except ImportError:  # pragma: no cover
    HAS_ADF = False
    adfuller = None  # type: ignore


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "econometrics", "machine_learning"):
        if hasattr(config, key):
            out[key] = getattr(config, key) or {}
    return out


def _seed(config: Any) -> int:
    if hasattr(config, "random_seed"):
        try:
            return int(config.random_seed())
        except Exception:
            pass
    return int((_as_mapping(config).get("research") or {}).get("random_seed", 42))


def _load_panel_and_predictors(project_root: Path, config: Any):
    from statistics.descriptive import load_analysis_panel, predictor_columns

    mapping = _as_mapping(config)
    eco = mapping.get("econometrics") or {}
    research = mapping.get("research") or {}
    independent = list(eco.get("independent") or [f"X{i}" for i in range(1, 11)])
    controls = list(eco.get("controls") or ["C1", "C2"])
    use_lagged = bool(eco.get("use_lagged_features", True))
    lag = int(research.get("feature_lag") or 1)
    target = str(eco.get("dependent") or research.get("dependent_variable") or "DMI")
    entity_col = str(eco.get("entity_col") or "country_iso3")
    time_col = str(eco.get("time_col") or "year")

    df = load_analysis_panel(project_root)
    df, predictors = predictor_columns(
        df,
        independent=independent,
        controls=controls,
        use_lagged=use_lagged,
        lag=lag,
    )
    return df, predictors, target, entity_col, time_col, eco


def _prepare_estimation_frame(
    df: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    cols = [entity_col, time_col, target] + list(predictors)
    if "group_id" in df.columns:
        cols = cols + ["group_id"]
    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    work = df[cols].copy()
    required = [target] + [c for c in predictors if c in work.columns]
    work = work.dropna(subset=required)
    work[time_col] = work[time_col].astype(int)
    return work


def _formula(target: str, predictors: Sequence[str]) -> str:
    rhs = " + ".join(predictors) if predictors else "1"
    return f"{target} ~ {rhs}"


def _coef_table_from_params(
    params: pd.Series,
    pvalues: pd.Series,
    bse: Optional[pd.Series] = None,
    model_name: str = "model",
) -> pd.DataFrame:
    rows = []
    for name in params.index:
        rows.append(
            {
                "model": model_name,
                "term": str(name),
                "coefficient": float(params[name]),
                "std_error": float(bse[name]) if bse is not None and name in bse.index else np.nan,
                "pvalue": float(pvalues[name]) if name in pvalues.index else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fit_fe_linearmodels(
    panel: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    entity_col: str,
    time_col: str,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    data = panel.set_index([entity_col, time_col]).sort_index()
    y = data[target]
    x = sm.add_constant(data[list(predictors)])
    mod = PanelOLS(y, x, entity_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    coef = _coef_table_from_params(res.params, res.pvalues, res.std_errors, "FE")
    metrics = {
        "model": "FE",
        "backend": "linearmodels",
        "nobs": int(res.nobs),
        "rsquared": float(getattr(res, "rsquared", np.nan)),
        "rsquared_within": float(getattr(res, "rsquared_within", np.nan)),
        "f_statistic": float(getattr(getattr(res, "f_statistic", None), "stat", np.nan))
        if getattr(res, "f_statistic", None) is not None
        else np.nan,
    }
    return res, coef, metrics


def _fit_re_linearmodels(
    panel: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    entity_col: str,
    time_col: str,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    data = panel.set_index([entity_col, time_col]).sort_index()
    y = data[target]
    x = sm.add_constant(data[list(predictors)])
    mod = RandomEffects(y, x)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    coef = _coef_table_from_params(res.params, res.pvalues, res.std_errors, "RE")
    metrics = {
        "model": "RE",
        "backend": "linearmodels",
        "nobs": int(res.nobs),
        "rsquared": float(getattr(res, "rsquared", np.nan)),
        "rsquared_overall": float(getattr(res, "rsquared_overall", np.nan)),
    }
    return res, coef, metrics


def _fit_fe_within_ols(
    panel: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    entity_col: str,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """Within-transformed OLS fallback for fixed effects."""
    work = panel.copy()
    cols = [target] + list(predictors)
    demeaned = work.groupby(entity_col)[cols].transform(lambda s: s - s.mean())
    y = demeaned[target]
    x = demeaned[list(predictors)]
    x = sm.add_constant(x, has_constant="add")
    # Drop constant if absorbed (all zeros after demeaning)
    if "const" in x.columns and np.allclose(x["const"], 0):
        x = x.drop(columns=["const"])
    res = sm.OLS(y, x, missing="drop").fit(cov_type="HC1")
    coef = _coef_table_from_params(res.params, res.pvalues, res.bse, "FE")
    metrics = {
        "model": "FE",
        "backend": "within_ols_fallback",
        "nobs": int(res.nobs),
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
        "fvalue": float(res.fvalue) if res.fvalue is not None else np.nan,
    }
    return res, coef, metrics


def _fit_pooled_ols(
    panel: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    model_name: str = "RE_fallback",
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    y = panel[target]
    x = sm.add_constant(panel[list(predictors)], has_constant="add")
    res = sm.OLS(y, x, missing="drop").fit(cov_type="HC1")
    coef = _coef_table_from_params(res.params, res.pvalues, res.bse, model_name)
    metrics = {
        "model": model_name,
        "backend": "statsmodels_pooled_ols",
        "nobs": int(res.nobs),
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
        "fvalue": float(res.fvalue) if res.fvalue is not None else np.nan,
        "aic": float(res.aic),
        "bic": float(res.bic),
    }
    return res, coef, metrics


def _mundlak_test(
    panel: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    entity_col: str,
) -> Dict[str, Any]:
    """
    Regression-based (Mundlak/Wooldridge) alternative to the classical Hausman test.

    Augments the pooled model with entity means of the regressors and tests
    their joint significance. Unlike the difference-in-covariance form it stays
    valid with clustered errors and cannot return a negative statistic, which
    matters in short panels where the classical statistic is often undefined.
    """
    out: Dict[str, Any] = {"method": "mundlak_auxiliary_regression"}
    try:
        cols = [target] + [c for c in predictors if c in panel.columns]
        work = panel[[entity_col] + cols].dropna().copy()
        mean_cols: List[str] = []
        for c in predictors:
            if c not in work.columns:
                continue
            name = f"{c}__entity_mean"
            work[name] = work.groupby(entity_col)[c].transform("mean")
            mean_cols.append(name)
        if not mean_cols or len(work) <= len(mean_cols) * 2:
            return {**out, "status": "skipped", "note": "insufficient rows for auxiliary regression"}
        exog = sm.add_constant(work[[c for c in predictors if c in work.columns] + mean_cols])
        res = sm.OLS(work[target], exog).fit(
            cov_type="cluster", cov_kwds={"groups": work[entity_col]}
        )
        restriction = np.zeros((len(mean_cols), exog.shape[1]))
        for i, name in enumerate(mean_cols):
            restriction[i, list(exog.columns).index(name)] = 1.0
        wald = res.f_test(restriction)
        pval = float(np.squeeze(wald.pvalue))
        return {
            **out,
            "status": "ok",
            "statistic": float(np.squeeze(wald.fvalue)),
            "pvalue": pval,
            "df_num": int(len(mean_cols)),
            "prefer": "FE" if pval < 0.05 else "RE",
            "note": (
                "Joint significance of entity means rejects the random-effects "
                "orthogonality assumption and favours fixed effects."
            ),
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {**out, "status": "failed", "note": f"mundlak_failed: {exc}"}


def _hausman_test(fe_res: Any, re_res: Any) -> Dict[str, Any]:
    """Hausman test when linearmodels results available; else approximate comparison."""
    out: Dict[str, Any] = {"method": None, "statistic": None, "pvalue": None, "note": None}
    if HAS_LINEARMODELS and fe_res is not None and re_res is not None:
        try:
            # Prefer compare-based chi2 if attributes align
            b_fe = fe_res.params
            b_re = re_res.params
            common = [i for i in b_fe.index if i in b_re.index and i != "const"]
            if not common:
                common = [i for i in b_fe.index if i in b_re.index]
            if len(common) >= 1:
                diff = b_fe[common] - b_re[common]
                cov_fe = fe_res.cov.loc[common, common]
                cov_re = re_res.cov.loc[common, common]
                cov_diff = cov_fe - cov_re
                # Regularize if not PD
                try:
                    inv = np.linalg.inv(cov_diff.values)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(cov_diff.values)
                stat = float(diff.values.T @ inv @ diff.values)
                df = len(common)
                from scipy.stats import chi2

                min_eig = float(np.min(np.linalg.eigvalsh(cov_diff.values)))
                # A negative statistic (equivalently a non-PSD covariance
                # difference) means the asymptotic assumptions behind the
                # classical form fail in this sample. Reporting the resulting
                # p-value of 1.0 as evidence "for RE" would be a misreading.
                if stat < 0 or min_eig < 0:
                    out.update(
                        {
                            "method": "hausman_fe_vs_re",
                            "statistic": stat,
                            "pvalue": None,
                            "df": df,
                            "status": "inconclusive",
                            "min_eigenvalue_cov_diff": min_eig,
                            "prefer": None,
                            "note": (
                                "Negative Hausman statistic / non-PSD covariance difference: "
                                "the classical test is uninformative in this short panel. "
                                "Model choice is decided by the Mundlak auxiliary regression."
                            ),
                        }
                    )
                    return out
                pval = float(chi2.sf(stat, df))
                out.update(
                    {
                        "method": "hausman_fe_vs_re",
                        "statistic": stat,
                        "pvalue": pval,
                        "df": df,
                        "status": "ok",
                        "min_eigenvalue_cov_diff": min_eig,
                        "prefer": "FE" if pval < 0.05 else "RE",
                    }
                )
                return out
        except Exception as exc:
            out["note"] = f"hausman_failed: {exc}"

    # Approximate: compare coefficient vectors / R2
    try:
        fe_params = getattr(fe_res, "params", None)
        re_params = getattr(re_res, "params", None)
        if fe_params is not None and re_params is not None:
            common = [i for i in fe_params.index if i in re_params.index]
            if common:
                delta = (fe_params[common] - re_params[common]).abs().mean()
                out.update(
                    {
                        "method": "approximate_mean_abs_coef_diff",
                        "statistic": float(delta),
                        "pvalue": None,
                        "prefer": "FE" if delta > 0.05 else "RE",
                        "note": "linearmodels Hausman unavailable; used mean |Δβ|",
                    }
                )
                return out
    except Exception as exc:
        out["note"] = f"approximate_hausman_failed: {exc}"
    out["note"] = out.get("note") or "Hausman not computed"
    return out


def _breusch_pagan(residuals: np.ndarray, exog: pd.DataFrame) -> Dict[str, Any]:
    try:
        x = sm.add_constant(exog, has_constant="add")
        lm, lm_pvalue, fval, f_pvalue = het_breuschpagan(residuals, x)
        return {
            "test": "breusch_pagan",
            "lm_stat": float(lm),
            "lm_pvalue": float(lm_pvalue),
            "f_stat": float(fval),
            "f_pvalue": float(f_pvalue),
        }
    except Exception as exc:
        var = float(np.var(residuals)) if len(residuals) else np.nan
        return {
            "test": "residual_variance_fallback",
            "residual_variance": var,
            "note": repr(exc),
        }


def _adf_by_country_average(
    df: pd.DataFrame,
    target: str,
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    rows = []
    if not HAS_ADF:
        return pd.DataFrame(
            [{"country_iso3": None, "note": "adfuller_unavailable", "status": "skipped"}]
        )
    for entity, g in df.groupby(entity_col):
        series = g.sort_values(time_col)[target].dropna()
        if len(series) < 4:
            rows.append(
                {
                    "country_iso3": entity,
                    "n": int(len(series)),
                    "adf_stat": np.nan,
                    "pvalue": np.nan,
                    "status": "MODEL_NOT_APPLICABLE",
                    "note": "series_too_short",
                }
            )
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, pval, usedlag, nobs, crit, icbest = adfuller(series.values, autolag="AIC")
            rows.append(
                {
                    "country_iso3": entity,
                    "n": int(nobs),
                    "adf_stat": float(stat),
                    "pvalue": float(pval),
                    "usedlag": int(usedlag),
                    "status": "ok",
                    "note": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "country_iso3": entity,
                    "n": int(len(series)),
                    "adf_stat": np.nan,
                    "pvalue": np.nan,
                    "status": "MODEL_NOT_APPLICABLE",
                    "note": repr(exc),
                }
            )
    return pd.DataFrame(rows)


def _copy_key(src: Path, tables_dir: Path) -> None:
    ensure_dir(tables_dir)
    shutil.copy2(src, tables_dir / src.name)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Estimate panel FE/RE (or fallbacks), Hausman, and diagnostics.

    Writes coefficients/metrics under ``results/econometrics/`` and ``outputs/tables/``.
    """
    root = Path(project_root).resolve()
    set_global_seed(_seed(config))
    registry = ResultRegistry(root)
    results_dir = ensure_dir(root / "results" / "econometrics")
    tables_dir = ensure_dir(root / "outputs" / "tables")

    df, predictors, target, entity_col, time_col, eco = _load_panel_and_predictors(
        root, config
    )
    min_obs = int(eco.get("min_observations") or 30)
    group_min_obs = int(eco.get("group_min_observations") or min_obs)
    group_stratified = bool(eco.get("group_stratified", True))
    diagnostics_cfg = eco.get("diagnostics") or {}

    panel = _prepare_estimation_frame(df, target, predictors, entity_col, time_col)
    predictors = [c for c in predictors if c in panel.columns]

    if len(panel) < min_obs:
        err = InsufficientSampleError(
            f"Pooled estimation requires >= {min_obs} observations; got {len(panel)}",
            details={"nobs": len(panel), "min_observations": min_obs},
        )
        write_json(
            results_dir / "insufficient_sample.json",
            err.to_dict(),
        )
        registry.register(
            "econometrics_insufficient_sample",
            err.to_dict(),
            category="econometrics",
            fmt="json",
        )
        raise err

    formula = _formula(target, predictors)
    write_json(
        results_dir / "model_spec.json",
        {
            "formula": formula,
            "target": target,
            "predictors": predictors,
            "entity_col": entity_col,
            "time_col": time_col,
            "nobs": int(len(panel)),
            "has_linearmodels": HAS_LINEARMODELS,
        },
    )

    coef_frames: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    applicability: List[Dict[str, Any]] = []
    fe_res = re_res = None
    fe_backend = re_backend = None

    # --- Pooled FE ---
    try:
        if HAS_LINEARMODELS:
            fe_res, fe_coef, fe_metrics = _fit_fe_linearmodels(
                panel, target, predictors, entity_col, time_col
            )
            fe_backend = "linearmodels"
        else:
            fe_res, fe_coef, fe_metrics = _fit_fe_within_ols(
                panel, target, predictors, entity_col
            )
            fe_backend = "within_ols_fallback"
        coef_frames.append(fe_coef)
        metric_rows.append(fe_metrics)
        applicability.append({"model": "FE", "status": "ok", "backend": fe_backend})
    except Exception as exc:
        logger.exception("FE estimation failed")
        applicability.append(
            {
                "model": "FE",
                "status": "MODEL_NOT_APPLICABLE",
                "code": ModelNotApplicableError.code,
                "note": repr(exc),
            }
        )

    # --- Pooled RE ---
    try:
        if HAS_LINEARMODELS:
            re_res, re_coef, re_metrics = _fit_re_linearmodels(
                panel, target, predictors, entity_col, time_col
            )
            re_backend = "linearmodels"
        else:
            re_res, re_coef, re_metrics = _fit_pooled_ols(
                panel, target, predictors, model_name="RE_fallback"
            )
            re_backend = "RE_fallback"
        coef_frames.append(re_coef)
        metric_rows.append(re_metrics)
        applicability.append({"model": "RE", "status": "ok", "backend": re_backend})
    except Exception as exc:
        logger.exception("RE estimation failed")
        applicability.append(
            {
                "model": "RE",
                "status": "MODEL_NOT_APPLICABLE",
                "code": ModelNotApplicableError.code,
                "note": repr(exc),
            }
        )

    # --- Hausman (+ Mundlak fallback for short panels) ---
    hausman: Dict[str, Any] = {"note": "skipped"}
    if diagnostics_cfg.get("hausman", True) and fe_res is not None and re_res is not None:
        hausman = _hausman_test(fe_res, re_res)
        mundlak = _mundlak_test(panel, target, predictors, entity_col)
        hausman["mundlak"] = mundlak
        if hausman.get("prefer") is None and mundlak.get("status") == "ok":
            hausman["prefer"] = mundlak.get("prefer")
            hausman["preference_source"] = "mundlak_auxiliary_regression"
        else:
            hausman["preference_source"] = "classical_hausman"
    hausman_path = results_dir / "hausman_test.json"
    write_json(hausman_path, hausman)
    registry.register(
        "hausman_test_result",
        hausman,
        category="econometrics",
        fmt="json",
        meta={"path": str(hausman_path.relative_to(root))},
    )

    # --- Diagnostics from FE residuals when possible ---
    diag: Dict[str, Any] = {}
    resid = None
    exog_for_bp = panel[list(predictors)]
    if fe_res is not None:
        try:
            resid = np.asarray(fe_res.resids if hasattr(fe_res, "resids") else fe_res.resid)
        except Exception:
            resid = None
    if resid is None and re_res is not None:
        try:
            resid = np.asarray(re_res.resids if hasattr(re_res, "resids") else re_res.resid)
        except Exception:
            resid = None

    if diagnostics_cfg.get("heteroskedasticity", True) and resid is not None:
        # Align exog length
        if len(resid) == len(exog_for_bp):
            diag["heteroskedasticity"] = _breusch_pagan(resid, exog_for_bp)
        else:
            diag["heteroskedasticity"] = {
                "test": "residual_variance_fallback",
                "residual_variance": float(np.var(resid)),
                "note": "exog/resid length mismatch",
            }

    if diagnostics_cfg.get("autocorrelation", True) and resid is not None and len(resid) > 2:
        try:
            diag["durbin_watson"] = {"statistic": float(durbin_watson(resid))}
        except Exception as exc:
            diag["durbin_watson"] = {"note": repr(exc)}

    adf_df = pd.DataFrame()
    if diagnostics_cfg.get("stationarity", True):
        adf_df = _adf_by_country_average(panel, target, entity_col, time_col)
        adf_path = results_dir / "adf_dmi_by_country.csv"
        write_df(adf_df, adf_path)
        _copy_key(adf_path, tables_dir)
        registry.register("adf_dmi_by_country", adf_df, category="econometrics")
        diag["stationarity_summary"] = {
            "n_countries": int(adf_df.shape[0]),
            "n_ok": int((adf_df.get("status") == "ok").sum()) if not adf_df.empty else 0,
        }

    diag_path = results_dir / "diagnostics.json"
    write_json(diag_path, diag)
    registry.register(
        "econometrics_diagnostics_result",
        diag,
        category="econometrics",
        fmt="json",
        meta={"path": str(diag_path.relative_to(root))},
    )

    # --- Group-stratified FE ---
    group_coef_frames: List[pd.DataFrame] = []
    if group_stratified and "group_id" in panel.columns:
        for gid, gdf in panel.groupby("group_id", dropna=False):
            n = len(gdf)
            if n < group_min_obs:
                applicability.append(
                    {
                        "model": "FE_group",
                        "group_id": gid,
                        "status": "MODEL_NOT_APPLICABLE",
                        "code": ModelNotApplicableError.code,
                        "nobs": n,
                        "min_observations": group_min_obs,
                        "note": "insufficient observations for group-stratified FE",
                    }
                )
                logger.warning(
                    "Skipping group FE for %s (n=%s < min=%s)",
                    gid,
                    n,
                    group_min_obs,
                )
                continue
            # Need variation across entities within group
            if gdf[entity_col].nunique() < 2:
                applicability.append(
                    {
                        "model": "FE_group",
                        "group_id": gid,
                        "status": "MODEL_NOT_APPLICABLE",
                        "code": ModelNotApplicableError.code,
                        "nobs": n,
                        "note": "fewer than 2 entities in group",
                    }
                )
                continue
            try:
                if HAS_LINEARMODELS:
                    _, gcoef, gmetrics = _fit_fe_linearmodels(
                        gdf, target, predictors, entity_col, time_col
                    )
                else:
                    _, gcoef, gmetrics = _fit_fe_within_ols(
                        gdf, target, predictors, entity_col
                    )
                gcoef = gcoef.copy()
                gcoef["group_id"] = gid
                gcoef["model"] = f"FE_group:{gid}"
                group_coef_frames.append(gcoef)
                gmetrics = dict(gmetrics)
                gmetrics["model"] = f"FE_group:{gid}"
                gmetrics["group_id"] = gid
                metric_rows.append(gmetrics)
                applicability.append(
                    {
                        "model": "FE_group",
                        "group_id": gid,
                        "status": "ok",
                        "nobs": n,
                    }
                )
            except Exception as exc:
                applicability.append(
                    {
                        "model": "FE_group",
                        "group_id": gid,
                        "status": "MODEL_NOT_APPLICABLE",
                        "code": ModelNotApplicableError.code,
                        "nobs": n,
                        "note": repr(exc),
                    }
                )
                logger.warning("Group FE failed for %s: %s", gid, exc)

    # --- Persist coefficients / metrics ---
    all_coef = pd.concat(coef_frames + group_coef_frames, ignore_index=True) if (
        coef_frames or group_coef_frames
    ) else pd.DataFrame()
    coef_path = results_dir / "coefficients.csv"
    write_df(all_coef, coef_path)
    _copy_key(coef_path, tables_dir)
    registry.register("econometrics_coefficients", all_coef, category="econometrics")

    # Wide p-value table for reporting
    if not all_coef.empty:
        pval_wide = all_coef.pivot_table(
            index="term", columns="model", values="pvalue", aggfunc="first"
        ).reset_index()
        pval_path = results_dir / "pvalues.csv"
        write_df(pval_wide, pval_path)
        _copy_key(pval_path, tables_dir)
        registry.register("econometrics_pvalues", pval_wide, category="econometrics")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = results_dir / "model_metrics.csv"
    write_df(metrics_df, metrics_path)
    _copy_key(metrics_path, tables_dir)
    registry.register("econometrics_metrics", metrics_df, category="econometrics")

    appl_df = pd.DataFrame(applicability)
    appl_path = results_dir / "model_applicability.csv"
    write_df(appl_df, appl_path)
    registry.register("econometrics_applicability", appl_df, category="econometrics")

    summary = {
        "formula": formula,
        "nobs": int(len(panel)),
        "has_linearmodels": HAS_LINEARMODELS,
        "hausman": hausman,
        "n_models_ok": int((appl_df.get("status") == "ok").sum()) if not appl_df.empty else 0,
        "artifacts": {
            "coefficients": str(coef_path.relative_to(root)),
            "metrics": str(metrics_path.relative_to(root)),
            "diagnostics": str(diag_path.relative_to(root)),
            "hausman": str(hausman_path.relative_to(root)),
        },
    }
    summary_path = results_dir / "econometrics_summary.json"
    write_json(summary_path, summary)
    registry.register(
        "econometrics_summary_result",
        summary,
        category="econometrics",
        fmt="json",
        meta={"path": str(summary_path.relative_to(root))},
    )
    logger.info("Econometrics complete (nobs=%s, linearmodels=%s)", len(panel), HAS_LINEARMODELS)
    return summary
