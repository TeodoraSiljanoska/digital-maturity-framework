"""Sigma- and beta-convergence analysis for the Digital Maturity Index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common.errors import FrameworkError, InsufficientSampleError
from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger
from common.seeds import set_global_seed
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.convergence")


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "econometrics", "convergence"):
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


def _load_panel(project_root: Path, config: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from statistics.descriptive import load_analysis_panel

    mapping = _as_mapping(config)
    conv = dict(mapping.get("convergence") or {})
    eco = mapping.get("econometrics") or {}
    research = mapping.get("research") or {}

    dependent = str(
        conv.get("dependent")
        or eco.get("dependent")
        or research.get("dependent_variable")
        or "DMI"
    )
    entity_col = str(eco.get("entity_col") or "country_iso3")
    time_col = str(eco.get("time_col") or "year")
    group_col = "group_id"
    controls = list(conv.get("conditional_controls") or ["C1", "C2"])

    df = load_analysis_panel(project_root)
    if dependent not in df.columns:
        raise KeyError(f"Dependent variable '{dependent}' missing from analysis panel")
    if entity_col not in df.columns or time_col not in df.columns:
        raise KeyError(f"Panel must contain {entity_col} and {time_col}")

    work = df.copy()
    work[time_col] = work[time_col].astype(int)
    work = work.dropna(subset=[dependent, entity_col, time_col])
    return work, {
        "dependent": dependent,
        "entity_col": entity_col,
        "time_col": time_col,
        "group_col": group_col if group_col in work.columns else None,
        "controls": [c for c in controls if c in work.columns],
        "config": conv,
    }


def sigma_convergence(
    df: pd.DataFrame,
    *,
    dependent: str,
    time_col: str = "year",
    group_col: Optional[str] = None,
    also_cv: bool = True,
) -> pd.DataFrame:
    """
    Cross-sectional dispersion of ``dependent`` by year.

    Returns std (and optionally CV = std/mean) for the full sample and by group.
    """
    rows: List[Dict[str, Any]] = []

    def _stats(sub: pd.DataFrame, sample: str, year: int) -> None:
        s = sub[dependent].astype(float).dropna()
        if s.empty:
            return
        mean = float(s.mean())
        std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        row = {
            "sample": sample,
            "year": int(year),
            "n": int(len(s)),
            "mean": mean,
            "std": std,
        }
        if also_cv:
            row["cv"] = float(std / mean) if abs(mean) > 1e-12 else np.nan
        rows.append(row)

    for year, g in df.groupby(time_col, sort=True):
        _stats(g, "full", int(year))

    if group_col and group_col in df.columns:
        for (grp, year), g in df.groupby([group_col, time_col], sort=True):
            _stats(g, str(grp), int(year))

    return pd.DataFrame(rows)


def _country_endpoints(
    df: pd.DataFrame,
    *,
    dependent: str,
    entity_col: str,
    time_col: str,
    controls: Sequence[str],
    control_mode: str = "initial",
) -> pd.DataFrame:
    """
    Build one row per country with initial/final DMI and optional controls.

    control_mode:
      - ``initial``: control values at the initial year
      - ``mean``: country-level mean of controls over the sample
    """
    rows = []
    for entity, g in df.groupby(entity_col, sort=False):
        g = g.sort_values(time_col)
        g = g.dropna(subset=[dependent])
        if len(g) < 2:
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        d0 = float(first[dependent])
        dT = float(last[dependent])
        if d0 <= 0 or dT <= 0:
            continue
        row: Dict[str, Any] = {
            entity_col: entity,
            "year_0": int(first[time_col]),
            "year_T": int(last[time_col]),
            "DMI_0": d0,
            "DMI_T": dT,
            "log_DMI_0": float(np.log(d0)),
            "growth": float(np.log(dT) - np.log(d0)),
            "n_obs": int(len(g)),
        }
        if "group_id" in g.columns:
            row["group_id"] = first["group_id"]
        for c in controls:
            if c not in g.columns:
                continue
            if control_mode == "mean":
                row[c] = float(g[c].astype(float).mean())
            else:
                val = first[c]
                row[c] = float(val) if pd.notna(val) else float(g[c].astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _ols_beta(frame: pd.DataFrame, regressors: Sequence[str]) -> Dict[str, Any]:
    """OLS of growth on log initial DMI (+ optional controls)."""
    cols = ["growth"] + list(regressors)
    data = frame.dropna(subset=cols).copy()
    n = len(data)
    if n < max(3, len(regressors) + 2):
        raise InsufficientSampleError(
            f"Beta-convergence sample too small: n={n}, regressors={list(regressors)}",
            details={"n": n, "regressors": list(regressors)},
        )
    y = data["growth"].astype(float)
    X = sm.add_constant(data[list(regressors)].astype(float), has_constant="add")
    res = sm.OLS(y, X).fit()
    params = {str(k): float(v) for k, v in res.params.items()}
    pvalues = {str(k): float(v) for k, v in res.pvalues.items()}
    bse = {str(k): float(v) for k, v in res.bse.items()}
    beta = params.get("log_DMI_0", np.nan)
    return {
        "n": int(n),
        "beta": float(beta),
        "beta_pvalue": float(pvalues.get("log_DMI_0", np.nan)),
        "beta_std_error": float(bse.get("log_DMI_0", np.nan)),
        "supports_convergence": bool(beta < 0) if np.isfinite(beta) else False,
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "params": params,
        "pvalues": pvalues,
        "std_errors": bse,
        "regressors": list(regressors),
    }


def beta_convergence(
    df: pd.DataFrame,
    *,
    dependent: str,
    entity_col: str,
    time_col: str,
    controls: Optional[Sequence[str]] = None,
    conditional: bool = False,
    control_mode: str = "initial",
    group_col: Optional[str] = None,
    by_group: bool = False,
    min_countries: int = 3,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """
    Absolute (and optional conditional) beta-convergence regressions.

    Returns (endpoints_df, result_dicts, coefficients_df).
    """
    controls = list(controls or [])
    endpoints = _country_endpoints(
        df,
        dependent=dependent,
        entity_col=entity_col,
        time_col=time_col,
        controls=controls,
        control_mode=control_mode,
    )
    results: List[Dict[str, Any]] = []
    coef_rows: List[Dict[str, Any]] = []

    def _run_one(sub: pd.DataFrame, sample: str, conditional_flag: bool) -> None:
        regs = ["log_DMI_0"]
        if conditional_flag:
            regs = regs + [c for c in controls if c in sub.columns]
        model_name = "conditional_beta" if conditional_flag else "beta"
        try:
            out = _ols_beta(sub, regs)
        except InsufficientSampleError as exc:
            results.append(
                {
                    "model": model_name,
                    "sample": sample,
                    "status": "skipped",
                    "warning": exc.message,
                    "n": int(len(sub)),
                }
            )
            return
        results.append(
            {
                "model": model_name,
                "sample": sample,
                "status": "ok",
                **{k: v for k, v in out.items() if k not in {"params", "pvalues", "std_errors"}},
                "params": out["params"],
                "pvalues": out["pvalues"],
                "std_errors": out["std_errors"],
            }
        )
        for term, coef in out["params"].items():
            coef_rows.append(
                {
                    "model": model_name,
                    "sample": sample,
                    "term": term,
                    "coefficient": coef,
                    "std_error": out["std_errors"].get(term, np.nan),
                    "pvalue": out["pvalues"].get(term, np.nan),
                }
            )

    if len(endpoints) >= min_countries:
        _run_one(endpoints, "full", conditional_flag=False)
        if conditional and controls:
            _run_one(endpoints, "full", conditional_flag=True)
    else:
        results.append(
            {
                "model": "beta",
                "sample": "full",
                "status": "skipped",
                "warning": f"Fewer than min_countries={min_countries}",
                "n": int(len(endpoints)),
            }
        )

    if by_group and group_col and group_col in endpoints.columns:
        for grp, sub in endpoints.groupby(group_col, sort=True):
            if len(sub) < min_countries:
                results.append(
                    {
                        "model": "beta",
                        "sample": str(grp),
                        "status": "skipped",
                        "warning": f"n_countries={len(sub)} < min_countries={min_countries}",
                        "n": int(len(sub)),
                    }
                )
                continue
            _run_one(sub, str(grp), conditional_flag=False)
            if conditional and controls:
                _run_one(sub, str(grp), conditional_flag=True)

    return endpoints, results, pd.DataFrame(coef_rows)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Estimate sigma- and beta-convergence of DMI (full sample and by group).

    Writes CSV/JSON under ``results/convergence/`` and ``outputs/convergence/``.
    """
    root = Path(project_root).resolve()
    mapping = _as_mapping(config)
    seed = _seed(config)
    set_global_seed(seed)

    df, meta = _load_panel(root, config)
    conv = meta["config"]
    dependent = meta["dependent"]
    entity_col = meta["entity_col"]
    time_col = meta["time_col"]
    group_col = meta["group_col"]
    controls = meta["controls"]

    methods = list(conv.get("methods") or ["sigma", "beta", "conditional_beta"])
    sigma_cfg = dict(conv.get("sigma") or {})
    also_cv = bool(sigma_cfg.get("also_cv", True))
    by_group = bool(conv.get("by_group", True))
    min_countries = int(conv.get("min_countries") or 3)
    # Prefer initial control values; allow override via config
    control_mode = str(conv.get("control_mode") or "initial")
    if control_mode not in {"initial", "mean"}:
        control_mode = "initial"

    results_dir = ensure_dir(root / "results" / "convergence")
    out_dir = ensure_dir(root / "outputs" / "convergence")
    registry = ResultRegistry(root)

    artifacts: Dict[str, Any] = {}
    warnings_log: List[Dict[str, Any]] = []
    produced = 0
    summary: Dict[str, Any] = {
        "dependent": dependent,
        "methods": methods,
        "by_group": by_group,
        "min_countries": min_countries,
        "control_mode": control_mode,
        "controls": controls,
        "seed": seed,
    }

    # --- Sigma ---
    if "sigma" in methods:
        try:
            sigma_df = sigma_convergence(
                df,
                dependent=dependent,
                time_col=time_col,
                group_col=group_col if by_group else None,
                also_cv=also_cv,
            )
            for directory in (results_dir, out_dir):
                write_df(sigma_df, directory / "sigma_convergence.csv")
            registry.register(
                "sigma_convergence",
                sigma_df,
                category="convergence",
                meta={"also_cv": also_cv, "by_group": by_group},
            )
            artifacts["sigma"] = "results/convergence/sigma_convergence.csv"
            summary["sigma_n_rows"] = int(len(sigma_df))
            # Compact year trend for full sample
            full = sigma_df[sigma_df["sample"] == "full"].sort_values("year")
            summary["sigma_trend"] = {
                "first_year_std": float(full["std"].iloc[0]) if len(full) else None,
                "last_year_std": float(full["std"].iloc[-1]) if len(full) else None,
                "declining": bool(full["std"].iloc[-1] < full["std"].iloc[0])
                if len(full) >= 2
                else None,
            }
            produced += 1
        except Exception as exc:
            logger.exception("Sigma-convergence failed")
            warnings_log.append({"method": "sigma", "warning": repr(exc)})

    # --- Beta / conditional beta ---
    want_beta = "beta" in methods or "conditional_beta" in methods
    if want_beta:
        try:
            endpoints, beta_results, coef_df = beta_convergence(
                df,
                dependent=dependent,
                entity_col=entity_col,
                time_col=time_col,
                controls=controls,
                conditional="conditional_beta" in methods,
                control_mode=control_mode,
                group_col=group_col,
                by_group=by_group,
                min_countries=min_countries,
            )
            for directory in (results_dir, out_dir):
                write_df(endpoints, directory / "beta_endpoints.csv")
                write_json(directory / "beta_convergence.json", beta_results)
                if not coef_df.empty:
                    write_df(coef_df, directory / "beta_coefficients.csv")

            registry.register(
                "beta_endpoints",
                endpoints,
                category="convergence",
            )
            registry.register(
                "beta_convergence",
                beta_results,
                category="convergence",
                fmt="json",
            )
            if not coef_df.empty:
                registry.register(
                    "beta_coefficients",
                    coef_df,
                    category="convergence",
                )

            artifacts["beta"] = {
                "endpoints": "results/convergence/beta_endpoints.csv",
                "results": "results/convergence/beta_convergence.json",
                "coefficients": "results/convergence/beta_coefficients.csv"
                if not coef_df.empty
                else None,
            }
            ok_betas = [r for r in beta_results if r.get("status") == "ok"]
            summary["beta_n_ok"] = len(ok_betas)
            summary["beta_full"] = next(
                (r for r in ok_betas if r.get("sample") == "full" and r.get("model") == "beta"),
                None,
            )
            summary["conditional_beta_full"] = next(
                (
                    r
                    for r in ok_betas
                    if r.get("sample") == "full" and r.get("model") == "conditional_beta"
                ),
                None,
            )
            for r in beta_results:
                if r.get("status") == "skipped":
                    warnings_log.append(
                        {
                            "method": r.get("model"),
                            "sample": r.get("sample"),
                            "warning": r.get("warning"),
                        }
                    )
            if ok_betas:
                produced += 1
            else:
                warnings_log.append(
                    {"method": "beta", "warning": "No successful beta regressions"}
                )
        except Exception as exc:
            logger.exception("Beta-convergence failed")
            warnings_log.append({"method": "beta", "warning": repr(exc)})

    summary["artifacts"] = artifacts
    summary["warnings"] = warnings_log
    summary["n_produced"] = produced
    summary["n_countries"] = int(df[entity_col].nunique())
    summary["year_min"] = int(df[time_col].min())
    summary["year_max"] = int(df[time_col].max())

    for directory in (results_dir, out_dir):
        write_json(directory / "convergence_summary.json", summary)
    registry.register("convergence_summary", summary, category="convergence", fmt="json")

    if warnings_log:
        for directory in (results_dir, out_dir):
            write_json(directory / "convergence_warnings.json", warnings_log)

    if produced == 0:
        err = {
            "error": "No convergence artifacts produced",
            "warnings": warnings_log,
        }
        write_json(results_dir / "convergence_error_report.json", err)
        write_json(out_dir / "convergence_error_report.json", err)
        raise FrameworkError(
            "Convergence analysis produced no results",
            details=err,
        )

    logger.info(
        "Convergence complete (%s artifacts, %s warnings)",
        produced,
        len(warnings_log),
    )
    return summary
