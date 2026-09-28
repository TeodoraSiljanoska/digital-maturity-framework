"""Evaluate research hypotheses H1 / H1.1–H1.8 from actual pipeline results."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from common.io import ensure_dir, read_df, read_json, write_json
from common.logging_utils import get_logger
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.hypotheses")

ALPHA = 0.05
HYPOTHESIZED_DRIVERS = {"X1", "X2", "X3", "X4", "X5", "X6", "C1", "C2"}


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "econometrics", "machine_learning", "xai"):
        if hasattr(config, key):
            out[key] = getattr(config, key) or {}
    return out


def _base_var(term: str) -> str:
    """Strip lag suffixes / interactions noise: X1_lag1 -> X1."""
    t = str(term)
    t = re.sub(r"_lag\d+$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^L\d+\.", "", t)
    return t


def _load_hypotheses(config: Any) -> List[Dict[str, Any]]:
    research = _as_mapping(config).get("research") or {}
    hyps = list(research.get("hypotheses") or [])
    if hyps:
        return hyps
    # Minimal defaults matching research.yaml ids
    return [
        {"id": "H1", "text": "Aggregate digital maturity drivers and AI prediction advantage."},
        {"id": "H1.1", "text": "Digital infrastructure positive effect.", "variables": ["X1", "X2"]},
        {"id": "H1.2", "text": "E-government positive effect.", "variables": ["X3"]},
        {"id": "H1.3", "text": "Digital skills important.", "variables": ["X4"]},
        {"id": "H1.4", "text": "GDP and education positive.", "variables": ["C1", "C2"]},
        {"id": "H1.5", "text": "Group differences in DMI."},
        {"id": "H1.6", "text": "ML more accurate than classical baselines."},
        {"id": "H1.7", "text": "Cybersecurity and innovation positive.", "variables": ["X5", "X6"]},
        {"id": "H1.8", "text": "XAI interprets factor importance."},
    ]


def _load_fe_coefficients(root: Path) -> pd.DataFrame:
    path = root / "results" / "econometrics" / "coefficients.csv"
    if not path.exists():
        alt = root / "outputs" / "tables" / "coefficients.csv"
        path = alt if alt.exists() else path
    if not path.exists():
        return pd.DataFrame()
    df = read_df(path)
    if df.empty:
        return df
    # Prefer full-sample FE
    if "model" in df.columns:
        fe = df[df["model"].astype(str).isin(["FE", "fe", "PanelOLS", "within"])].copy()
        if fe.empty:
            # first non-group model
            fe = df[~df["model"].astype(str).str.contains("group", case=False, na=False)].copy()
        if fe.empty:
            fe = df.copy()
    else:
        fe = df.copy()
    if "term" in fe.columns:
        fe["base_var"] = fe["term"].map(_base_var)
    return fe


def _coef_evidence(
    coef_df: pd.DataFrame,
    variables: Sequence[str],
    *,
    expect_positive: bool = True,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    if coef_df.empty:
        return {
            "status": "inconclusive",
            "reason": "econometric coefficients missing",
            "variables": list(variables),
            "details": [],
        }
    details = []
    supported_flags = []
    for var in variables:
        rows = coef_df[coef_df["base_var"] == var] if "base_var" in coef_df.columns else pd.DataFrame()
        if rows.empty and "term" in coef_df.columns:
            rows = coef_df[coef_df["term"].astype(str).str.startswith(var)]
        if rows.empty:
            details.append({"variable": var, "found": False})
            supported_flags.append(False)
            continue
        row = rows.iloc[0]
        coef = float(row["coefficient"]) if "coefficient" in row and pd.notna(row["coefficient"]) else np.nan
        pval = float(row["pvalue"]) if "pvalue" in row and pd.notna(row["pvalue"]) else np.nan
        sign_ok = (coef > 0) if expect_positive else (coef < 0)
        sig = bool(pd.notna(pval) and pval < alpha)
        ok = bool(sign_ok and sig)
        supported_flags.append(ok)
        details.append(
            {
                "variable": var,
                "found": True,
                "term": str(row.get("term")),
                "coefficient": coef,
                "pvalue": pval,
                "significant": sig,
                "sign_matches": bool(sign_ok),
                "supported": ok,
            }
        )

    n_ok = sum(1 for x in supported_flags if x)
    n = len(variables)
    if n_ok == n and n > 0:
        status = "supported"
    elif n_ok > 0:
        status = "partially_supported"
    elif any(d.get("sign_matches") for d in details if d.get("found")):
        status = "partially_supported"
        # correct signs but not significant
    elif all(not d.get("found") for d in details):
        status = "inconclusive"
    else:
        status = "not_supported"

    return {
        "status": status,
        "variables": list(variables),
        "n_supported": n_ok,
        "n_variables": n,
        "details": details,
        "alpha": alpha,
    }


def _load_alternative_coefficients(root: Path, filename: str) -> pd.DataFrame:
    """Load an alternative-specification coefficient table from results/imputation/."""
    path = root / "results" / "imputation" / filename
    if not path.exists():
        return pd.DataFrame()
    df = read_df(path)
    if df.empty or "term" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["base_var"] = df["term"].map(_base_var)
    return df


def _robustness_for_coefficients(
    root: Path, variables: Sequence[str], headline_status: str
) -> Dict[str, Any]:
    """
    Re-adjudicate a coefficient hypothesis under alternative missing-data handling.

    ``mi_pooled`` uses Rubin-pooled standard errors across multiple imputations
    (the inference-valid variant when cells are reconstructed); ``complete_case``
    uses observed cells only.
    """
    out: Dict[str, Any] = {}
    specs = {
        "mi_pooled": "mi_pooled_fe.csv",
        "complete_case": "complete_case_fe.csv",
    }
    statuses = []
    for label, filename in specs.items():
        table = _load_alternative_coefficients(root, filename)
        if table.empty:
            continue
        evidence = _coef_evidence(table, variables)
        out[label] = {
            "status": evidence.get("status"),
            "details": evidence.get("details"),
        }
        statuses.append(evidence.get("status"))
    if statuses:
        out["consistent_with_headline"] = all(s == headline_status for s in statuses)
        out["statuses"] = {"point_panel": headline_status, **{k: v["status"] for k, v in out.items() if isinstance(v, dict) and "status" in v}}
    return out


def _eval_h15(root: Path) -> Dict[str, Any]:
    """ANOVA / Kruskal–Wallis group differences on DMI."""
    panel_paths = [
        root / "data" / "processed" / "analysis_panel.parquet",
        root / "outputs" / "data" / "dmi_panel.parquet",
        root / "data" / "processed" / "dmi_panel.parquet",
    ]
    panel = None
    for p in panel_paths:
        if p.exists():
            panel = read_df(p)
            break
    if panel is None or "DMI" not in panel.columns or "group_id" not in panel.columns:
        return {"status": "inconclusive", "reason": "panel with DMI/group_id missing"}

    groups = []
    labels = []
    for gid, g in panel.groupby("group_id"):
        vals = g["DMI"].dropna().values
        if len(vals) >= 2:
            groups.append(vals)
            labels.append(str(gid))
    if len(groups) < 2:
        return {"status": "inconclusive", "reason": "fewer than 2 groups with data", "groups": labels}

    # Prefer ANOVA if roughly normal / equal; always report Kruskal as robust
    anova_stat, anova_p = stats.f_oneway(*groups)
    kruskal_stat, kruskal_p = stats.kruskal(*groups)
    # Decision: either test significant => supported
    supported = bool(
        (pd.notna(anova_p) and anova_p < ALPHA)
        or (pd.notna(kruskal_p) and kruskal_p < ALPHA)
    )
    means = {lab: float(np.mean(g)) for lab, g in zip(labels, groups)}
    return {
        "status": "supported" if supported else "not_supported",
        "anova": {"statistic": float(anova_stat), "pvalue": float(anova_p)},
        "kruskal_wallis": {"statistic": float(kruskal_stat), "pvalue": float(kruskal_p)},
        "group_means": means,
        "groups": labels,
        "alpha": ALPHA,
    }


def _eval_h16(root: Path) -> Dict[str, Any]:
    """ML best RMSE < econometric/linear baseline RMSE."""
    path = root / "results" / "machine_learning" / "model_comparison.csv"
    if not path.exists():
        return {"status": "inconclusive", "reason": "model_comparison.csv missing"}
    df = read_df(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty or "rmse" not in df.columns:
        return {"status": "inconclusive", "reason": "no successful ML models with RMSE"}

    baselines = {"linear_regression", "ridge", "ols", "econometric", "fe", "re"}
    df["model_type"] = df["model_type"].astype(str)
    base = df[df["model_type"].str.lower().isin(baselines)]
    ml = df[~df["model_type"].str.lower().isin(baselines)]

    if ml.empty:
        return {"status": "inconclusive", "reason": "no non-baseline ML models"}
    if base.empty:
        return {"status": "inconclusive", "reason": "no linear/econometric baseline RMSE"}

    best_ml = ml.sort_values("rmse").iloc[0]
    best_base = base.sort_values("rmse").iloc[0]
    ml_rmse = float(best_ml["rmse"])
    base_rmse = float(best_base["rmse"])
    supported = ml_rmse < base_rmse
    return {
        "status": "supported" if supported else "not_supported",
        "best_ml": {
            "model_type": str(best_ml["model_type"]),
            "rmse": ml_rmse,
        },
        "best_baseline": {
            "model_type": str(best_base["model_type"]),
            "rmse": base_rmse,
        },
        "rmse_improvement": base_rmse - ml_rmse,
        "relative_improvement": (base_rmse - ml_rmse) / base_rmse if base_rmse else None,
    }


def _top_xai_features(root: Path, top_k: int = 10) -> Tuple[List[str], Dict[str, Any]]:
    candidates = [
        root / "results" / "xai" / "shap_global_importance.csv",
        root / "results" / "xai" / "xai_shap_global.csv",
        root / "results" / "xai" / "shap_importance.csv",
        root / "results" / "xai" / "feature_importance.csv",
        root / "results" / "xai" / "xai_feature_importance.csv",
        root / "outputs" / "xai" / "shap_global_importance.csv",
        root / "outputs" / "xai" / "shap_importance.csv",
        root / "outputs" / "xai" / "feature_importance.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    meta: Dict[str, Any] = {"xai_artifacts_exist": False}
    xai_dirs = [root / "results" / "xai", root / "outputs" / "xai"]
    artifact_count = 0
    for d in xai_dirs:
        if d.is_dir():
            artifact_count += sum(1 for f in d.rglob("*") if f.is_file())
    meta["artifact_count"] = artifact_count
    meta["xai_artifacts_exist"] = artifact_count > 0

    if path is None and artifact_count == 0:
        return [], meta

    if path is None:
        return [], meta

    df = read_df(path)
    meta["xai_artifacts_exist"] = True
    meta["source"] = str(path.relative_to(root))
    cols = {c.lower(): c for c in df.columns}
    feature_col = next(
        (cols[k] for k in ("feature", "variable", "term", "name") if k in cols),
        df.columns[0],
    )
    value_col = next(
        (
            cols[k]
            for k in ("mean_abs_shap", "importance", "shap_importance", "value")
            if k in cols
        ),
        None,
    )
    if value_col is None:
        numeric = [
            c for c in df.columns if c != feature_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric:
            return [], meta
        value_col = numeric[0]
    work = df[[feature_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float).abs()
    work = work.sort_values(value_col, ascending=False).head(top_k)
    features = [_base_var(x) for x in work[feature_col].astype(str).tolist()]
    return features, meta


def _eval_h18(root: Path) -> Dict[str, Any]:
    features, meta = _top_xai_features(root)
    if not meta.get("xai_artifacts_exist"):
        return {
            "status": "not_supported",
            "reason": "XAI artifacts missing under results/xai or outputs/xai",
            **meta,
        }
    if not features:
        # Artifacts exist but no parseable importance table
        return {
            "status": "partially_supported",
            "reason": "XAI artifacts exist but importance table not parseable",
            **meta,
        }
    overlap = sorted(set(features) & HYPOTHESIZED_DRIVERS)
    # Also allow lag-stripped partial matches like X1 from X1_lag1 already handled
    ratio = len(overlap) / max(len(HYPOTHESIZED_DRIVERS), 1)
    if len(overlap) >= 3 or ratio >= 0.25:
        status = "supported"
    elif overlap:
        status = "partially_supported"
    else:
        status = "not_supported"
    return {
        "status": status,
        "top_features": features,
        "overlap_with_hypothesized_drivers": overlap,
        "hypothesized_drivers": sorted(HYPOTHESIZED_DRIVERS),
        **meta,
    }


def _aggregate_h1(sub: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    order = ["H1.1", "H1.2", "H1.3", "H1.4", "H1.5", "H1.6", "H1.7", "H1.8"]
    scores = {"supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0, "inconclusive": 0.0}
    vals = []
    breakdown = {}
    for hid in order:
        st = (sub.get(hid) or {}).get("status", "inconclusive")
        breakdown[hid] = st
        vals.append(scores.get(st, 0.0))
    mean_score = float(np.mean(vals)) if vals else 0.0
    n_sup = sum(1 for v in breakdown.values() if v == "supported")
    n_part = sum(1 for v in breakdown.values() if v == "partially_supported")
    if mean_score >= 0.75:
        status = "supported"
    elif mean_score >= 0.4:
        status = "partially_supported"
    else:
        status = "not_supported"
    return {
        "status": status,
        "mean_support_score": mean_score,
        "n_supported": n_sup,
        "n_partially_supported": n_part,
        "n_subhypotheses": len(order),
        "breakdown": breakdown,
    }


def evaluate_hypotheses(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    hyps = _load_hypotheses(config)
    hyp_text = {h.get("id"): h for h in hyps}
    coef = _load_fe_coefficients(root)

    results: Dict[str, Dict[str, Any]] = {}

    def wrap(hid: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        meta = hyp_text.get(hid) or {}
        return {
            "id": hid,
            "text": meta.get("text"),
            "variables": meta.get("variables"),
            **evidence,
        }

    # Coefficient-based hypotheses
    coefficient_hypotheses = {
        "H1.1": ["X1", "X2"],
        "H1.2": ["X3"],
        "H1.3": ["X4"],
        "H1.4": ["C1", "C2"],
        "H1.7": ["X5", "X6"],
    }
    for hid, default_vars in coefficient_hypotheses.items():
        variables = (hyp_text.get(hid) or {}).get("variables") or default_vars
        evidence = _coef_evidence(coef, variables)
        robustness = _robustness_for_coefficients(root, variables, evidence.get("status", ""))
        if robustness:
            evidence["robustness"] = robustness
        results[hid] = wrap(hid, evidence)

    results["H1.5"] = wrap("H1.5", _eval_h15(root))
    h16 = _eval_h16(root)
    leakage_safe = read_json(
        root / "results" / "imputation" / "leakage_safe_ml_verdict.json"
    ) if (root / "results" / "imputation" / "leakage_safe_ml_verdict.json").exists() else None
    if leakage_safe:
        h16["robustness"] = {
            "leakage_safe": leakage_safe,
            "consistent_with_headline": leakage_safe.get("status") == h16.get("status"),
            "note": (
                "Leakage-safe variant re-fits the imputer on training years only, so "
                "hold-out accuracy carries no look-ahead information."
            ),
        }
    results["H1.6"] = wrap("H1.6", h16)
    results["H1.8"] = wrap("H1.8", _eval_h18(root))
    results["H1"] = wrap("H1", _aggregate_h1(results))

    robustness_summary: Dict[str, Any] = {}
    for hid, body in results.items():
        rb = body.get("robustness") or {}
        if not rb:
            continue
        entry: Dict[str, Any] = {"point_panel": body.get("status")}
        for label in ("mi_pooled", "complete_case"):
            if isinstance(rb.get(label), dict):
                entry[label] = rb[label].get("status")
        if isinstance(rb.get("leakage_safe"), dict):
            entry["leakage_safe"] = rb["leakage_safe"].get("status")
        entry["consistent"] = bool(rb.get("consistent_with_headline"))
        robustness_summary[hid] = entry

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alpha": ALPHA,
        "hypotheses": results,
        "summary": {
            hid: results[hid]["status"]
            for hid in ["H1", "H1.1", "H1.2", "H1.3", "H1.4", "H1.5", "H1.6", "H1.7", "H1.8"]
        },
        "robustness_summary": robustness_summary,
    }


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """Evaluate hypotheses and write JSON under results/ and outputs/hypothesis_testing/."""
    root = Path(project_root).resolve()
    payload = evaluate_hypotheses(root, config)

    results_dir = ensure_dir(root / "results" / "hypotheses")
    out_dir = ensure_dir(root / "outputs" / "hypothesis_testing")

    primary = results_dir / "hypothesis_evaluation.json"
    write_json(primary, payload)
    write_json(out_dir / "hypothesis_evaluation.json", payload)

    # Flat summary table
    rows = []
    for hid, body in (payload.get("hypotheses") or {}).items():
        rows.append(
            {
                "id": hid,
                "status": body.get("status"),
                "text": body.get("text"),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "hypothesis_summary.csv", index=False)
    summary_df.to_csv(results_dir / "hypothesis_summary.csv", index=False)

    # Markdown brief
    lines = ["# Hypothesis evaluation", "", f"Generated: {payload['generated_at']}", ""]
    for hid in ["H1", "H1.1", "H1.2", "H1.3", "H1.4", "H1.5", "H1.6", "H1.7", "H1.8"]:
        body = payload["hypotheses"][hid]
        lines.append(f"## {hid}: **{body.get('status')}**")
        if body.get("text"):
            lines.append(f"{body['text']}")
        lines.append("")
    (out_dir / "hypothesis_evaluation.md").write_text("\n".join(lines), encoding="utf-8")

    registry = ResultRegistry(root)
    registry.register(
        "hypothesis_evaluation",
        payload,
        category="hypotheses",
        fmt="json",
        meta={"alpha": ALPHA},
    )

    logger.info("Hypothesis evaluation written to %s", primary)
    return {
        "path": str(primary.relative_to(root)),
        "summary": payload["summary"],
    }
