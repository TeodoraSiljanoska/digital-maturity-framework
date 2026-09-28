"""Copy / prepare pipeline artifacts into ``outputs/dashboard/`` for the Streamlit app."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from common.io import ensure_dir, read_df, write_df, write_json
from common.logging_utils import get_logger

logger = get_logger("dmf.dashboard.export")


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "dashboard"):
        if hasattr(config, key):
            out[key] = getattr(config, key) or {}
    return out


def _copy_if_exists(src: Path, dest: Path) -> Optional[str]:
    if not src.exists():
        return None
    ensure_dir(dest.parent)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return str(dest)


def _panel_to_dashboard(root: Path, out_dir: Path) -> Optional[str]:
    candidates = [
        root / "data" / "processed" / "analysis_panel.parquet",
        root / "outputs" / "data" / "dmi_panel.parquet",
        root / "data" / "processed" / "dmi_panel.parquet",
    ]
    for src in candidates:
        if src.exists():
            df = read_df(src)
            # Keep lean columns for dashboard
            keep = [
                c
                for c in (
                    "country_iso3",
                    "year",
                    "group_id",
                    "DMI",
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
                )
                if c in df.columns
            ]
            out = out_dir / "panel.parquet"
            write_df(df[keep] if keep else df, out)
            write_df(df[keep] if keep else df, out_dir / "panel.csv")
            return str(out.relative_to(root))
    return None


def _export_predictions(root: Path, out_dir: Path) -> Optional[str]:
    pred_dir = root / "outputs" / "predictions"
    if not pred_dir.is_dir():
        return None
    files = sorted(pred_dir.glob("*.csv"))
    if not files:
        return None

    # Prefer best model predictions when available
    best_type = None
    best_path = root / "outputs" / "models" / "best_model.json"
    if best_path.exists():
        try:
            best_type = json.loads(best_path.read_text(encoding="utf-8")).get("best_model")
        except Exception:
            best_type = None

    chosen = files[-1]
    if best_type:
        for f in reversed(files):
            if f.name.startswith(str(best_type)):
                chosen = f
                break

    frames = []
    for f in files:
        try:
            df = read_df(f)
            if "model_type" not in df.columns:
                df["model_type"] = f.stem.rsplit("_", 1)[0]
            frames.append(df)
        except Exception as exc:
            logger.warning("Skip prediction file %s: %s", f, exc)

    if frames:
        all_pred = pd.concat(frames, ignore_index=True)
        write_df(all_pred, out_dir / "predictions.parquet")
        write_df(all_pred, out_dir / "predictions.csv")

    shutil.copy2(chosen, out_dir / "predictions_best.csv")
    return str((out_dir / "predictions.parquet").relative_to(root))


def _importance_rows_from_df(df: pd.DataFrame, top_k: int = 15) -> List[Dict[str, Any]]:
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
            return []
        value_col = numeric[0]
    work = df[[feature_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float).abs()
    work = work.sort_values(value_col, ascending=False).head(top_k)
    return [
        {"feature": str(r[feature_col]), "importance": float(r[value_col])}
        for _, r in work.iterrows()
    ]


def _export_xai_summary(root: Path, out_dir: Path) -> Optional[str]:
    importance_csvs = [
        root / "outputs" / "xai" / "shap_global_importance.csv",
        root / "results" / "xai" / "shap_global_importance.csv",
        root / "results" / "xai" / "xai_shap_global.csv",
        root / "results" / "xai" / "feature_importance.csv",
        root / "outputs" / "xai" / "feature_importance.csv",
        root / "results" / "xai" / "xai_feature_importance.csv",
        root / "results" / "xai" / "shap_importance.csv",
        root / "outputs" / "xai" / "shap_importance.csv",
    ]
    top_factors: List[Dict[str, Any]] = []
    importance_source = None
    for src in importance_csvs:
        if src.exists():
            df = read_df(src)
            write_df(df, out_dir / "xai_summary.csv")
            write_df(df, out_dir / "xai_summary.parquet")
            top_factors = _importance_rows_from_df(df)
            importance_source = str(src.relative_to(root))
            break

    summary_json_candidates = [
        root / "outputs" / "xai" / "xai_summary.json",
        root / "results" / "xai" / "xai_summary.json",
    ]
    payload: Dict[str, Any] = {}
    for src in summary_json_candidates:
        if src.exists():
            try:
                payload = json.loads(src.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {"data": payload}
                payload.setdefault("source", str(src.relative_to(root)))
            except Exception:
                payload = {"source": str(src.relative_to(root))}
            break

    if top_factors:
        payload["top_factors"] = top_factors
        payload["importance_source"] = importance_source
    if payload:
        write_json(out_dir / "xai_summary.json", payload)
        return str((out_dir / "xai_summary.json").relative_to(root))

    # Fall through: try CSV-only path already handled; next is correlation fallback
    for src in importance_csvs:
        if not src.exists():
            continue
        df = read_df(src)
        write_df(df, out_dir / "xai_summary.csv")
        write_df(df, out_dir / "xai_summary.parquet")
        payload = {
            "source": str(src.relative_to(root)),
            "top_factors": _importance_rows_from_df(df),
        }
        write_json(out_dir / "xai_summary.json", payload)
        return str((out_dir / "xai_summary.json").relative_to(root))

    # Correlation fallback so dashboard explanations section is not empty when XAI missing
    corr_path = root / "results" / "descriptive" / "correlation_matrix.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
        if "DMI" in corr.columns:
            s = (
                corr["DMI"]
                .drop(labels=["DMI"], errors="ignore")
                .abs()
                .sort_values(ascending=False)
                .head(15)
            )
            rows = [{"feature": str(k), "importance": float(v)} for k, v in s.items()]
            write_df(pd.DataFrame(rows), out_dir / "xai_summary.csv")
            write_json(
                out_dir / "xai_summary.json",
                {
                    "source": "results/descriptive/correlation_matrix.csv",
                    "note": "Fallback abs(|corr|) with DMI; replace when XAI stage completes",
                    "top_factors": rows,
                },
            )
            return "outputs/dashboard/xai_summary.json"
    return None


def _export_model_comparison(root: Path, out_dir: Path) -> Optional[str]:
    src = root / "results" / "machine_learning" / "model_comparison.csv"
    if not src.exists():
        alt = root / "outputs" / "tables" / "model_comparison.csv"
        src = alt if alt.exists() else src
    if not src.exists():
        return None
    df = read_df(src)
    write_df(df, out_dir / "model_comparison.csv")
    write_df(df, out_dir / "model_comparison.parquet")
    best = root / "outputs" / "models" / "best_model.json"
    if best.exists():
        shutil.copy2(best, out_dir / "best_model.json")
    return str((out_dir / "model_comparison.parquet").relative_to(root))


def _export_convergence(root: Path, out_dir: Path, panel_rel: Optional[str]) -> Optional[str]:
    candidates = [
        root / "results" / "convergence" / "sigma_convergence.csv",
        root / "results" / "convergence" / "sigma.csv",
        root / "outputs" / "convergence" / "sigma_convergence.csv",
        root / "outputs" / "figures" / "sigma_series.csv",
    ]
    for src in candidates:
        if src.exists():
            df = read_df(src)
            write_df(df, out_dir / "convergence.csv")
            write_df(df, out_dir / "convergence.parquet")
            return str((out_dir / "convergence.parquet").relative_to(root))

    # Compute from panel already exported
    panel_path = out_dir / "panel.parquet"
    if panel_path.exists():
        panel = read_df(panel_path)
        if "year" in panel.columns and "DMI" in panel.columns:
            g = panel.groupby("year", as_index=False)["DMI"].agg(
                sigma="std", mean="mean", n="count"
            )
            g["cv"] = g["sigma"] / g["mean"].replace(0, pd.NA)
            g["source"] = "computed_from_panel"
            write_df(g, out_dir / "convergence.csv")
            write_df(g, out_dir / "convergence.parquet")
            return str((out_dir / "convergence.parquet").relative_to(root))
    return None


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Prepare dashboard-ready parquet/json copies from pipeline outputs only.

    Writes under ``outputs/dashboard/``. Does not invent research numbers.
    """
    root = Path(project_root).resolve()
    mapping = _as_mapping(config)
    dash_cfg = mapping.get("dashboard") or {}
    out_rel = dash_cfg.get("data_dir") or "outputs/dashboard"
    out_dir = ensure_dir(root / out_rel)

    exported: Dict[str, Optional[str]] = {}
    exported["panel"] = _panel_to_dashboard(root, out_dir)
    exported["predictions"] = _export_predictions(root, out_dir)
    exported["xai_summary"] = _export_xai_summary(root, out_dir)
    exported["model_comparison"] = _export_model_comparison(root, out_dir)
    exported["convergence"] = _export_convergence(root, out_dir, exported["panel"])

    extra_copies = {
        "coefficients": root / "results" / "econometrics" / "coefficients.csv",
        "vif": root / "outputs" / "tables" / "vif_predictors.csv",
        "correlation": root / "outputs" / "tables" / "correlation_matrix.csv",
        "group_comparison": root / "results" / "comparative" / "group_comparison.csv",
        "sparql_answers": root / "outputs" / "ontology" / "sparql_answers.json",
        "competency_answers": root / "outputs" / "ontology" / "competency_answers.json",
        "hypothesis_evaluation": root / "results" / "hypotheses" / "hypothesis_evaluation.json",
        "leakage_safe_verdict": root / "results" / "imputation" / "leakage_safe_ml_verdict.json",
        "sigma_mapping": root / "ontology" / "sigma.md",
        "choropleth": root / "outputs" / "figures" / "choropleth_latest.html",
    }
    for name, src in extra_copies.items():
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            exported[name] = str((out_dir / src.name).relative_to(root))

    prov = root / "data" / "processed" / "cell_provenance.parquet"
    if prov.exists():
        pdf = read_df(prov)
        write_df(pdf, out_dir / "provenance.parquet")
        write_df(pdf, out_dir / "provenance.csv")
        exported["provenance"] = "outputs/dashboard/provenance.parquet"

    try:
        from catalog.data_catalog import DataCatalog

        cat = DataCatalog.from_project(root).reusable_frame()
        write_df(cat, out_dir / "indicator_catalog.csv")
        exported["indicator_catalog"] = "outputs/dashboard/indicator_catalog.csv"
    except Exception as exc:
        logger.warning("Could not export indicator catalog: %s", exc)

    ont = root / "outputs" / "ontology" / "graph.json"
    if ont.exists():
        shutil.copy2(ont, out_dir / "ontology_graph.json")
        exported["ontology_graph"] = "outputs/dashboard/ontology_graph.json"

    # Preserve framework_status if already written; otherwise create a minimal stub pointer
    status_path = out_dir / "framework_status.json"
    if not status_path.exists():
        try:
            from framework.outputs import build_framework_status

            write_json(status_path, build_framework_status(root, config))
            exported["framework_status"] = str(status_path.relative_to(root))
        except Exception as exc:
            logger.warning("Could not build framework_status during dashboard export: %s", exc)
            exported["framework_status"] = None
    else:
        exported["framework_status"] = str(status_path.relative_to(root))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": out_rel,
        "artifacts": exported,
        "title": dash_cfg.get("title") or "Digital Maturity Monitoring Dashboard",
    }
    write_json(out_dir / "manifest.json", manifest)
    logger.info("Dashboard export complete: %s", exported)
    return manifest
