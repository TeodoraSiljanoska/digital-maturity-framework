"""Intelligent framework facade packaging monitoring / prediction / explanation APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.io import ensure_dir, read_df, read_json, write_json
from common.logging_utils import get_logger
from registries.model_registry import ModelRegistry
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.framework")


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "dashboard", "machine_learning", "xai", "convergence"):
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
    raise FileNotFoundError("DMI/analysis panel not found for framework outputs")


def _latest_glob(paths: Sequence[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _best_model_metrics(root: Path) -> Dict[str, Any]:
    best_path = root / "outputs" / "models" / "best_model.json"
    selection_path = root / "outputs" / "models" / "selection.json"
    comparison_path = root / "results" / "machine_learning" / "model_comparison.csv"
    out: Dict[str, Any] = {}
    if best_path.exists():
        out.update(read_json(best_path))
    elif selection_path.exists():
        out.update(read_json(selection_path))
    if comparison_path.exists():
        df = read_df(comparison_path)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        if not df.empty and "rmse" in df.columns:
            row = df.sort_values("rmse").iloc[0]
            out.setdefault("best_model", row.get("model_type"))
            out.setdefault("model_id", row.get("model_id"))
            out.setdefault("rmse", float(row["rmse"]) if pd.notna(row["rmse"]) else None)
            for metric in ("mae", "mape", "r2"):
                if metric in row and pd.notna(row[metric]):
                    out.setdefault(metric, float(row[metric]))
            out["comparison_n_ok"] = int(len(df))
    return out


def _top_shap_factors(root: Path, top_k: int = 8) -> List[Dict[str, Any]]:
    candidates = [
        root / "results" / "xai" / "shap_global_importance.csv",
        root / "results" / "xai" / "xai_shap_global.csv",
        root / "results" / "xai" / "shap_importance.csv",
        root / "results" / "xai" / "feature_importance.csv",
        root / "results" / "xai" / "xai_feature_importance.csv",
        root / "outputs" / "xai" / "shap_global_importance.csv",
        root / "outputs" / "xai" / "shap_importance.csv",
        root / "outputs" / "xai" / "feature_importance.csv",
        root / "outputs" / "xai" / "shap_summary.csv",
    ]
    path = _latest_glob(candidates)
    if path is None:
        search_dirs = [root / "results" / "xai", root / "outputs" / "xai"]
        found: List[Path] = []
        for d in search_dirs:
            if d.is_dir():
                found.extend(d.rglob("*.csv"))
        for cand in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
            if "shap" in cand.name.lower() or "importance" in cand.name.lower():
                path = cand
                break
    if path is None:
        # Fallback: absolute correlation with DMI from descriptive results
        corr_path = root / "results" / "descriptive" / "correlation_matrix.csv"
        if corr_path.exists():
            corr = pd.read_csv(corr_path, index_col=0)
            if "DMI" in corr.columns:
                s = corr["DMI"].drop(labels=["DMI"], errors="ignore").abs().sort_values(ascending=False)
                return [
                    {"feature": str(k), "importance": float(v), "source": "abs_corr_fallback"}
                    for k, v in s.head(top_k).items()
                ]
        return []

    df = read_df(path)
    cols = {c.lower(): c for c in df.columns}
    feature_col = next(
        (cols[k] for k in ("feature", "variable", "term", "name", "column") if k in cols),
        df.columns[0],
    )
    value_col = next(
        (
            cols[k]
            for k in (
                "mean_abs_shap",
                "importance",
                "shap_importance",
                "mean_shap",
                "value",
                "abs_mean",
            )
            if k in cols
        ),
        None,
    )
    if value_col is None:
        numeric = [
            c
            for c in df.columns
            if c != feature_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric:
            return []
        value_col = numeric[0]
    work = df[[feature_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float).abs()
    work = work.sort_values(value_col, ascending=False).head(top_k)
    return [
        {
            "feature": str(r[feature_col]),
            "importance": float(r[value_col]),
            "source": str(path.relative_to(root)),
        }
        for _, r in work.iterrows()
    ]


def _convergence_summary(root: Path, panel: pd.DataFrame) -> Dict[str, Any]:
    candidates = [
        root / "results" / "convergence" / "sigma_convergence.csv",
        root / "results" / "convergence" / "sigma.csv",
        root / "outputs" / "convergence" / "sigma_convergence.csv",
        root / "results" / "convergence" / "convergence_summary.json",
        root / "outputs" / "convergence" / "convergence_summary.json",
    ]
    for path in candidates:
        if path.exists() and path.suffix == ".json":
            return {"source": str(path.relative_to(root)), **read_json(path)}
        if path.exists() and path.suffix == ".csv":
            df = read_df(path)
            year_col = "year" if "year" in df.columns else df.columns[0]
            y_col = next(
                (c for c in ("sigma", "std", "dispersion") if c in df.columns),
                None,
            )
            if y_col is None:
                return {"source": str(path.relative_to(root)), "n_rows": int(len(df))}
            df = df.sort_values(year_col)
            first = float(df.iloc[0][y_col])
            last = float(df.iloc[-1][y_col])
            return {
                "source": str(path.relative_to(root)),
                "sigma_start": first,
                "sigma_end": last,
                "sigma_change": last - first,
                "converging": bool(last < first),
                "years": [int(df.iloc[0][year_col]), int(df.iloc[-1][year_col])],
            }

    if "year" in panel.columns and "DMI" in panel.columns:
        g = panel.groupby("year")["DMI"].std().dropna().sort_index()
        if len(g) >= 2:
            first, last = float(g.iloc[0]), float(g.iloc[-1])
            return {
                "source": "computed_from_panel",
                "sigma_start": first,
                "sigma_end": last,
                "sigma_change": last - first,
                "converging": bool(last < first),
                "years": [int(g.index[0]), int(g.index[-1])],
            }
    return {"source": None, "available": False}


def _latest_maturity_table(panel: pd.DataFrame) -> List[Dict[str, Any]]:
    if panel.empty or "DMI" not in panel.columns:
        return []
    work = panel.copy()
    if "year" in work.columns:
        latest_year = int(work["year"].max())
        work = work[work["year"] == latest_year]
    else:
        latest_year = None
    rows: List[Dict[str, Any]] = []
    group_cols = [c for c in ("country_iso3", "group_id") if c in work.columns]
    if not group_cols:
        return [{"year": latest_year, "mean_dmi": float(work["DMI"].mean())}]
    agg = work.groupby(group_cols, as_index=False)["DMI"].mean()
    for _, r in agg.iterrows():
        item = {"year": latest_year, "DMI": float(r["DMI"])}
        for c in group_cols:
            item[c] = r[c]
        rows.append(item)
    return rows


def build_framework_status(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    panel = _load_panel(root)
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "latest_dmi_by_country": _latest_maturity_table(panel),
        "best_model_metrics": _best_model_metrics(root),
        "top_shap_factors": _top_shap_factors(root),
        "convergence_summary": _convergence_summary(root, panel),
        "panel_rows": int(len(panel)),
        "panel_years": (
            [int(panel["year"].min()), int(panel["year"].max())]
            if "year" in panel.columns and len(panel)
            else None
        ),
        "n_countries": int(panel["country_iso3"].nunique())
        if "country_iso3" in panel.columns
        else None,
        "result_registry_count": len(ResultRegistry(root).list()),
        "model_registry_count": len(ModelRegistry(root).list()),
    }
    return status


class IntelligentFramework:
    """
    Monitoring / prediction / explanation facade over pipeline artifacts and registries.
    """

    def __init__(self, project_root: Path | str, config: Any = None):
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.result_registry = ResultRegistry(self.project_root)
        self.model_registry = ModelRegistry(self.project_root)
        self._status: Optional[Dict[str, Any]] = None
        self._panel: Optional[pd.DataFrame] = None

    def _ensure_status(self) -> Dict[str, Any]:
        if self._status is None:
            self._status = build_framework_status(self.project_root, self.config)
        return self._status

    def _ensure_panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = _load_panel(self.project_root)
        return self._panel

    def latest_maturity(
        self, country_iso3: Optional[str] = None, group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return latest DMI levels, optionally filtered by country or group."""
        status = self._ensure_status()
        rows = list(status.get("latest_dmi_by_country") or [])
        if country_iso3:
            rows = [r for r in rows if str(r.get("country_iso3")) == str(country_iso3)]
        if group_id:
            rows = [r for r in rows if str(r.get("group_id")) == str(group_id)]
        panel = self._ensure_panel()
        summary = {
            "year": rows[0]["year"] if rows else None,
            "records": rows,
            "mean_dmi": float(np.mean([r["DMI"] for r in rows])) if rows else None,
            "n": len(rows),
        }
        if "group_id" in panel.columns and rows:
            by_group = (
                pd.DataFrame(rows)
                .groupby("group_id")["DMI"]
                .mean()
                .to_dict()
                if "group_id" in pd.DataFrame(rows).columns
                else {}
            )
            summary["mean_by_group"] = {str(k): float(v) for k, v in by_group.items()}
        return summary

    def predict_path_info(self) -> Dict[str, Any]:
        """Describe available prediction artifacts and best-model prediction path."""
        root = self.project_root
        pred_dir = root / "outputs" / "predictions"
        files = sorted(pred_dir.glob("*.csv")) if pred_dir.is_dir() else []
        best = _best_model_metrics(root)
        best_type = best.get("best_model")
        matched = None
        if best_type:
            for f in reversed(files):
                if f.name.startswith(str(best_type)):
                    matched = f
                    break
        return {
            "best_model": best,
            "prediction_dir": "outputs/predictions",
            "prediction_files": [str(f.relative_to(root)) for f in files],
            "best_prediction_path": str(matched.relative_to(root)) if matched else None,
            "n_prediction_files": len(files),
        }

    def explanations_summary(self) -> Dict[str, Any]:
        """Summarize framework semantics (σ) and result XAI (SHAP/LIME)."""
        factors = _top_shap_factors(self.project_root)
        xai_dirs = [
            self.project_root / "results" / "xai",
            self.project_root / "outputs" / "xai",
        ]
        artifacts: List[str] = []
        for d in xai_dirs:
            if d.is_dir():
                for f in d.rglob("*"):
                    if f.is_file() and f.suffix.lower() in {".csv", ".json", ".png", ".html"}:
                        artifacts.append(str(f.relative_to(self.project_root)))

        semantics: List[Dict[str, Any]] = []
        scenarios: List[Dict[str, Any]] = []
        graph_path = self.project_root / "outputs" / "ontology" / "graph.json"
        if graph_path.exists():
            graph = read_json(graph_path)
            for ind in graph.get("indicators") or []:
                semantics.append(
                    {
                        "indicator_id": ind.get("indicator_id"),
                        "uri": ind.get("uri"),
                        "pillar": ind.get("pillar"),
                        "collection_method": ind.get("collection_method"),
                        "source": ind.get("source"),
                        "process_phases": ind.get("process_phases"),
                        "role": ind.get("role"),
                        "sigma": f"{ind.get('indicator_id')} → pillar {ind.get('pillar')} → method {ind.get('collection_method')}",
                    }
                )
            scenarios = list(graph.get("scenarios") or [])
            explanations = list(graph.get("explanations") or [])
        else:
            explanations = [
                {"indicator_id": f.get("feature"), "mean_abs_shap": f.get("importance"), "method": "SHAP"}
                for f in factors
            ]

        return {
            "top_factors": factors,
            "framework_semantics": semantics,
            "result_explanations": explanations,
            "contrasting_scenarios": scenarios,
            "n_xai_artifacts": len(artifacts),
            "artifacts": artifacts[:50],
            "available": bool(factors) or bool(artifacts) or bool(semantics),
            "dual": {
                "framework": "layered TBox + σ readings + process phase + collection method",
                "results": "reified SHAP/LIME linked to DecisionContext CTX_prediction_xai",
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """Lightweight readiness check across critical artifact layers."""
        root = self.project_root
        checks = {
            "raw_data": (root / "data" / "raw" / "integrated_raw.parquet").exists(),
            "analysis_panel": (root / "data" / "processed" / "analysis_panel.parquet").exists()
            or (root / "outputs" / "data" / "dmi_panel.parquet").exists(),
            "model_comparison": (
                root / "results" / "machine_learning" / "model_comparison.csv"
            ).exists(),
            "econometrics_coefficients": (
                root / "results" / "econometrics" / "coefficients.csv"
            ).exists(),
            "figures_dir": (root / "outputs" / "figures").is_dir(),
            "dashboard_dir": (root / "outputs" / "dashboard").is_dir(),
            "framework_status": (
                root / "outputs" / "dashboard" / "framework_status.json"
            ).exists(),
        }
        return {
            "ok": all(
                checks[k]
                for k in ("raw_data", "analysis_panel", "model_comparison")
            ),
            "checks": checks,
            "best_model_present": bool(_best_model_metrics(root).get("best_model")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Alias matching user-facing API wording
    def predict_path(self) -> Dict[str, Any]:
        return self.predict_path_info()


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """Build framework status JSON and return facade health snapshot."""
    root = Path(project_root).resolve()
    fw = IntelligentFramework(root, config)
    status = build_framework_status(root, config)
    out_dir = ensure_dir(root / "outputs" / "dashboard")
    status_path = out_dir / "framework_status.json"
    write_json(status_path, status)

    # Also mirror under outputs/framework for audit convenience
    fw_dir = ensure_dir(root / "outputs" / "framework")
    write_json(fw_dir / "framework_status.json", status)
    write_json(fw_dir / "health_check.json", fw.health_check())
    write_json(fw_dir / "explanations_summary.json", fw.explanations_summary())
    write_json(fw_dir / "predict_path_info.json", fw.predict_path_info())
    expl = fw.explanations_summary()
    write_json(fw_dir / "framework_semantics.json", expl.get("framework_semantics") or [])
    write_json(fw_dir / "contrasting_scenarios.json", expl.get("contrasting_scenarios") or [])

    logger.info("Wrote framework status to %s", status_path)
    return {
        "status_path": str(status_path.relative_to(root)),
        "health": fw.health_check(),
        "latest_maturity_n": len(status.get("latest_dmi_by_country") or []),
        "best_model": (status.get("best_model_metrics") or {}).get("best_model"),
    }
