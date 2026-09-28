"""Explainability (XAI) for the best trained DMI prediction model."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from common.errors import FrameworkError
from common.io import ensure_dir, read_json, write_df, write_json
from common.logging_utils import get_logger
from common.seeds import set_global_seed
from registries.model_registry import ModelRegistry
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.explainability")

TREE_MODEL_TYPES = {"random_forest", "xgboost", "lightgbm", "catboost"}


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


def _seed(config: Any, xai_cfg: Dict[str, Any]) -> int:
    if "random_seed" in xai_cfg:
        return int(xai_cfg["random_seed"])
    if hasattr(config, "random_seed"):
        try:
            return int(config.random_seed())
        except Exception:
            pass
    return int((_as_mapping(config).get("research") or {}).get("random_seed", 42))


def _load_xy(project_root: Path, config: Any):
    """Load analysis panel and lagged predictor set matching ML training."""
    from statistics.descriptive import load_analysis_panel, predictor_columns

    mapping = _as_mapping(config)
    ml_cfg = mapping.get("machine_learning") or {}
    eco = mapping.get("econometrics") or {}
    research = mapping.get("research") or {}

    features_cfg = list(ml_cfg.get("features") or eco.get("independent") or [])
    if not features_cfg:
        features_cfg = [f"X{i}" for i in range(1, 11)] + ["C1", "C2"]
    independent = [c for c in features_cfg if str(c).startswith("X")]
    controls = [c for c in features_cfg if c not in independent]
    if not independent:
        independent = list(features_cfg)
        controls = []

    use_lagged = bool(ml_cfg.get("use_lagged_features", eco.get("use_lagged_features", True)))
    lag = int(research.get("feature_lag") or 1)
    target = str(ml_cfg.get("target") or eco.get("dependent") or research.get("dependent_variable") or "DMI")

    df = load_analysis_panel(project_root)
    df, predictors = predictor_columns(
        df,
        independent=independent,
        controls=controls,
        use_lagged=use_lagged,
        lag=lag,
    )
    if not use_lagged:
        predictors = [c for c in features_cfg if c in df.columns]
    return df, predictors, target, ml_cfg


def _resolve_best_model(project_root: Path) -> Dict[str, Any]:
    """
    Resolve the best model artifact.

    Preference order:
    1. outputs/models/best_model.json (+ registry entry)
    2. outputs/models/registry.json (lowest test RMSE)
    3. results/machine_learning/model_comparison.csv (lowest RMSE among ok rows)
    """
    root = Path(project_root).resolve()
    best_path = root / "outputs" / "models" / "best_model.json"
    registry = ModelRegistry(root)

    if best_path.exists():
        best = read_json(best_path)
        model_id = best.get("model_id")
        if model_id:
            try:
                entry = registry.get(model_id)
                return {
                    "model_id": model_id,
                    "model_type": (entry.get("meta") or {}).get("model_type")
                    or best.get("best_model"),
                    "artifact_path": entry.get("artifact_path"),
                    "predictors": (entry.get("meta") or {}).get("predictors"),
                    "target": (entry.get("meta") or {}).get("target") or "DMI",
                    "rmse": best.get("rmse"),
                    "source": "best_model.json",
                    "entry": entry,
                }
            except KeyError:
                logger.warning("best_model.json model_id %s not in registry", model_id)

    # Registry scan by lowest RMSE
    models = registry.list()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in models:
        metrics = entry.get("metrics") or {}
        rmse = metrics.get("rmse")
        if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
            continue
        scored.append((float(rmse), entry))
    if scored:
        scored.sort(key=lambda x: x[0])
        rmse, entry = scored[0]
        return {
            "model_id": entry.get("model_id"),
            "model_type": (entry.get("meta") or {}).get("model_type"),
            "artifact_path": entry.get("artifact_path"),
            "predictors": (entry.get("meta") or {}).get("predictors"),
            "target": (entry.get("meta") or {}).get("target") or "DMI",
            "rmse": rmse,
            "source": "registry.json",
            "entry": entry,
        }

    # Fallback: model_comparison.csv
    comparison_candidates = [
        root / "results" / "machine_learning" / "model_comparison.csv",
        root / "outputs" / "tables" / "model_comparison.csv",
        root / "outputs" / "tables" / "ml_model_comparison.csv",
    ]
    for path in comparison_candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "rmse" not in df.columns:
            continue
        work = df.copy()
        if "status" in work.columns:
            work = work[work["status"].astype(str).str.lower() == "ok"]
        work = work.dropna(subset=["rmse"])
        if work.empty:
            continue
        row = work.loc[work["rmse"].astype(float).idxmin()]
        model_type = str(row.get("model_type") or "")
        model_id = row.get("model_id")
        artifact = row.get("artifact_path")
        if (not model_id or pd.isna(model_id)) and artifact:
            model_id = Path(str(artifact)).stem
        return {
            "model_id": None if model_id is None or (isinstance(model_id, float) and np.isnan(model_id)) else str(model_id),
            "model_type": model_type,
            "artifact_path": None if artifact is None or (isinstance(artifact, float) and np.isnan(artifact)) else str(artifact),
            "predictors": None,
            "target": "DMI",
            "rmse": float(row["rmse"]),
            "source": str(path.relative_to(root)),
            "entry": row.to_dict(),
        }

    raise FrameworkError(
        "Could not resolve a best ML model for explainability",
        details={
            "checked": [
                "outputs/models/best_model.json",
                "outputs/models/registry.json",
                "results/machine_learning/model_comparison.csv",
            ]
        },
    )


def _load_model_artifact(project_root: Path, info: Dict[str, Any]):
    root = Path(project_root).resolve()
    rel = info.get("artifact_path")
    candidates: List[Path] = []
    if rel:
        candidates.append(root / rel)
    model_id = info.get("model_id")
    if model_id:
        candidates.append(root / "outputs" / "models" / str(model_id) / f"{model_id}.joblib")
        candidates.append(root / "outputs" / "models" / f"{model_id}.joblib")
    model_type = info.get("model_type")
    if model_type and model_id is None:
        # glob latest flat artifact
        flat = sorted((root / "outputs" / "models").glob(f"{model_type}_*.joblib"))
        if flat:
            candidates.append(flat[-1])

    for path in candidates:
        if path is not None and path.exists():
            logger.info("Loading model artifact from %s", path)
            return joblib.load(path), path

    # Last resort: ModelRegistry.load_model
    if model_id:
        try:
            return ModelRegistry(root).load_model(str(model_id)), root / "outputs" / "models" / str(model_id)
        except Exception as exc:
            logger.warning("ModelRegistry.load_model failed: %s", exc)

    raise FileNotFoundError(
        f"No model artifact found for {info.get('model_id') or info.get('model_type')}"
    )


def _is_tree_estimator(model: Any, model_type: Optional[str]) -> bool:
    if model_type and str(model_type).lower() in TREE_MODEL_TYPES:
        return True
    name = type(model).__name__.lower()
    tree_hints = (
        "randomforest",
        "xgb",
        "lgbm",
        "lightgbm",
        "catboost",
        "decisiontree",
        "extratrees",
        "gradientboost",
        "histgradient",
    )
    return any(h in name for h in tree_hints)


def _native_importance(model: Any, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    """Extract native feature_importances_ or absolute coef_ when available."""
    est = model
    # Unwrap sklearn Pipeline
    if hasattr(model, "named_steps"):
        # Prefer final step with importances/coefs
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                est = step
                break

    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
        if len(vals) == len(feature_names):
            return pd.DataFrame(
                {"feature": list(feature_names), "importance": vals, "method": "native_feature_importances"}
            ).sort_values("importance", ascending=False)

    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float).ravel()
        if len(coef) == len(feature_names):
            vals = np.abs(coef)
            return pd.DataFrame(
                {"feature": list(feature_names), "importance": vals, "method": "native_abs_coef"}
            ).sort_values("importance", ascending=False)
    return None


def _permutation_importance_df(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_repeats: int = 10,
) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "method": "permutation",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _run_shap(
    model: Any,
    X: pd.DataFrame,
    *,
    model_type: Optional[str],
    max_samples: int,
    background_samples: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    import shap

    rng = np.random.default_rng(seed)
    n = len(X)
    if n == 0:
        raise ValueError("Empty feature matrix for SHAP")

    sample_idx = rng.choice(n, size=min(max_samples, n), replace=False)
    X_sample = X.iloc[sample_idx]
    bg_idx = rng.choice(n, size=min(background_samples, n), replace=False)
    X_bg = X.iloc[bg_idx]

    tree = _is_tree_estimator(model, model_type)
    if tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        explainer_type = "TreeExplainer"
    else:
        # KernelExplainer expects a predict callable and numpy background
        def _predict(data):
            frame = pd.DataFrame(data, columns=list(X.columns))
            return np.asarray(model.predict(frame), dtype=float)

        explainer = shap.KernelExplainer(_predict, X_bg.values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_sample.values, nsamples="auto")
        explainer_type = "KernelExplainer"

    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim == 3:
        # multiclass-style; take first output or mean abs across outputs
        sv = sv[0] if sv.shape[0] < sv.shape[-1] else sv.mean(axis=0)
    if sv.ndim != 2:
        sv = np.reshape(sv, (len(X_sample), -1))

    mean_abs = np.mean(np.abs(sv), axis=0)
    global_df = pd.DataFrame(
        {"feature": list(X.columns), "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    global_path = out_dir / "shap_global_importance.csv"
    write_df(global_df, global_path)

    # Sample local explanations (up to 10 rows)
    n_local = min(10, len(X_sample))
    local = []
    for i in range(n_local):
        row_vals = {feat: float(sv[i, j]) for j, feat in enumerate(X.columns)}
        local.append(
            {
                "sample_index": int(sample_idx[i]),
                "prediction": float(model.predict(X_sample.iloc[[i]])[0]),
                "shap_values": row_vals,
            }
        )
    local_path = out_dir / "shap_local_explanations.json"
    write_json(
        local_path,
        {
            "explainer": explainer_type,
            "n_samples": int(len(X_sample)),
            "n_background": int(len(X_bg)),
            "explanations": local,
        },
    )

    plot_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, max(4, 0.35 * len(X.columns))))
        shap.summary_plot(sv, X_sample, show=False, max_display=min(20, X.shape[1]))
        plot_path = out_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("SHAP summary plot failed: %s", exc)
        plot_path = None

    return {
        "explainer": explainer_type,
        "global_importance_path": str(global_path),
        "local_explanations_path": str(local_path),
        "summary_plot_path": str(plot_path) if plot_path else None,
        "global_df": global_df,
    }


def _run_lime(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_samples: int,
    num_features: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    from lime.lime_tabular import LimeTabularExplainer

    rng = np.random.default_rng(seed)
    n = len(X)
    n_explain = min(n_samples, n)
    idx = rng.choice(n, size=n_explain, replace=False)
    X_np = X.values.astype(float)

    explainer = LimeTabularExplainer(
        X_np,
        feature_names=list(X.columns),
        class_names=[str(y.name or "target")],
        mode="regression",
        random_state=seed,
        discretize_continuous=True,
    )

    def _predict(data):
        frame = pd.DataFrame(data, columns=list(X.columns))
        return np.asarray(model.predict(frame), dtype=float)

    explanations = []
    for i in idx:
        exp = explainer.explain_instance(
            X_np[i],
            _predict,
            num_features=min(num_features, X.shape[1]),
        )
        explanations.append(
            {
                "sample_index": int(i),
                "prediction": float(model.predict(X.iloc[[i]])[0]),
                "y_true": float(y.iloc[i]) if i < len(y) else None,
                "explanation": [
                    {"feature": str(feat), "weight": float(weight)}
                    for feat, weight in exp.as_list()
                ],
            }
        )

    path = out_dir / "lime_explanations.json"
    write_json(
        path,
        {
            "n_samples": int(n_explain),
            "num_features": int(num_features),
            "explanations": explanations,
        },
    )
    return {"path": str(path), "n_samples": int(n_explain)}


def _run_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    native = _native_importance(model, list(X.columns))
    if native is not None:
        df = native
    else:
        df = _permutation_importance_df(model, X, y, seed=seed)
    path = out_dir / "feature_importance.csv"
    write_df(df, path)
    return {"path": str(path), "method": str(df["method"].iloc[0]) if len(df) else None, "df": df}


def _run_pdp(
    model: Any,
    X: pd.DataFrame,
    importance_df: Optional[pd.DataFrame],
    *,
    top_k: int,
    grid_resolution: int,
    out_dir: Path,
) -> Dict[str, Any]:
    from sklearn.inspection import partial_dependence

    if importance_df is not None and not importance_df.empty:
        col = "mean_abs_shap" if "mean_abs_shap" in importance_df.columns else "importance"
        if col not in importance_df.columns:
            col = importance_df.columns[1]
        ranked = list(importance_df.sort_values(col, ascending=False)["feature"])
    else:
        ranked = list(X.columns)

    features = [f for f in ranked if f in X.columns][: max(1, int(top_k))]
    rows = []
    for feat in features:
        try:
            pd_result = partial_dependence(
                model,
                X,
                features=[feat],
                grid_resolution=grid_resolution,
                kind="average",
            )
            # sklearn >=1.0 returns Bunch with grid_values / average
            grid = np.asarray(pd_result["grid_values"][0], dtype=float)
            avg = np.asarray(pd_result["average"][0], dtype=float)
            for g, a in zip(grid, avg):
                rows.append({"feature": feat, "grid_value": float(g), "partial_dependence": float(a)})
        except Exception as exc:
            logger.warning("PDP failed for feature %s: %s", feat, exc)

    pdp_df = pd.DataFrame(rows)
    csv_path = out_dir / "partial_dependence.csv"
    write_df(pdp_df, csv_path)

    plot_path = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            from sklearn.inspection import PartialDependenceDisplay

            fig, ax = plt.subplots(
                nrows=1,
                ncols=len(features),
                figsize=(4 * max(len(features), 1), 3.5),
                squeeze=False,
            )
            PartialDependenceDisplay.from_estimator(
                model,
                X,
                features=features,
                kind="average",
                grid_resolution=grid_resolution,
                ax=ax[0],
            )
            fig.suptitle("Partial dependence (top features)")
            plot_path = out_dir / "partial_dependence.png"
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            # Manual fallback plot from computed values
            if not pdp_df.empty:
                n_feat = len(features)
                fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 3.5), squeeze=False)
                for i, feat in enumerate(features):
                    sub = pdp_df[pdp_df["feature"] == feat]
                    axes[0, i].plot(sub["grid_value"], sub["partial_dependence"])
                    axes[0, i].set_title(feat)
                    axes[0, i].set_xlabel(feat)
                    axes[0, i].set_ylabel("Partial dependence")
                plot_path = out_dir / "partial_dependence.png"
                fig.tight_layout()
                fig.savefig(plot_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.warning("PDP plot failed: %s", exc)

    return {
        "csv_path": str(csv_path),
        "plot_path": str(plot_path) if plot_path else None,
        "features": features,
        "df": pdp_df,
    }


def _rel(path: Path | str, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Explain the best trained model via SHAP, LIME, feature importance, and PDP.

    Partial failures are recorded; raises only if no artifacts are produced.
    """
    root = Path(project_root).resolve()
    mapping = _as_mapping(config)
    xai_cfg = dict(mapping.get("xai") or {})
    seed = _seed(config, xai_cfg)
    set_global_seed(seed)

    methods = list(xai_cfg.get("methods") or ["shap", "lime", "feature_importance", "pdp"])
    shap_cfg = dict(xai_cfg.get("shap") or {})
    lime_cfg = dict(xai_cfg.get("lime") or {})
    pdp_cfg = dict(xai_cfg.get("pdp") or {})
    out_rel = xai_cfg.get("output_dir") or "outputs/xai"
    out_dir = ensure_dir(root / out_rel)
    results_dir = ensure_dir(root / "results" / "xai")
    registry = ResultRegistry(root)

    warnings_log: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {}
    produced = 0

    # Resolve model + data
    try:
        info = _resolve_best_model(root)
        model, artifact_path = _load_model_artifact(root, info)
        df, predictors, target, _ml_cfg = _load_xy(root, config)

        # Prefer predictors stored with the model when available
        stored_preds = info.get("predictors")
        if stored_preds:
            missing = [c for c in stored_preds if c not in df.columns]
            if not missing:
                predictors = list(stored_preds)
            else:
                warnings_log.append(
                    {
                        "stage": "predictors",
                        "warning": f"Stored predictors missing columns: {missing}; using training defaults",
                    }
                )
        target = str(info.get("target") or target)
        required = [target] + list(predictors)
        work = df.dropna(subset=[c for c in required if c in df.columns]).copy()
        if work.empty:
            raise FrameworkError("No complete rows available for XAI after dropna")
        X = work[list(predictors)]
        y = work[target]
        model_type = info.get("model_type")
    except Exception as exc:
        err = {
            "error": repr(exc),
            "message": "Failed to load model or analysis panel for XAI",
            "warnings": warnings_log,
        }
        write_json(results_dir / "xai_error_report.json", err)
        write_json(out_dir / "xai_error_report.json", err)
        raise FrameworkError(
            "Explainability produced no results: model/data load failed",
            details=err,
        ) from exc

    importance_df: Optional[pd.DataFrame] = None

    # --- Feature importance (always attempt; works without shap/lime) ---
    if "feature_importance" in methods:
        try:
            fi = _run_feature_importance(model, X, y, seed=seed, out_dir=out_dir)
            importance_df = fi["df"]
            write_df(importance_df, results_dir / "feature_importance.csv")
            registry.register(
                "xai_feature_importance",
                importance_df,
                category="xai",
                meta={"method": fi.get("method"), "model_id": info.get("model_id")},
            )
            artifacts["feature_importance"] = _rel(fi["path"], root)
            produced += 1
        except Exception as exc:
            logger.exception("Feature importance failed")
            warnings_log.append({"method": "feature_importance", "warning": repr(exc)})

    # --- SHAP ---
    if "shap" in methods:
        try:
            import shap  # noqa: F401
        except ImportError as exc:
            msg = f"shap unavailable: {exc}"
            logger.warning(msg)
            warnings_log.append({"method": "shap", "warning": msg})
        else:
            try:
                shap_out = _run_shap(
                    model,
                    X,
                    model_type=model_type,
                    max_samples=int(shap_cfg.get("max_samples") or 80),
                    background_samples=int(shap_cfg.get("background_samples") or 40),
                    seed=seed,
                    out_dir=out_dir,
                )
                write_df(shap_out["global_df"], results_dir / "shap_global_importance.csv")
                # Mirror local JSON into results
                local_src = Path(shap_out["local_explanations_path"])
                if local_src.exists():
                    write_json(results_dir / "shap_local_explanations.json", read_json(local_src))
                registry.register(
                    "xai_shap_global",
                    shap_out["global_df"],
                    category="xai",
                    meta={"explainer": shap_out["explainer"], "model_id": info.get("model_id")},
                )
                artifacts["shap"] = {
                    "global": _rel(shap_out["global_importance_path"], root),
                    "local": _rel(shap_out["local_explanations_path"], root),
                    "plot": _rel(shap_out["summary_plot_path"], root)
                    if shap_out.get("summary_plot_path")
                    else None,
                }
                # Prefer SHAP ranking for PDP when available
                if importance_df is None:
                    importance_df = shap_out["global_df"].rename(
                        columns={"mean_abs_shap": "importance"}
                    )
                produced += 1
            except Exception as exc:
                logger.exception("SHAP failed")
                warnings_log.append({"method": "shap", "warning": repr(exc)})

    # --- LIME ---
    if "lime" in methods:
        try:
            import lime  # noqa: F401
        except ImportError as exc:
            msg = f"lime unavailable: {exc}"
            logger.warning(msg)
            warnings_log.append({"method": "lime", "warning": msg})
        else:
            try:
                lime_out = _run_lime(
                    model,
                    X,
                    y,
                    n_samples=int(lime_cfg.get("n_samples") or 20),
                    num_features=int(lime_cfg.get("num_features") or 8),
                    seed=seed,
                    out_dir=out_dir,
                )
                # Mirror to results
                src = Path(lime_out["path"])
                if src.exists():
                    write_json(results_dir / "lime_explanations.json", read_json(src))
                registry.register(
                    "xai_lime",
                    read_json(src) if src.exists() else lime_out,
                    category="xai",
                    fmt="json",
                    meta={"model_id": info.get("model_id")},
                )
                artifacts["lime"] = _rel(lime_out["path"], root)
                produced += 1
            except Exception as exc:
                logger.exception("LIME failed")
                warnings_log.append({"method": "lime", "warning": repr(exc)})

    # --- PDP ---
    if "pdp" in methods:
        try:
            pdp_out = _run_pdp(
                model,
                X,
                importance_df,
                top_k=int(pdp_cfg.get("features_top_k") or 4),
                grid_resolution=int(pdp_cfg.get("grid_resolution") or 20),
                out_dir=out_dir,
            )
            if pdp_out.get("df") is not None and not pdp_out["df"].empty:
                write_df(pdp_out["df"], results_dir / "partial_dependence.csv")
                registry.register(
                    "xai_partial_dependence",
                    pdp_out["df"],
                    category="xai",
                    meta={"features": pdp_out.get("features"), "model_id": info.get("model_id")},
                )
                artifacts["pdp"] = {
                    "csv": _rel(pdp_out["csv_path"], root),
                    "plot": _rel(pdp_out["plot_path"], root) if pdp_out.get("plot_path") else None,
                }
                produced += 1
            else:
                warnings_log.append({"method": "pdp", "warning": "PDP produced empty results"})
        except Exception as exc:
            logger.exception("PDP failed")
            warnings_log.append({"method": "pdp", "warning": repr(exc)})

    summary = {
        "model_id": info.get("model_id"),
        "model_type": model_type,
        "model_source": info.get("source"),
        "artifact_path": _rel(artifact_path, root),
        "rmse": info.get("rmse"),
        "target": target,
        "predictors": list(predictors),
        "n_rows": int(len(X)),
        "methods_requested": methods,
        "artifacts": artifacts,
        "warnings": warnings_log,
        "n_produced": produced,
        "seed": seed,
    }
    write_json(results_dir / "xai_summary.json", summary)
    write_json(out_dir / "xai_summary.json", summary)
    registry.register("xai_summary", summary, category="xai", fmt="json")

    if warnings_log:
        write_json(results_dir / "xai_warnings.json", warnings_log)
        write_json(out_dir / "xai_warnings.json", warnings_log)
        registry.register("xai_warnings", warnings_log, category="xai", fmt="json")

    if produced == 0:
        err = {
            "error": "No XAI artifacts produced",
            "warnings": warnings_log,
            "model_id": info.get("model_id"),
        }
        write_json(results_dir / "xai_error_report.json", err)
        write_json(out_dir / "xai_error_report.json", err)
        raise FrameworkError(
            "Explainability produced no results",
            details=err,
        )

    logger.info(
        "XAI complete for %s (%s produced, %s warnings)",
        info.get("model_id") or model_type,
        produced,
        len(warnings_log),
    )
    return summary
