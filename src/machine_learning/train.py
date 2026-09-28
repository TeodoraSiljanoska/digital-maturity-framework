"""Train ML models for lagged-feature → DMI prediction with temporal validation."""

from __future__ import annotations

import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common.errors import FrameworkError, InsufficientSampleError
from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger
from common.seeds import set_global_seed
from experiment_tracking.tracker import ExperimentTracker
from model_selection import compare, save_comparison
from registries.model_registry import ModelRegistry
from registries.result_registry import ResultRegistry

logger = get_logger("dmf.machine_learning")


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


def _seed(config: Any, ml_cfg: Dict[str, Any]) -> int:
    if "random_seed" in ml_cfg:
        return int(ml_cfg["random_seed"])
    if hasattr(config, "random_seed"):
        try:
            return int(config.random_seed())
        except Exception:
            pass
    return int((_as_mapping(config).get("research") or {}).get("random_seed", 42))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_xy(project_root: Path, config: Any):
    from statistics.descriptive import load_analysis_panel, predictor_columns

    mapping = _as_mapping(config)
    ml_cfg = mapping.get("machine_learning") or {}
    eco = mapping.get("econometrics") or {}
    research = mapping.get("research") or {}

    features_cfg = list(ml_cfg.get("features") or eco.get("independent") or [])
    if not features_cfg:
        features_cfg = [f"X{i}" for i in range(1, 11)] + ["C1", "C2"]
    # Split into X vs C heuristically for lag builder
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
    # If config listed explicit lag names already present, prefer intersection
    if not use_lagged:
        predictors = [c for c in features_cfg if c in df.columns]
    return df, predictors, target, ml_cfg


def temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str = "year",
    test_years: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(int(y) for y in test_years)
    if not years:
        raise ValueError("test_years must be non-empty")
    min_test = min(years)
    train = df[df[time_col] < min_test].copy()
    test = df[df[time_col].isin(years)].copy()
    return train, test


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # Avoid division explosions in MAPE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            mask = np.abs(y_true) > 1e-8
            mape = (
                float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
                if mask.any()
                else float("nan")
            )
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _param_distributions(model_type: str, seed: int) -> Dict[str, Any]:
    if model_type == "random_forest":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 1.0],
        }
    if model_type == "xgboost":
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": [100, 200, 300],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_samples": [5, 10, 20],
            "subsample": [0.7, 0.9, 1.0],
        }
    if model_type == "catboost":
        return {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
        }
    if model_type == "svr":
        return {
            "svr__C": [0.1, 1.0, 10.0],
            "svr__epsilon": [0.01, 0.1, 0.2],
            "svr__gamma": ["scale", "auto"],
        }
    if model_type == "mlp":
        return {
            "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-3, 1e-2],
        }
    return {}


def _build_estimator(model_type: str, seed: int):
    """Return (estimator, optional ImportError warning message)."""
    if model_type == "random_forest":
        return (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            None,
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor

            return (
                XGBRegressor(
                    random_state=seed,
                    n_jobs=-1,
                    objective="reg:squarederror",
                    verbosity=0,
                ),
                None,
            )
        except Exception as exc:  # ImportError or missing libomp runtime
            return None, f"xgboost unavailable: {exc}"
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return (
                LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1),
                None,
            )
        except Exception as exc:
            return None, f"lightgbm unavailable: {exc}"
    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor

            return (
                CatBoostRegressor(
                    random_seed=seed, verbose=False, loss_function="RMSE"
                ),
                None,
            )
        except Exception as exc:
            return None, f"catboost unavailable: {exc}"
    if model_type == "svr":
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ]
        )
        return pipe, None
    if model_type == "mlp":
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        random_state=seed,
                        max_iter=500,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        return pipe, None
    if model_type == "linear_regression":
        return LinearRegression(), None
    if model_type == "ridge":
        return Ridge(random_state=seed, alpha=1.0), None
    return None, f"unknown model_type: {model_type}"


def _is_tree_model(model_type: str) -> bool:
    return model_type in {"random_forest", "xgboost", "lightgbm", "catboost"}


def _fit_with_search(
    model_type: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int,
    n_iter: int,
    cv_folds: int,
    scoring: str,
):
    param_dist = _param_distributions(model_type, seed)
    if not param_dist:
        estimator.fit(X_train, y_train)
        return estimator, {}

    # Approximate cardinality and cap n_iter
    card = 1
    for v in param_dist.values():
        card *= max(len(list(v)), 1)
    n_iter_eff = max(1, min(n_iter, card))
    if model_type in {"svr", "mlp"}:
        n_iter_eff = min(n_iter_eff, 6)

    n_splits = min(cv_folds, max(2, len(X_train) // 5))
    n_splits = min(n_splits, len(X_train))
    if n_splits < 2 or len(X_train) < 10:
        estimator.fit(X_train, y_train)
        return estimator, {"search": "skipped_small_sample"}

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter_eff,
        scoring=scoring,
        cv=n_splits,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "n_iter": int(search.n_iter),
        "cv_folds": int(n_splits),
    }


def _save_model_artifact(
    project_root: Path,
    model_type: str,
    model,
    timestamp: str,
) -> Path:
    models_dir = ensure_dir(project_root / "outputs" / "models")
    path = models_dir / f"{model_type}_{timestamp}.joblib"
    joblib.dump(model, path)
    return path


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Train configured ML models (+ linear/ridge baselines), evaluate on temporal holdout,
    register artifacts, and select the best model by test RMSE.
    """
    root = Path(project_root).resolve()
    mapping = _as_mapping(config)
    df, predictors, target, ml_cfg = _load_xy(root, config)
    seed = _seed(config, ml_cfg)
    set_global_seed(seed)

    validation = ml_cfg.get("validation") or {}
    test_years = list(validation.get("test_years") or [2021, 2022, 2023])
    cv_folds = int(validation.get("cv_folds") or 3)
    hp = ml_cfg.get("hyperparameter_search") or {}
    n_iter = int(hp.get("n_iter") or 8)
    scoring = str(hp.get("scoring") or "neg_root_mean_squared_error")
    min_train = int(ml_cfg.get("min_train_rows") or 25)
    model_types = list(ml_cfg.get("models") or [
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "svr",
        "mlp",
    ])
    # Econometric-style baselines for H1.6
    baseline_types = ["linear_regression", "ridge"]

    if "year" not in df.columns:
        raise KeyError("Analysis panel missing 'year' for temporal split")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' missing from analysis panel")

    # Drop ultra-sparse predictors (documented); keep official series otherwise.
    drop_ratio = float(ml_cfg.get("drop_predictor_missing_ratio") or 0.55)
    kept: List[str] = []
    dropped: List[str] = []
    for col in predictors:
        if col not in df.columns:
            dropped.append(col)
            continue
        miss = float(df[col].isna().mean())
        if miss > drop_ratio:
            dropped.append(col)
        else:
            kept.append(col)
    predictors = kept
    if dropped:
        logger.warning(
            "Dropped sparse/missing predictors for ML (>%s missing): %s",
            drop_ratio,
            ",".join(dropped),
        )
    if len(predictors) < 3:
        raise InsufficientSampleError(
            "Fewer than 3 usable predictors after sparsity filter",
            details={"dropped": dropped},
        )

    work = df.dropna(subset=[target]).copy()
    train_df, test_df = temporal_split(work, time_col="year", test_years=test_years)
    if len(train_df) < min_train:
        err = InsufficientSampleError(
            f"Training sample too small: {len(train_df)} < {min_train}",
            details={"n_train": len(train_df), "min_train_rows": min_train},
        )
        ensure_dir(root / "results" / "machine_learning")
        write_json(
            root / "results" / "machine_learning" / "insufficient_sample.json",
            err.to_dict(),
        )
        raise err
    if test_df.empty:
        raise InsufficientSampleError(
            "Temporal test set is empty; check test_years vs panel years",
            details={"test_years": test_years},
        )

    impute_strategy = str(ml_cfg.get("impute_features") or "median")
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train = pd.DataFrame(
        imputer.fit_transform(train_df[predictors]),
        columns=predictors,
        index=train_df.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(test_df[predictors]),
        columns=predictors,
        index=test_df.index,
    )
    y_train = train_df[target]
    y_test = test_df[target]
    write_json(
        ensure_dir(root / "results" / "machine_learning") / "feature_prep.json",
        {
            "predictors": predictors,
            "dropped_predictors": dropped,
            "impute_strategy": impute_strategy,
            "drop_predictor_missing_ratio": drop_ratio,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "test_years": list(test_years),
        },
    )

    results_dir = ensure_dir(root / "results" / "machine_learning")
    pred_dir = ensure_dir(root / "outputs" / "predictions")
    tables_dir = ensure_dir(root / "outputs" / "tables")
    model_registry = ModelRegistry(root)
    result_registry = ResultRegistry(root)
    tracker = ExperimentTracker(root)

    exp = tracker.create(
        name="ml_dmi_temporal",
        params={
            "target": target,
            "predictors": predictors,
            "test_years": test_years,
            "seed": seed,
            "n_iter": n_iter,
            "models": model_types + baseline_types,
        },
        tags=["machine_learning", "H1.6"],
    )
    experiment_id = exp["experiment_id"]
    tracker.update(experiment_id, status="running")

    ts = _timestamp()
    comparison_rows: List[Dict[str, Any]] = []
    warnings_log: List[Dict[str, Any]] = []
    successes = 0
    failures = 0

    all_types = list(model_types) + list(baseline_types)
    for model_type in all_types:
        logger.info("Training model: %s", model_type)
        estimator, import_warning = _build_estimator(model_type, seed)
        if estimator is None:
            msg = import_warning or f"Could not build {model_type}"
            logger.warning(msg)
            warnings_log.append(
                {"model_type": model_type, "status": "skipped", "warning": msg}
            )
            comparison_rows.append(
                {
                    "model_type": model_type,
                    "status": "skipped",
                    "warning": msg,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "r2": np.nan,
                }
            )
            failures += 1
            continue

        try:
            if model_type in baseline_types:
                # Simple fit; light search for ridge alpha
                if model_type == "ridge":
                    search = RandomizedSearchCV(
                        Ridge(random_state=seed),
                        param_distributions={"alpha": [0.1, 1.0, 10.0, 50.0]},
                        n_iter=min(4, n_iter),
                        scoring=scoring,
                        cv=min(cv_folds, max(2, len(X_train) // 5)),
                        random_state=seed,
                        n_jobs=-1,
                        refit=True,
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    search_meta = {"best_params": search.best_params_}
                else:
                    model = estimator
                    model.fit(X_train, y_train)
                    search_meta = {}
            else:
                model, search_meta = _fit_with_search(
                    model_type,
                    estimator,
                    X_train,
                    y_train,
                    seed=seed,
                    n_iter=n_iter,
                    cv_folds=cv_folds,
                    scoring=scoring,
                )

            y_hat_train = model.predict(X_train)
            y_hat_test = model.predict(X_test)
            train_metrics = regression_metrics(y_train, y_hat_train)
            test_metrics = regression_metrics(y_test, y_hat_test)

            artifact_path = _save_model_artifact(root, model_type, model, ts)
            # Also register via ModelRegistry (directory entry + index)
            model_id = f"{model_type}_{ts}"
            model_registry.register(
                model_id,
                model,
                meta={
                    "model_type": model_type,
                    "target": target,
                    "predictors": predictors,
                    "test_years": test_years,
                    "seed": seed,
                    "search": search_meta,
                    "artifact_flat_path": str(artifact_path.relative_to(root)),
                    "experiment_id": experiment_id,
                    "role": "baseline" if model_type in baseline_types else "primary",
                },
                metrics={**{f"train_{k}": v for k, v in train_metrics.items()}, **test_metrics},
                tags=["ml", model_type] + (["baseline", "H1.6"] if model_type in baseline_types else []),
                artifact_name=f"{model_type}_{ts}.joblib",
            )

            # Predictions
            pred_frame = test_df[
                [c for c in ("country_iso3", "year", "group_id") if c in test_df.columns]
            ].copy()
            pred_frame["y_true"] = y_test.values
            pred_frame["y_pred"] = y_hat_test
            pred_frame["model_type"] = model_type
            pred_frame["residual"] = pred_frame["y_true"] - pred_frame["y_pred"]
            pred_path = pred_dir / f"{model_type}_{ts}.csv"
            write_df(pred_frame, pred_path)

            row = {
                "model_type": model_type,
                "model_id": model_id,
                "status": "ok",
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "artifact_path": str(artifact_path.relative_to(root)),
                "prediction_path": str(pred_path.relative_to(root)),
                **test_metrics,
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
            comparison_rows.append(row)
            successes += 1
            tracker.log_metrics(
                experiment_id,
                {f"{model_type}_{k}": v for k, v in test_metrics.items()},
            )
            tracker.add_artifact(experiment_id, artifact_path.relative_to(root))
            tracker.add_artifact(experiment_id, pred_path.relative_to(root))
            logger.info(
                "%s test RMSE=%.4f R2=%.4f",
                model_type,
                test_metrics["rmse"],
                test_metrics["r2"],
            )
        except Exception as exc:
            logger.exception("Training failed for %s", model_type)
            failures += 1
            warnings_log.append(
                {"model_type": model_type, "status": "failed", "warning": repr(exc)}
            )
            comparison_rows.append(
                {
                    "model_type": model_type,
                    "status": "failed",
                    "warning": repr(exc),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "r2": np.nan,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    ok_df = comparison_df[comparison_df["status"] == "ok"].copy()

    if ok_df.empty or successes < 2:
        tracker.update(experiment_id, status="failed", meta={"warnings": warnings_log})
        raise FrameworkError(
            "Machine learning requires at least 2 successful models; "
            f"got {successes} (failures/skips={failures})",
            details={"warnings": warnings_log, "successes": successes, "failures": failures},
        )

    # Persist warnings
    if warnings_log:
        write_json(results_dir / "model_warnings.json", warnings_log)
        result_registry.register(
            "ml_model_warnings",
            warnings_log,
            category="machine_learning",
            fmt="json",
        )

    selection = compare(ok_df, metric="rmse", higher_is_better=False)
    paths = save_comparison(root, comparison_df, selection)
    # Ensure required path results/machine_learning/model_comparison.csv
    write_df(comparison_df, results_dir / "model_comparison.csv")
    write_df(comparison_df, tables_dir / "model_comparison.csv")
    shutil.copy2(
        results_dir / "model_comparison.csv",
        root / "outputs" / "tables" / "ml_model_comparison.csv",
    )

    result_registry.register(
        "ml_model_comparison",
        comparison_df,
        category="machine_learning",
        meta={"best_model": selection.get("best_model")},
    )
    result_registry.register(
        "ml_model_selection",
        selection,
        category="machine_learning",
        fmt="json",
    )

    # Touch/update registry.json via ModelRegistry (already done); add best pointer
    best_model = selection.get("best_model")
    best_row = selection.get("best_row") or {}
    write_json(
        root / "outputs" / "models" / "best_model.json",
        {
            "best_model": best_model,
            "model_id": best_row.get("model_id"),
            "rmse": selection.get("best_value"),
            "experiment_id": experiment_id,
            "selected_at": _timestamp(),
        },
    )

    summary = {
        "experiment_id": experiment_id,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "predictors": predictors,
        "target": target,
        "test_years": test_years,
        "successes": successes,
        "failures": failures,
        "best_model": best_model,
        "best_rmse": selection.get("best_value"),
        "comparison_path": str(paths["comparison"].relative_to(root)),
        "seed": seed,
    }
    write_json(results_dir / "training_summary.json", summary)
    result_registry.register(
        "ml_training_summary", summary, category="machine_learning", fmt="json"
    )
    tracker.update(
        experiment_id,
        status="completed",
        metrics={"best_rmse": selection.get("best_value"), "n_success": successes},
        meta={"best_model": best_model, "warnings": warnings_log},
    )
    logger.info(
        "ML training complete: best=%s RMSE=%.4f (%s successes)",
        best_model,
        float(selection.get("best_value") or np.nan),
        successes,
    )
    return summary
