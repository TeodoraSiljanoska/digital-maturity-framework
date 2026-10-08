"""Time-series view of the panel: series coverage, publication lag and a
rolling-origin (expanding-window) forecast evaluation of the DMI.

The rolling-origin protocol is leakage-safe by construction. For origin T the
models see only official values of years <= T; gaps are carried inside that
window only (at most two years, as in the main pipeline), the remaining
feature gaps are filled with training medians, and no imputer is fitted. The
target year T + 1 is forecast from predictors of year T. A persistence
forecast (DMI of year T) is the reference every model has to beat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common.io import ensure_dir, write_df, write_json
from common.logging_utils import get_logger
from knowledge_layer.context import series_table

logger = get_logger("dmf.knowledge.timeseries")

CORE = [f"X{i}" for i in range(1, 11)] + ["C1", "C2"]
ORIGINS = list(range(2017, 2025))  # last training year T; forecast year T + 1


def official_values(root: Path) -> pd.DataFrame:
    """Values of the processed panel kept only where the cell is published."""
    vals = pd.read_parquet(root / "data/processed/panel_wide_imputed.parquet")
    prov = pd.read_parquet(root / "data/processed/cell_provenance.parquet")
    vals = vals.set_index(["country_iso3", "year"]).sort_index()
    prov = prov.set_index(["country_iso3", "year"]).reindex(vals.index)
    out = vals[CORE].where(prov[CORE] == "official")
    return out.reset_index()


def coverage_by_year(prov: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    long = prov.melt(id_vars=["country_iso3", "year"], value_vars=ids, var_name="indicator_id", value_name="state")
    tab = long.groupby(["year", "state"]).size().unstack(fill_value=0)
    tab["official_share"] = tab.get("official", 0) / tab.sum(axis=1)
    return tab.reset_index()


def _models(seed: int) -> Dict[str, Callable[[], Any]]:
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    return {
        "ols": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "svr": lambda: make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1)),
        "random_forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "xgboost": lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=seed, n_jobs=2),
        "catboost": lambda: CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, random_seed=seed, verbose=0, thread_count=2),
    }


def _design(official: pd.DataFrame, target: pd.Series, origin: int, k: int = 2) -> pd.DataFrame:
    """Rows (country, t) with predictors of year t-1 built from data of years <= origin."""
    win = official[official["year"] <= origin].sort_values(["country_iso3", "year"]).copy()
    for c in CORE:
        win[c] = win.groupby("country_iso3")[c].transform(lambda s: s.ffill(limit=k).bfill(limit=k))
    feats = win.rename(columns={c: f"{c}_lag1" for c in CORE})
    feats["year"] = feats["year"] + 1
    df = feats.merge(target.rename("y").reset_index(), on=["country_iso3", "year"], how="inner")
    lagged_y = target.rename("y_lag1").reset_index()
    lagged_y["year"] = lagged_y["year"] + 1
    df = df.merge(lagged_y, on=["country_iso3", "year"], how="left")
    return df


def rolling_origin(root: Path, target_name: str, target: pd.Series, seed: int = 42) -> pd.DataFrame:
    official = official_values(root)
    feat_cols = [f"{c}_lag1" for c in CORE]
    preds: List[pd.DataFrame] = []
    models = _models(seed)
    for T in ORIGINS:
        df = _design(official, target, T)
        train = df[df["year"] <= T].copy()
        test = df[df["year"] == T + 1].copy()
        if test.empty:
            continue
        med = train[feat_cols].median()
        Xtr = train[feat_cols].fillna(med)
        Xte = test[feat_cols].fillna(med)
        # columns with no published value in the window carry no information
        keep = [c for c in feat_cols if Xtr[c].notna().all()]
        Xtr, Xte = Xtr[keep], Xte[keep]
        out = test[["country_iso3", "year", "y", "y_lag1"]].copy()
        out["origin"] = T
        out["n_train"] = len(train)
        out["pred_persistence"] = test["y_lag1"].values
        for name, make in models.items():
            m = make()
            m.fit(Xtr.values, train["y"].values)
            out[f"pred_{name}"] = m.predict(Xte.values)
            # ARX variant: the same learner with the target's own lag as an extra input
            if name in ("ridge", "catboost"):
                m2 = make()
                Xtr2 = np.column_stack([Xtr.values, train["y_lag1"].values])
                Xte2 = np.column_stack([Xte.values, test["y_lag1"].values])
                ok = ~np.isnan(Xtr2).any(axis=1)
                m2.fit(Xtr2[ok], train["y"].values[ok])
                out[f"pred_{name}_arx"] = m2.predict(Xte2)
        preds.append(out)
        logger.info("rolling origin %s T=%d: train=%d test=%d", target_name, T, len(train), len(test))
    res = pd.concat(preds, ignore_index=True)
    res["target"] = target_name
    return res


def score(preds: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = [c[len("pred_"):] for c in preds.columns if c.startswith("pred_")]
    countries = preds["country_iso3"].unique()
    rows = []
    for tgt, g in preds.groupby("target"):
        err = {m: (g[f"pred_{m}"] - g["y"]).to_numpy() for m in models}
        base = err["persistence"]
        idx_by_c = {c: np.where(g["country_iso3"].to_numpy() == c)[0] for c in countries}
        for m in models:
            e = err[m]
            rmse = float(np.sqrt(np.mean(e ** 2)))
            mae = float(np.mean(np.abs(e)))
            per_origin = g.assign(e=e).groupby("origin")["e"].apply(lambda s: np.sqrt(np.mean(s ** 2)))
            base_origin = g.assign(e=base).groupby("origin")["e"].apply(lambda s: np.sqrt(np.mean(s ** 2)))
            wins = int((per_origin < base_origin).sum())
            # country-cluster bootstrap of RMSE(model) - RMSE(persistence)
            diffs = []
            for _ in range(n_boot):
                pick = rng.choice(countries, size=len(countries), replace=True)
                ii = np.concatenate([idx_by_c[c] for c in pick])
                diffs.append(np.sqrt(np.mean(e[ii] ** 2)) - np.sqrt(np.mean(base[ii] ** 2)))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append(
                {
                    "target": tgt,
                    "model": m,
                    "n_forecasts": int(len(e)),
                    "rmse": rmse,
                    "mae": mae,
                    "delta_rmse_vs_persistence": rmse - float(np.sqrt(np.mean(base ** 2))),
                    "delta_ci_low": float(lo),
                    "delta_ci_high": float(hi),
                    "origins_beating_persistence": wins,
                    "n_origins": int(per_origin.size),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "rmse"])


def run(project_root: Path | str) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    out = ensure_dir(root / "outputs" / "knowledge_layer" / "timeseries")
    prov = pd.read_parquet(root / "data/processed/cell_provenance.parquet")
    vals = pd.read_parquet(root / "data/processed/panel_wide_imputed.parquet")
    series = series_table(prov, vals, CORE, vintage_year=2025)
    write_df(series, out / "core_series.csv")
    cov = coverage_by_year(prov, CORE)
    write_df(cov, out / "core_coverage_by_year.csv")

    dmi_v3 = pd.read_parquet(root / "data/processed/dmi_panel.parquet").set_index(["country_iso3", "year"])["DMI"]
    dmi_v3b = pd.read_parquet(root / "versions/v3b_official_only/data_processed/dmi_panel.parquet").set_index(["country_iso3", "year"])["DMI"]
    preds = pd.concat(
        [rolling_origin(root, "DMI_mice", dmi_v3), rolling_origin(root, "DMI_official_only", dmi_v3b)],
        ignore_index=True,
    )
    write_df(preds, out / "rolling_origin_predictions.csv")
    scores = score(preds)
    write_df(scores, out / "rolling_origin_scores.csv")
    write_json(out / "rolling_origin_protocol.json", {
        "origins": ORIGINS,
        "forecast_years": [t + 1 for t in ORIGINS],
        "features": [f"{c}_lag1" for c in CORE],
        "carry_max_years": 2,
        "feature_gap_fill": "training-set median",
        "imputer": "none",
        "bootstrap": "country-cluster, 2000 draws",
    })
    return {"n_series": int(len(series)), "scores": scores.to_dict(orient="records")}
