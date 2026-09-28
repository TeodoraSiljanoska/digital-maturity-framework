"""Model comparison and selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from common.io import ensure_dir, write_df, write_json


def compare(
    metrics: Union[pd.DataFrame, list],
    *,
    metric: str = "rmse",
    higher_is_better: bool = False,
    min_models: int = 1,
) -> Dict[str, Any]:
    """
    Select the best model by a metric column (default: lowest RMSE).

    Parameters
    ----------
    metrics:
        DataFrame or list of dicts with at least ``model_type`` and ``metric``.
    metric:
        Column name used for ranking.
    higher_is_better:
        If True, maximize the metric (e.g. R2); otherwise minimize.
    """
    df = pd.DataFrame(metrics) if not isinstance(metrics, pd.DataFrame) else metrics.copy()
    if df.empty:
        raise ValueError("No model metrics available for comparison")
    if metric not in df.columns:
        raise KeyError(f"Metric column '{metric}' not found in comparison table")

    work = df.dropna(subset=[metric]).copy()
    if len(work) < min_models:
        raise ValueError(
            f"Need at least {min_models} models with valid {metric}; got {len(work)}"
        )

    if higher_is_better:
        best_idx = work[metric].astype(float).idxmax()
    else:
        best_idx = work[metric].astype(float).idxmin()
    best_row = work.loc[best_idx]
    ranking = work.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)

    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "best_model": str(best_row.get("model_type", best_row.get("model_id", ""))),
        "best_value": float(best_row[metric]),
        "best_row": best_row.to_dict(),
        "ranking": ranking.to_dict(orient="records"),
        "n_models": int(len(work)),
    }


def save_comparison(
    project_root: Path | str,
    comparison_df: pd.DataFrame,
    selection: Dict[str, Any],
    *,
    results_subdir: str = "machine_learning",
) -> Dict[str, Path]:
    """Persist model comparison table and selection JSON."""
    root = Path(project_root).resolve()
    results_dir = ensure_dir(root / "results" / results_subdir)
    tables_dir = ensure_dir(root / "outputs" / "tables")
    models_dir = ensure_dir(root / "outputs" / "models")

    cmp_path = results_dir / "model_comparison.csv"
    write_df(comparison_df, cmp_path)
    write_df(comparison_df, tables_dir / "model_comparison.csv")

    sel_path = results_dir / "model_selection.json"
    write_json(sel_path, selection)

    # Lightweight registry pointer update (full registry managed by ModelRegistry)
    registry_ptr = {
        "best_model": selection.get("best_model"),
        "metric": selection.get("metric"),
        "best_value": selection.get("best_value"),
        "comparison_path": str(cmp_path.relative_to(root)),
        "selection_path": str(sel_path.relative_to(root)),
    }
    ptr_path = models_dir / "selection.json"
    write_json(ptr_path, registry_ptr)

    return {
        "comparison": cmp_path,
        "selection": sel_path,
        "selection_pointer": ptr_path,
    }


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Optional stage entrypoint: re-read comparison CSV if present and re-select.

    Primary selection happens inside ``machine_learning.train.run``.
    """
    root = Path(project_root).resolve()
    cmp_path = root / "results" / "machine_learning" / "model_comparison.csv"
    if not cmp_path.exists():
        return {"status": "skipped", "reason": "model_comparison.csv not found"}
    df = pd.read_csv(cmp_path)
    ml_cfg = {}
    if config is not None:
        ml_cfg = getattr(config, "machine_learning", None) or (
            config.get("machine_learning", {}) if isinstance(config, dict) else {}
        )
    metric = "rmse"
    metrics_list = (ml_cfg or {}).get("metrics") or ["rmse"]
    if metrics_list:
        metric = str(metrics_list[0]).lower()
    selection = compare(df, metric=metric, higher_is_better=(metric == "r2"))
    paths = save_comparison(root, df, selection)
    return {"selection": selection, "paths": {k: str(v) for k, v in paths.items()}}


__all__ = ["compare", "save_comparison", "run"]
