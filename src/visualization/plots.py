"""Generate research figures from pipeline outputs into ``outputs/figures/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from common.io import ensure_dir, read_df, read_json, write_json
from common.logging_utils import get_logger

logger = get_logger("dmf.visualization")

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    px = go = None  # type: ignore
    HAS_PLOTLY = False


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "as_dict"):
        return config.as_dict()
    if isinstance(config, dict):
        return config
    out: Dict[str, Any] = {}
    for key in ("research", "visualization", "dashboard", "convergence"):
        if hasattr(config, key):
            out[key] = getattr(config, key) or {}
    return out


def _viz_cfg(config: Any) -> Dict[str, Any]:
    return dict(_as_mapping(config).get("visualization") or {})


def _apply_style(style: Optional[str]) -> None:
    if not style:
        return
    try:
        plt.style.use(style)
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            pass


def _load_panel(root: Path) -> pd.DataFrame:
    candidates = [
        root / "data" / "processed" / "analysis_panel.parquet",
        root / "outputs" / "data" / "dmi_panel.parquet",
        root / "data" / "processed" / "dmi_panel.parquet",
    ]
    for path in candidates:
        if path.exists():
            return read_df(path)
    raise FileNotFoundError(
        "No DMI/analysis panel found for visualization "
        "(expected data/processed/analysis_panel.parquet or dmi_panel)."
    )


def _latest_file(paths: Sequence[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _find_shap_table(root: Path) -> Optional[pd.DataFrame]:
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
    path = _latest_file(candidates)
    if path is None:
        # Any csv under results/xai or outputs/xai mentioning shap/importance
        search_dirs = [root / "results" / "xai", root / "outputs" / "xai"]
        found: List[Path] = []
        for d in search_dirs:
            if d.is_dir():
                found.extend(d.rglob("*.csv"))
        for path in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
            name = path.name.lower()
            if "shap" in name or "importance" in name:
                try:
                    return read_df(path)
                except Exception:
                    continue
        return None
    return read_df(path)


def _load_sigma(root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        root / "results" / "convergence" / "sigma_convergence.csv",
        root / "results" / "convergence" / "sigma.csv",
        root / "outputs" / "convergence" / "sigma_convergence.csv",
        root / "outputs" / "convergence" / "sigma.csv",
    ]
    path = _latest_file(candidates)
    if path is not None:
        return read_df(path)

    # Fallback: compute cross-sectional sigma from panel
    if "year" not in panel.columns or "DMI" not in panel.columns:
        return pd.DataFrame()
    g = panel.groupby("year", as_index=False)["DMI"].agg(
        sigma="std", mean="mean", n="count"
    )
    g["cv"] = g["sigma"] / g["mean"].replace(0, np.nan)
    g["source"] = "computed_from_panel"
    return g


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_plotly(fig: Any, path: Path) -> Optional[Path]:
    if not HAS_PLOTLY or fig is None:
        return None
    ensure_dir(path.parent)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def plot_dmi_trends(panel: pd.DataFrame, out_dir: Path, dpi: int) -> List[str]:
    paths: List[str] = []
    if "DMI" not in panel.columns or "year" not in panel.columns:
        return paths

    # By country
    if "country_iso3" in panel.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for iso, grp in panel.groupby("country_iso3"):
            g = grp.sort_values("year")
            ax.plot(g["year"], g["DMI"], marker="o", linewidth=1.5, label=str(iso))
        ax.set_xlabel("Year")
        ax.set_ylabel("DMI")
        ax.set_title("DMI trends by country")
        ax.legend(ncol=3, fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
        p = _save_fig(fig, out_dir / "dmi_trends_by_country.png", dpi)
        paths.append(str(p))

        if HAS_PLOTLY:
            fig_p = px.line(
                panel.sort_values(["country_iso3", "year"]),
                x="year",
                y="DMI",
                color="country_iso3",
                title="DMI trends by country",
            )
            hp = _save_plotly(fig_p, out_dir / "dmi_trends_by_country.html")
            if hp:
                paths.append(str(hp))

    # By group
    if "group_id" in panel.columns:
        gmean = (
            panel.groupby(["group_id", "year"], as_index=False)["DMI"]
            .mean()
            .sort_values(["group_id", "year"])
        )
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for gid, grp in gmean.groupby("group_id"):
            ax.plot(grp["year"], grp["DMI"], marker="o", linewidth=2, label=str(gid))
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean DMI")
        ax.set_title("DMI trends by country group")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        p = _save_fig(fig, out_dir / "dmi_trends_by_group.png", dpi)
        paths.append(str(p))

        if HAS_PLOTLY:
            fig_p = px.line(
                gmean,
                x="year",
                y="DMI",
                color="group_id",
                title="DMI trends by country group",
            )
            hp = _save_plotly(fig_p, out_dir / "dmi_trends_by_group.html")
            if hp:
                paths.append(str(hp))
    return paths


def plot_group_comparison(panel: pd.DataFrame, out_dir: Path, dpi: int) -> List[str]:
    paths: List[str] = []
    if "group_id" not in panel.columns or "DMI" not in panel.columns:
        return paths

    # Box
    groups = sorted(panel["group_id"].dropna().unique())
    data = [panel.loc[panel["group_id"] == g, "DMI"].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.boxplot(data, tick_labels=[str(g) for g in groups], patch_artist=True)
    except TypeError:
        ax.boxplot(data, labels=[str(g) for g in groups], patch_artist=True)
    ax.set_ylabel("DMI")
    ax.set_title("DMI distribution by country group")
    ax.grid(True, axis="y", alpha=0.3)
    p = _save_fig(fig, out_dir / "group_comparison_box.png", dpi)
    paths.append(str(p))

    # Bar of means
    means = panel.groupby("group_id")["DMI"].mean().reindex(groups)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(g) for g in groups], means.values, color="#2F6F8F")
    ax.set_ylabel("Mean DMI")
    ax.set_title("Mean DMI by country group")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(means.values):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    p = _save_fig(fig, out_dir / "group_comparison_bar.png", dpi)
    paths.append(str(p))

    if HAS_PLOTLY:
        fig_p = px.box(
            panel,
            x="group_id",
            y="DMI",
            title="DMI distribution by country group",
            points="outliers",
        )
        hp = _save_plotly(fig_p, out_dir / "group_comparison_box.html")
        if hp:
            paths.append(str(hp))
    return paths


def plot_correlation_heatmap(root: Path, panel: pd.DataFrame, out_dir: Path, dpi: int) -> List[str]:
    paths: List[str] = []
    corr_path = root / "results" / "descriptive" / "correlation_matrix.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
    else:
        num_cols = [
            c
            for c in panel.columns
            if c not in ("country_iso3", "year", "group_id")
            and pd.api.types.is_numeric_dtype(panel[c])
        ]
        if not num_cols:
            return paths
        corr = panel[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = _save_fig(fig, out_dir / "correlation_heatmap.png", dpi)
    paths.append(str(p))

    if HAS_PLOTLY:
        fig_p = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig_p.update_layout(title="Correlation heatmap")
        hp = _save_plotly(fig_p, out_dir / "correlation_heatmap.html")
        if hp:
            paths.append(str(hp))
    return paths


def plot_choropleth(panel: pd.DataFrame, out_dir: Path) -> List[str]:
    paths: List[str] = []
    if not HAS_PLOTLY:
        logger.warning("Plotly unavailable; skipping choropleth")
        return paths
    if "country_iso3" not in panel.columns or "DMI" not in panel.columns:
        return paths
    latest_year = int(panel["year"].max()) if "year" in panel.columns else None
    latest = panel.copy()
    if latest_year is not None:
        latest = latest[latest["year"] == latest_year]
    agg = latest.groupby("country_iso3", as_index=False)["DMI"].mean()
    fig = px.choropleth(
        agg,
        locations="country_iso3",
        color="DMI",
        hover_name="country_iso3",
        color_continuous_scale="Blues",
        title=f"Latest DMI by country (ISO3)"
        + (f" — {latest_year}" if latest_year else ""),
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    hp = _save_plotly(fig, out_dir / "choropleth_latest.html")
    if hp:
        paths.append(str(hp))
    return paths


def plot_model_comparison(root: Path, out_dir: Path, dpi: int) -> List[str]:
    paths: List[str] = []
    candidates = [
        root / "results" / "machine_learning" / "model_comparison.csv",
        root / "outputs" / "tables" / "model_comparison.csv",
        root / "outputs" / "tables" / "ml_model_comparison.csv",
    ]
    path = _latest_file(candidates)
    if path is None:
        logger.warning("model_comparison.csv not found; skipping model comparison plot")
        return paths
    df = read_df(path)
    if "model_type" not in df.columns or "rmse" not in df.columns:
        return paths
    work = df.copy()
    if "status" in work.columns:
        work = work[work["status"] == "ok"]
    work = work.dropna(subset=["rmse"]).sort_values("rmse")
    if work.empty:
        return paths

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(work["model_type"].astype(str), work["rmse"], color="#3B7A57")
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title("ML model comparison")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    p = _save_fig(fig, out_dir / "model_comparison.png", dpi)
    paths.append(str(p))

    if HAS_PLOTLY:
        fig_p = px.bar(
            work,
            x="rmse",
            y="model_type",
            orientation="h",
            title="ML model comparison (RMSE)",
        )
        hp = _save_plotly(fig_p, out_dir / "model_comparison.html")
        if hp:
            paths.append(str(hp))
    return paths


def plot_shap_importance(root: Path, out_dir: Path, dpi: int) -> List[str]:
    paths: List[str] = []
    df = _find_shap_table(root)
    if df is None or df.empty:
        logger.info("No SHAP/XAI importance table found; skipping shap plot")
        return paths

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    feature_col = None
    for key in ("feature", "variable", "term", "name", "column"):
        if key in cols:
            feature_col = cols[key]
            break
    value_col = None
    for key in (
        "mean_abs_shap",
        "importance",
        "shap_importance",
        "mean_shap",
        "value",
        "abs_mean",
    ):
        if key in cols:
            value_col = cols[key]
            break
    if feature_col is None:
        feature_col = df.columns[0]
    if value_col is None:
        numeric = [c for c in df.columns if c != feature_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return paths
        value_col = numeric[0]

    work = df[[feature_col, value_col]].dropna().copy()
    work[value_col] = work[value_col].astype(float).abs()
    work = work.sort_values(value_col, ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(work[feature_col].astype(str), work[value_col], color="#8B4513")
    ax.set_xlabel(str(value_col))
    ax.set_title("SHAP / feature importance")
    ax.grid(True, axis="x", alpha=0.3)
    p = _save_fig(fig, out_dir / "shap_importance.png", dpi)
    paths.append(str(p))

    if HAS_PLOTLY:
        fig_p = px.bar(
            work.sort_values(value_col, ascending=False),
            x=value_col,
            y=feature_col,
            orientation="h",
            title="SHAP / feature importance",
        )
        hp = _save_plotly(fig_p, out_dir / "shap_importance.html")
        if hp:
            paths.append(str(hp))
    return paths


def plot_sigma_convergence(
    root: Path, panel: pd.DataFrame, out_dir: Path, dpi: int
) -> List[str]:
    paths: List[str] = []
    sigma = _load_sigma(root, panel)
    if sigma.empty:
        return paths

    year_col = "year" if "year" in sigma.columns else sigma.columns[0]
    y_col = None
    for cand in ("sigma", "std", "dispersion", "cv"):
        if cand in sigma.columns:
            y_col = cand
            break
    if y_col is None:
        numeric = [c for c in sigma.columns if c != year_col and pd.api.types.is_numeric_dtype(sigma[c])]
        if not numeric:
            return paths
        y_col = numeric[0]

    work = sigma.sort_values(year_col)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(work[year_col], work[y_col], marker="o", linewidth=2, color="#1F4E79")
    if "cv" in work.columns and y_col != "cv":
        ax2 = ax.twinx()
        ax2.plot(work[year_col], work["cv"], marker="s", linestyle="--", color="#C45C26", label="CV")
        ax2.set_ylabel("Coefficient of variation")
        ax2.legend(loc="upper right", frameon=False)
    ax.set_xlabel("Year")
    ax.set_ylabel(str(y_col))
    ax.set_title("Sigma convergence of DMI")
    ax.grid(True, alpha=0.3)
    p = _save_fig(fig, out_dir / "convergence_sigma.png", dpi)
    paths.append(str(p))

    # Persist computed sigma if only fallback was used
    out_sigma = ensure_dir(root / "outputs" / "figures") / "sigma_series.csv"
    work.to_csv(out_sigma, index=False)

    if HAS_PLOTLY:
        fig_p = px.line(
            work,
            x=year_col,
            y=y_col,
            markers=True,
            title="Sigma convergence of DMI",
        )
        hp = _save_plotly(fig_p, out_dir / "convergence_sigma.html")
        if hp:
            paths.append(str(hp))
    return paths


def _run_process_diagrams(root: Path, config: Any) -> List[str]:
    from visualization.process_diagrams import run as run_diagrams

    result = run_diagrams(root, config)
    return list(result.get("artifacts") or [])


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """
    Build configured figures under ``outputs/figures/``.

    Uses non-interactive Agg backend. Generates Plotly HTML when plotly is installed.
    """
    root = Path(project_root).resolve()
    vcfg = _viz_cfg(config)
    out_dir = ensure_dir(root / (vcfg.get("output_dir") or "outputs/figures"))
    dpi = int(vcfg.get("dpi") or 150)
    _apply_style(vcfg.get("style"))

    figures = list(
        vcfg.get("figures")
        or [
            "dmi_trends",
            "group_comparison",
            "correlation_heatmap",
            "choropleth_latest",
            "model_comparison",
            "shap_summary",
            "convergence_sigma",
            "process_diagrams",
        ]
    )

    panel = _load_panel(root)
    produced: List[str] = []
    skipped: List[str] = []

    dispatch = {
        "dmi_trends": lambda: plot_dmi_trends(panel, out_dir, dpi),
        "group_comparison": lambda: plot_group_comparison(panel, out_dir, dpi),
        "correlation_heatmap": lambda: plot_correlation_heatmap(root, panel, out_dir, dpi),
        "choropleth_latest": lambda: plot_choropleth(panel, out_dir),
        "model_comparison": lambda: plot_model_comparison(root, out_dir, dpi),
        "shap_summary": lambda: plot_shap_importance(root, out_dir, dpi),
        "convergence_sigma": lambda: plot_sigma_convergence(root, panel, out_dir, dpi),
        "process_diagrams": lambda: _run_process_diagrams(root, config),
    }

    for name in figures:
        fn = dispatch.get(name)
        if fn is None:
            skipped.append(name)
            continue
        try:
            paths = fn()
            if paths:
                produced.extend(paths)
            else:
                skipped.append(name)
                logger.info("Figure '%s' produced no artifacts (missing inputs)", name)
        except Exception as exc:
            logger.exception("Failed to generate figure %s: %s", name, exc)
            skipped.append(name)

    manifest = {
        "output_dir": str(out_dir.relative_to(root)),
        "figures_requested": figures,
        "artifacts": [str(Path(p).relative_to(root)) if Path(p).is_absolute() else p for p in produced],
        "skipped_or_empty": skipped,
        "plotly_available": HAS_PLOTLY,
    }
    # relativize safely
    rel_artifacts: List[str] = []
    for p in produced:
        pp = Path(p)
        try:
            rel_artifacts.append(str(pp.relative_to(root)))
        except ValueError:
            rel_artifacts.append(str(pp))
    manifest["artifacts"] = rel_artifacts
    write_json(out_dir / "figures_manifest.json", manifest)
    logger.info("Wrote %d figure artifacts to %s", len(rel_artifacts), out_dir)
    return manifest
