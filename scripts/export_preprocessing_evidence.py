#!/usr/bin/env python3
"""Collect the evidence on the retrieval and preprocessing layer from the track snapshots.

Reads only the frozen and re-run tracks under versions/ and writes
outputs/preprocessing_evidence/:
  paper_numbers.json   every figure quoted in the write-up, with its source file
  tables/*.csv         the tables behind it
  figures/*.png        architecture diagram, provenance map, edition-break panel

Tracks:
  v2_mice_imputed        frozen v2 (macOS environment, August 2026)
  v2r_reproduction       v2 configuration and data re-run with the current code
  v3_edition_harmonised  edition-aware snapshots (primary)
  v3b_official_only      v3 with reconstruction switched off

Usage:
    python scripts/export_preprocessing_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validation.edition_breaks import detect_edition_breaks  # noqa: E402

VERSIONS = ROOT / "versions"
OUT = ROOT / "outputs" / "preprocessing_evidence"
INDICATORS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "C1", "C2"]
EDITION_INDICATORS = ["X3", "X5", "X6", "X7", "X10", "C2"]
STATES = ["official", "carried_forward", "mice_imputed", "missing"]
LABELS = {
    "X1": "Mobile subscriptions",
    "X2": "Fixed broadband",
    "X3": "E-Government Dev. Index",
    "X4": "Internet users",
    "X5": "Cyber Security Index",
    "X6": "Global Innovation Index",
    "X7": "AI Readiness Index",
    "X8": "R&D expenditure",
    "X9": "ICT service exports",
    "X10": "Online Service Index",
    "C1": "GDP per capita (PPP)",
    "C2": "Education index",
}

NUMBERS: Dict[str, Any] = {}


def put(key: str, value: Any, source: str) -> None:
    """Record a quoted figure together with the artifact it came from."""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    NUMBERS[key] = {"value": value, "source": source}


def track(name: str) -> Path:
    path = VERSIONS / name
    if not path.is_dir():
        raise FileNotFoundError(f"missing track snapshot: {path}")
    return path


def processed(name: str, file: str) -> pd.DataFrame:
    return pd.read_parquet(track(name) / "data_processed" / file)


def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(["country_iso3", "year"]).sort_index()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def true_provenance_v2() -> pd.DataFrame:
    """Recompute frozen-v2 labels from its own raw, carried and imputed panels (keyed by country-year)."""
    v2 = track("v2_mice_imputed")
    raw = pd.read_parquet(ROOT / "versions" / "v2r_reproduction" / "data_raw" / "integrated_raw.parquet")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    official = raw.pivot_table(index=["country_iso3", "year"], columns="indicator_id", values="value", aggfunc="last")
    pre = keyed(pd.read_parquet(v2 / "data_processed" / "panel_wide_preimputation.parquet"))[INDICATORS]
    imp = keyed(pd.read_parquet(v2 / "data_processed" / "panel_wide.parquet"))[INDICATORS]
    official = official.reindex(index=pre.index, columns=INDICATORS)
    labels = np.where(
        official.notna(), "official",
        np.where(pre.notna(), "carried_forward", np.where(imp.notna(), "mice_imputed", "missing")),
    )
    return pd.DataFrame(labels, index=pre.index, columns=INDICATORS)


def provenance_counts(labels: pd.DataFrame) -> Dict[str, int]:
    values = labels[INDICATORS].to_numpy()
    return {s: int((values == s).sum()) for s in STATES}


def provenance_section() -> pd.DataFrame:
    stored_v2 = keyed(processed("v2_mice_imputed", "cell_provenance.parquet"))[INDICATORS]
    fixed_v2 = true_provenance_v2()
    mislabelled = int((stored_v2.reindex(fixed_v2.index) != fixed_v2).to_numpy().sum())
    put("v2_stored_provenance", provenance_counts(stored_v2), "versions/v2_mice_imputed/data_processed/cell_provenance.parquet")
    put("v2_corrected_provenance", provenance_counts(fixed_v2), "recomputed from v2 raw, pre-imputation and imputed panels")
    put("v2_mislabelled_cells", mislabelled, "stored vs recomputed v2 labels")
    put("v2r_provenance", provenance_counts(processed("v2r_reproduction", "cell_provenance.parquet")),
        "versions/v2r_reproduction/data_processed/cell_provenance.parquet")

    v3 = keyed(processed("v3_edition_harmonised", "cell_provenance.parquet"))[INDICATORS]
    scope = keyed(processed("v3_edition_harmonised", "cell_reconstruction_scope.parquet"))[INDICATORS]
    put("v3_provenance", provenance_counts(v3), "versions/v3_edition_harmonised/data_processed/cell_provenance.parquet")
    scope_values = scope.to_numpy()
    put("v3_reconstruction_scope",
        {s: int((scope_values == s).sum()) for s in ("interior", "backcast", "forecast", "unobserved_series")},
        "versions/v3_edition_harmonised/data_processed/cell_reconstruction_scope.parquet")

    rows = []
    for var in INDICATORS:
        counts = v3[var].value_counts()
        rows.append(
            {
                "indicator": var,
                "label": LABELS[var],
                "official": int(counts.get("official", 0)),
                "carried_forward": int(counts.get("carried_forward", 0)),
                "reconstructed": int(counts.get("mice_imputed", 0)),
                "backcast": int((scope[var] == "backcast").sum()),
                "interior": int((scope[var] == "interior").sum()),
                "forecast": int((scope[var] == "forecast").sum()),
                "v2_official": int((fixed_v2[var] == "official").sum()),
                "v2_reconstructed": int((fixed_v2[var] == "mice_imputed").sum()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "provenance_by_indicator.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieval_section() -> pd.DataFrame:
    raw_dir = track("v3_edition_harmonised") / "data_raw"
    meta = json.loads((raw_dir / "source_metadata.json").read_text(encoding="utf-8"))
    rows = []
    for source, info in meta["sources"].items():
        validation = info.get("validation") or {}
        rows.append(
            {
                "source": source,
                "adapter_mode": (info.get("adapter_metadata") or {}).get("mode", "snapshot"),
                "rows": info.get("rows"),
                "non_null_ratio": round(float(validation.get("non_null_ratio", np.nan)), 3),
                "sha256": str(info.get("checksum_sha256", ""))[:12],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "retrieval_by_source.csv", index=False)
    put("retrieval_sources", rows, "versions/v3_edition_harmonised/data_raw/source_metadata.json")
    put("integrated_rows", meta["integrated"]["rows"], "source_metadata.json integrated.rows")

    sidecars = sorted(raw_dir.glob("*/*_value_provenance.csv"))
    prov = pd.concat([pd.read_csv(p) for p in sidecars], ignore_index=True)
    used = prov.loc[~prov["indicator_code"].str.contains("LEGACY")]
    put("snapshot_values_by_method", used.groupby("method").size().to_dict(), "data_raw/*_value_provenance.csv")
    put("snapshot_values_total", int(len(used)), "data_raw/*_value_provenance.csv")
    put("snapshot_values_cross_checked",
        int(used["verification"].str.contains("match|reproduced|equals|rank", case=False).sum()),
        "data_raw/*_value_provenance.csv verification column")
    validation = json.loads((track("v3_edition_harmonised") / "outputs" / "audit" / "validation_report.json").read_text())
    put("validation_checks", validation["checks"], "versions/v3_edition_harmonised/outputs/audit/validation_report.json")
    return table


# ---------------------------------------------------------------------------
# Edition breaks and vintages
# ---------------------------------------------------------------------------

def editions_section() -> pd.DataFrame:
    import yaml

    editions_cfg = yaml.safe_load((track("v3_edition_harmonised") / "config" / "editions.yaml").read_text(encoding="utf-8"))
    frames = []
    for name, label in (("v2r_reproduction", "v2"), ("v3_edition_harmonised", "v3")):
        official = keyed(processed(name, "panel_wide_preimputation.parquet"))
        labels = keyed(processed(name, "cell_provenance.parquet"))
        published = official.where(labels.reindex(official.index) == "official").reset_index()
        breaks = detect_edition_breaks(published, EDITION_INDICATORS, editions_cfg=editions_cfg)
        breaks.insert(0, "track", label)
        frames.append(breaks)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUT / "tables" / "edition_breaks.csv", index=False)
    for label in ("v2", "v3"):
        sub = table.loc[(table["track"] == label) & table["status"].isin(["detected", "declared", "declared_and_detected"])]
        put(f"edition_breaks_{label}", sub.drop(columns="track").to_dict("records"), "tables/edition_breaks.csv")

    # Same-method comparison for X7 (2022->2023) and the 2019->2020 step in v2.
    v2 = keyed(processed("v2r_reproduction", "panel_wide_preimputation.parquet"))["X7"].unstack()
    put("x7_v2_step_2019_2020_median", float((v2[2020] - v2[2019]).median()), "v2r pre-imputation panel")
    put("x7_v2_step_2019_2020_negative", int(((v2[2020] - v2[2019]) < 0).sum()), "v2r pre-imputation panel")
    put("x7_v2_step_2022_2023_median_abs", float((v2[2023] - v2[2022]).abs().median()), "v2r pre-imputation panel")

    # Vintage: NCSI values attributed to 2023 in v2 against the version in force on 31 Dec 2023.
    ncsi = pd.read_csv(track("v3_edition_harmonised") / "data_raw" / "ncsi" / "ncsi_panel_value_provenance.csv")
    v2_ncsi = pd.read_csv(ROOT / "versions" / "v2r_reproduction" / "data_raw" / "ncsi" / "ncsi_panel.csv")
    v2_2023 = v2_ncsi.loc[v2_ncsi["year"] == 2023].set_index("country_iso3")["NCSI_SCORE"]
    in_force = ncsi.loc[(ncsi["indicator_code"] == "NCSI_SCORE") & (ncsi["reference_year"] == 2023)].set_index("country_iso3")["value"]
    put("ncsi_v2_2023_vs_in_force", {c: [float(v2_2023[c]), float(in_force[c])] for c in in_force.index},
        "v2 ncsi_panel.csv vs NCSI 3.0 version history")
    put("ncsi_countries_ranked_by_end_2023", int(len(in_force)), "ncsi_panel_value_provenance.csv")

    # HDR vintage revisions: v2 mixed-vintage values vs the single HDR 2025 series.
    old = pd.read_csv(ROOT / "versions" / "v2r_reproduction" / "data_raw" / "undp" / "education_index_panel.csv")
    new = pd.read_csv(track("v3_edition_harmonised") / "data_raw" / "undp" / "education_index_panel.csv")
    merged = old.merge(new, on=["country_iso3", "year"], suffixes=("_v2", "_v3")).dropna()
    diff = (merged["EDU_INDEX_v3"] - merged["EDU_INDEX_v2"]).abs()
    put("hdr_revision_mean_abs", float(diff.mean()), "v2 vs v3 education_index_panel.csv")
    put("hdr_revision_max_abs", float(diff.max()), "v2 vs v3 education_index_panel.csv")
    return table


# ---------------------------------------------------------------------------
# Index and downstream effects
# ---------------------------------------------------------------------------

def dmi_section() -> pd.DataFrame:
    def dmi(name: str) -> pd.Series:
        return keyed(processed(name, "dmi_panel.parquet"))["DMI"]

    pairs = {
        "v2_vs_v2r": ("v2_mice_imputed", "v2r_reproduction"),
        "v2r_vs_v3": ("v2r_reproduction", "v3_edition_harmonised"),
        "v3_vs_v3b": ("v3_edition_harmonised", "v3b_official_only"),
    }
    rows = []
    for key, (a, b) in pairs.items():
        left, right = dmi(a), dmi(b)
        both = pd.concat([left, right], axis=1, keys=["a", "b"]).dropna()
        diff = (both["b"] - both["a"]).abs()
        record = {
            "pair": key,
            "n": int(len(both)),
            "pearson_r": float(both["a"].corr(both["b"])),
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
        }
        rows.append(record)
        put(f"dmi_{key}", record, f"versions/{a} and versions/{b} dmi_panel.parquet")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "dmi_agreement.csv", index=False)

    v3 = keyed(processed("v3_edition_harmonised", "dmi_panel.parquet"))
    put("v3_dmi_summary", {k: float(v) for k, v in v3["DMI"].describe().items()}, "v3 dmi_panel.parquet")
    groups = v3.reset_index().groupby("group_id")["DMI"].mean()
    put("v3_dmi_group_means", {k: float(v) for k, v in groups.items()}, "v3 dmi_panel.parquet")

    reliability = json.loads((track("v3_edition_harmonised") / "outputs" / "tables" / "dmi_reliability.json").read_text())
    put("v3_cronbach_alpha", {"indicators": reliability.get("cronbach_alpha_indicators"),
                              "pillars": reliability.get("cronbach_alpha_pillars")}, "v3 dmi_reliability.json")
    sens = pd.read_csv(track("v3_edition_harmonised") / "outputs" / "tables" / "dmi_sensitivity.csv")
    put("v3_dmi_pca_r", float(sens["DMI"].corr(sens["DMI_pca"])), "v3 dmi_sensitivity.csv")
    put("v3_weight_shock_mean_sd", float(sens["DMI_shock_std"].mean()), "v3 dmi_sensitivity.csv")
    vif = pd.read_csv(track("v3_edition_harmonised") / "results" / "descriptive" / "vif_predictors.csv")
    put("v3_vif_range", [float(vif["vif"].min()), float(vif["vif"].max())], "v3 vif_predictors.csv")
    put("v3_vif_above_10", int((vif["vif"] > 10).sum()), "v3 vif_predictors.csv")

    coef = {}
    for name in ("v2r_reproduction", "v3_edition_harmonised"):
        c = pd.read_csv(track(name) / "results" / "econometrics" / "coefficients.csv")
        c = c.loc[c["model"] == "FE"].set_index("term")
        coef[name] = c[["coefficient", "pvalue"]]
    fe = coef["v2r_reproduction"].join(coef["v3_edition_harmonised"], lsuffix="_v2r", rsuffix="_v3")
    fe.to_csv(OUT / "tables" / "fe_coefficients_v2r_v3.csv")
    put("fe_v2r_v3", json.loads(fe.to_json(orient="index")), "results/econometrics/coefficients.csv (FE)")
    return table


def benchmark_section() -> pd.DataFrame:
    frames = []
    for name in ("v2r_reproduction", "v3_edition_harmonised"):
        base = track(name) / "results" / "imputation"
        for pool, file in (("all_non_missing", "imputation_validation_raw.csv"),
                           ("official_only", "imputation_validation_official_raw.csv")):
            path = base / file
            if not path.exists():
                continue
            raw = pd.read_csv(path)
            agg = raw.groupby("configuration").agg(nrmse=("nrmse_sd", "mean"), n_masked=("n_masked", "sum"))
            agg["n_masked"] = agg["n_masked"] / raw["repeat"].nunique()
            agg = agg.reset_index()
            agg.insert(0, "mask_pool", pool)
            agg.insert(0, "track", name)
            frames.append(agg)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUT / "tables" / "masked_cell_benchmark.csv", index=False)
    put("masked_cell_benchmark", table.to_dict("records"), "results/imputation/imputation_validation*_raw.csv")
    return table


def protocol_section() -> pd.DataFrame:
    rows = []
    point = pd.read_csv(track("v3_edition_harmonised") / "outputs" / "tables" / "ml_model_comparison.csv")
    safe = pd.read_csv(track("v3_edition_harmonised") / "results" / "imputation" / "leakage_safe_ml.csv")
    verdict = json.loads((track("v3_edition_harmonised") / "results" / "imputation" / "leakage_safe_ml_verdict.json").read_text())
    official = pd.read_csv(track("v3b_official_only") / "outputs" / "tables" / "ml_model_comparison.csv")
    for model in ["linear_regression", "ridge", "random_forest", "xgboost", "lightgbm", "catboost", "svr", "mlp"]:
        rows.append(
            {
                "model": model,
                "imputation_before_split": _rmse(point, model),
                "leakage_safe": _rmse(safe, model),
                "official_only": _rmse(official, model),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "protocol_comparison.csv", index=False)
    put("protocol_rmse", table.to_dict("records"), "v3 ml_model_comparison.csv, leakage_safe_ml.csv; v3b ml_model_comparison.csv")
    put("protocol_n_train", {"imputation_before_split": int(point["n_train"].iloc[0]),
                             "leakage_safe": int(verdict.get("n_train", 0)),
                             "official_only": int(official["n_train"].iloc[0])}, "same files")
    put("protocol_test_years", verdict.get("test_years"), "leakage_safe_ml_verdict.json")
    return table


def _rmse(frame: pd.DataFrame, model: str):
    col = "model_type"
    hit = frame.loc[frame[col] == model]
    return None if hit.empty else float(hit["rmse"].iloc[0])


# ---------------------------------------------------------------------------
# Figures (print-first: 300 dpi, one-hue ramp plus hatching, two categorical hues)
# ---------------------------------------------------------------------------

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def figure_architecture() -> Path:
    plt = _style()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    stages = [
        ("Acquire", "WDI API v2 (cached vintage)\nsnapshot adapters\n7-column long schema"),
        ("Validate", "ISO-3 allow-list, years,\nduplicate keys, non-null\nedition-break check"),
        ("Harmonise", "edition families\n(exclude other frameworks)\nannual 12×14 grid"),
        ("Fill gaps", "carry ≤ 2 years\nMICE (RF) or none\nPMM draws, m = 10"),
        ("Compute", "t−1 lags, growth\nDMI: min–max, 5 pillars\nreliability, VIF"),
        ("Publish", "Streamlit package\nPower BI star schema\nRDF / SPARQL, DOI"),
    ]
    width, gap, top = 14.2, 2.8, 20.0
    for i, (name, body) in enumerate(stages):
        x = 1.0 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, top), width, 15.5, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    linewidth=0.8, edgecolor=BLUE, facecolor="#eef4fc"))
        ax.text(x + width / 2, top + 13.0, name, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + width / 2, top + 5.9, body, ha="center", va="center", fontsize=6.1, color=INK, linespacing=1.25)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.35, top + 7.7), (x + width + gap - 0.35, top + 7.7),
                                         arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color=MUTED))
    rail_y = 3.0
    ax.add_patch(FancyBboxPatch((1.0, rail_y), 98.0, 11.0, boxstyle="round,pad=0.3,rounding_size=1.2",
                                linewidth=0.8, edgecolor=ORANGE, facecolor="#fdf1ea", linestyle=(0, (3, 2))))
    ax.text(50.0, rail_y + 8.2, "Provenance rail (written by every stage)", ha="center", va="center",
            fontsize=7, fontweight="bold", color=INK)
    ax.text(50.0, rail_y + 3.4,
            "per-value source record (edition, vintage, URL, SHA-256)  ·  validation and edition-break reports  ·  "
            "cell state official / carried / reconstructed (interior, backcast, forecast)  ·  lineage graph  ·  config snapshot",
            ha="center", va="center", fontsize=5.9, color=INK)
    for i in range(len(stages)):
        x = 1.0 + i * (width + gap) + width / 2
        ax.add_patch(FancyArrowPatch((x, top - 0.3), (x, rail_y + 11.3), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=0.7, color=ORANGE, linestyle=(0, (2, 1.5))))
    path = OUT / "figures" / "fig1_pipeline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_provenance_and_breaks() -> Path:
    plt = _style()
    import matplotlib.colors as mcolors

    v3 = keyed(processed("v3_edition_harmonised", "cell_provenance.parquet"))[INDICATORS]
    scope = keyed(processed("v3_edition_harmonised", "cell_reconstruction_scope.parquet"))[INDICATORS]
    years = sorted(v3.index.get_level_values("year").unique())
    order = ["X1", "X2", "X4", "X8", "X9", "C1", "X3", "X10", "X6", "X7", "X5", "C2"]
    official = np.array([[int((v3.xs(y, level="year")[v] == "official").sum()) for y in years] for v in order])
    backcast = np.array([[int((scope.xs(y, level="year")[v] == "backcast").sum()) for y in years] for v in order])
    rebuilt = np.array([[int((v3.xs(y, level="year")[v] == "mice_imputed").sum()) for y in years] for v in order])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.75), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.38})
    ramp = mcolors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"])
    im = ax1.imshow(official, cmap=ramp, vmin=0, vmax=12, aspect="auto")
    for i in range(len(order)):
        for j in range(len(years)):
            if rebuilt[i, j]:
                hatch = "////" if backcast[i, j] else "...."
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch=hatch,
                                            edgecolor=ORANGE, linewidth=0))
            ax1.text(j, i, str(official[i, j]), ha="center", va="center", fontsize=4.6,
                     color="white" if official[i, j] >= 8 else INK)
    ax1.set_xticks(range(len(years)))
    ax1.set_xticklabels([str(y)[2:] for y in years], fontsize=6)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([f"{v}  {LABELS[v]}" for v in order], fontsize=5.8)
    ax1.set_xlabel("Year (20..)")
    ax1.set_title("(a) Published values per indicator-year (of 12 countries)", fontsize=7, loc="left", color=INK)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_visible(False)

    v2 = keyed(processed("v2r_reproduction", "panel_wide.parquet"))["X7"].unstack()
    v3x = keyed(processed("v3_edition_harmonised", "panel_wide.parquet"))["X7"].unstack()
    for frame, color, style, label in ((v2, ORANGE, (0, (3, 1.5)), "v2: 2019 edition rescaled ×10"),
                                       (v3x, BLUE, "solid", "v3: 2020–2024 family only")):
        med = frame.median(axis=0)
        ax2.plot(med.index, med.values, color=color, linestyle=style, linewidth=1.6, label=label)
        ax2.fill_between(med.index, frame.quantile(0.25, axis=0), frame.quantile(0.75, axis=0),
                         color=color, alpha=0.12, linewidth=0)
    ax2.axvspan(2019.5, 2024.5, color=GRID, alpha=0.35, linewidth=0)
    ax2.text(2022.0, ax2.get_ylim()[0] + 1.0, "published\n2020–2024", ha="center", va="bottom", fontsize=5.5, color=MUTED)
    ax2.set_xlim(2012, 2025)
    ax2.set_xticks([2012, 2015, 2018, 2021, 2024])
    ax2.set_ylabel("AI Readiness (X7), median and IQR")
    ax2.set_title("(b) X7 series before and after edition filtering", fontsize=7, loc="left", color=INK)
    ax2.grid(axis="y", color=GRID, linewidth=0.5)
    ax2.legend(frameon=False, fontsize=5.8, loc="upper right")
    path = OUT / "figures" / "fig2_provenance_breaks.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    provenance_section()
    retrieval_section()
    editions_section()
    dmi_section()
    benchmark_section()
    protocol_section()
    put("figure_1", str(figure_architecture().relative_to(ROOT)), "generated")
    put("figure_2", str(figure_provenance_and_breaks().relative_to(ROOT)), "generated")
    (OUT / "paper_numbers.json").write_text(json.dumps(NUMBERS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(NUMBERS)} quoted figures to {OUT / 'paper_numbers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
