"""
Streamlit VDA dashboard: by country, indicator, phase, provenance, with year slider.

Run from project root:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "outputs" / "dashboard"
INDICATORS = [f"X{i}" for i in range(1, 11)] + ["C1", "C2", "DMI"]


def _read_df(name: str) -> Optional[pd.DataFrame]:
    for suffix in (".parquet", ".csv"):
        path = DATA_DIR / f"{name}{suffix}"
        if path.exists():
            return pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
    # also try exact filename
    path = DATA_DIR / name
    if path.exists() and path.suffix == ".csv":
        return pd.read_csv(path)
    return None


def _read_json(name: str) -> Optional[Dict[str, Any]]:
    path = DATA_DIR / name
    if not path.exists():
        alt = DATA_DIR / f"{name}.json" if not name.endswith(".json") else path
        path = alt
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_data() -> bool:
    if not DATA_DIR.is_dir():
        st.error(f"Dashboard data directory missing: `{DATA_DIR}`.")
        return False
    panel = _read_df("panel")
    if panel is None or panel.empty:
        st.error("No panel data in `outputs/dashboard/`. Run dashboard.export first.")
        return False
    return True


def _year_slider(panel: pd.DataFrame) -> int:
    years = sorted(int(y) for y in panel["year"].dropna().unique())
    if len(years) == 1:
        return years[0]
    return int(st.sidebar.slider("Year", min_value=years[0], max_value=years[-1], value=years[-1]))


def _group_filter(panel: pd.DataFrame) -> pd.DataFrame:
    """Sidebar group filter. Default = all groups (same view as before the control existed)."""
    if panel is None or panel.empty or "group_id" not in panel.columns:
        return panel
    groups = sorted(str(g) for g in panel["group_id"].dropna().unique())
    selected = st.sidebar.multiselect("Country groups", groups, default=groups)
    if not selected:
        st.sidebar.caption("No group selected — showing all countries.")
        return panel
    return panel[panel["group_id"].astype(str).isin(selected)].copy()


def page_overview(panel: pd.DataFrame, year: int) -> None:
    st.header("Overview")
    st.caption("DMI is the dependent variable Y. C1 and C2 are controls. X1–X10 are independents.")
    latest = panel[panel["year"] == year]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", int(panel["country_iso3"].nunique()))
    c2.metric("Year (slider)", year)
    c3.metric("Panel rows", len(panel))
    mean_dmi = float(latest["DMI"].mean()) if "DMI" in latest.columns else float("nan")
    c4.metric(f"Mean DMI ({year})", f"{mean_dmi:.2f}" if pd.notna(mean_dmi) else "—")
    if "DMI" in panel.columns:
        st.subheader("DMI trends by country")
        pivot = panel.pivot_table(index="year", columns="country_iso3", values="DMI", aggfunc="mean")
        st.line_chart(pivot)
        st.caption(
            "How to read: each line is one country. The year slider filters tables below, not this trend."
        )
    st.subheader(f"DMI by country, {year}")
    show = latest[["country_iso3", "group_id", "DMI"]].sort_values("DMI", ascending=False) if "group_id" in latest.columns else latest
    st.dataframe(show, use_container_width=True)
    if "country_iso3" in show.columns and "DMI" in show.columns:
        st.bar_chart(show.set_index("country_iso3")["DMI"])


def page_country(panel: pd.DataFrame, year: int) -> None:
    st.header("By country")
    countries = sorted(panel["country_iso3"].unique())
    iso = st.selectbox("Country", countries)
    sub = panel[panel["country_iso3"] == iso].sort_values("year")
    row = sub[sub["year"] == year]
    st.subheader(f"{iso} — DMI trajectory")
    if "DMI" in sub.columns:
        st.line_chart(sub.set_index("year")[["DMI"]])
        st.caption("How to read: constructed composite 0–100. The table below lists X1–X10, C1, C2 and DMI for the selected year.")
    choropleth = PROJECT_ROOT / "outputs" / "figures" / "choropleth_latest.html"
    dash_choro = DATA_DIR / "choropleth_latest.html"
    html_path = dash_choro if dash_choro.exists() else choropleth
    if html_path.exists():
        st.subheader("Latest DMI map (frozen figure)")
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=420, scrolling=True)
        st.caption("How to read: choropleth of latest DMI in the panel. Generated from the frozen panel, not a new estimate.")
    cols = [c for c in INDICATORS if c in sub.columns]
    st.subheader(f"Indicator values, {iso}, {year}")
    if not row.empty:
        st.dataframe(row[cols].T.rename(columns={row.index[0]: "value"}), use_container_width=True)
    st.subheader("Full series")
    st.dataframe(sub[["year"] + cols], use_container_width=True)


def page_indicator(panel: pd.DataFrame, year: int) -> None:
    st.header("By indicator")
    catalog = _read_df("indicator_catalog")
    if catalog is not None and not catalog.empty:
        st.subheader("Reusable indicator catalog (URI, method, phase, source)")
        st.dataframe(catalog, use_container_width=True)
        st.caption(
            "Each indicator is a semantic object with a project IRI "
            "(http://purl.org/dmi-framework#Ind_*, not a registered PURL), "
            "collection method, and process phases."
        )
    cols = [c for c in INDICATORS if c in panel.columns]
    iid = st.selectbox("Indicator", cols)
    slice_ = panel[panel["year"] == year][["country_iso3", "group_id", iid] if "group_id" in panel.columns else ["country_iso3", iid]]
    st.subheader(f"{iid} by country, {year}")
    st.bar_chart(slice_.set_index("country_iso3")[iid])
    st.dataframe(slice_.sort_values(iid, ascending=False), use_container_width=True)
    ts = panel.pivot_table(index="year", columns="country_iso3", values=iid, aggfunc="mean")
    st.line_chart(ts)
    st.caption(f"How to read: {iid} in native units. Compare countries at the selected year, then inspect the trajectory.")


def page_obtaining(panel: pd.DataFrame) -> None:
    st.header("By obtaining method")
    st.caption(
        "Reusable catalog sliced by CollectionMethod. "
        "official_api and curated_snapshot are used; constructed is DMI; "
        "primary_survey and sensor_automated are unused. "
        "ScoreEntry bars respect the country-group filter."
    )
    catalog = _read_df("indicator_catalog")
    ont_scores = PROJECT_ROOT / "outputs" / "ontology" / "score_entries.parquet"
    scores = pd.read_parquet(ont_scores) if ont_scores.exists() else None
    if scores is not None and not scores.empty and "country_iso3" in scores.columns:
        keep = set(panel["country_iso3"].astype(str)) if "country_iso3" in panel.columns else set()
        if keep:
            scores = scores[scores["country_iso3"].astype(str).isin(keep)]
    if catalog is not None and not catalog.empty:
        method_col = "collection_method" if "collection_method" in catalog.columns else None
        if method_col:
            methods = sorted(catalog[method_col].dropna().unique())
            chosen = st.selectbox("Collection method", methods)
            st.dataframe(catalog[catalog[method_col] == chosen], use_container_width=True)
        else:
            st.dataframe(catalog, use_container_width=True)
    if scores is not None and not scores.empty and "collection_method" in scores.columns:
        counts = (
            scores.groupby("collection_method")
            .size()
            .rename("n_cells")
            .reset_index()
        )
        st.subheader("Cells by obtaining method")
        st.dataframe(counts, use_container_width=True)
        st.bar_chart(counts.set_index("collection_method")["n_cells"])
        st.caption("How to read: each bar is one CollectionMethod. Same ScoreEntry cube as the ontology.")


def _process_captions() -> Dict[str, str]:
    path = PROJECT_ROOT / "outputs" / "figures" / "processes" / "captions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def page_processes() -> None:
    st.header("Processes")
    st.caption(
        "Drawn and partitioned from code (`src/visualization/process_diagrams.py`), not hand slides. "
        "Each box is a stage or CollectionMethod. How to read is under the figure."
    )
    proc = PROJECT_ROOT / "outputs" / "figures" / "processes"
    captions = _process_captions()
    order = [
        ("acquisition_methods.png", "Obtaining methods"),
        ("missing_data_workflow.png", "Missing-data workflow"),
        ("model_to_phase.png", "Methods mapped to phases"),
        ("construction_p1_p8.png", "Ontology construction P1–P8"),
        ("ontology_layers.png", "Ontology layers Ctax / Cdec / Ceval"),
        ("assessment_pipeline.png", "Assessment process (12 PipelineStage boxes; 8 OWL ProcessPhase)"),
        ("correlation_vif_process.png", "How r and VIF are obtained"),
    ]
    missing = [name for name, _title in order if not (proc / name).exists()]
    if missing:
        st.warning(
            "Process PNGs missing (gitignored). Regenerate: "
            "`PYTHONPATH=src python -c 'from visualization.process_diagrams import run; from pathlib import Path; run(Path(\".\"))'`"
        )
    for name, title in order:
        path = proc / name
        st.subheader(title)
        if path.exists():
            st.image(str(path), use_container_width=True)
            cap = captions.get(name, "")
            if cap:
                st.caption(f"How to read: {cap}")
        else:
            st.info(f"Missing `{name}`.")


def page_phase() -> None:
    st.header("By process phase")
    pipeline = PROJECT_ROOT / "outputs" / "figures" / "processes" / "assessment_pipeline.png"
    if pipeline.exists():
        st.image(str(pipeline), use_container_width=True)
        st.caption(
            "How to read: each box is a PipelineStage with a concrete I/O artifact. "
            "The Processes view shows the other drawings, including ontology layers."
        )
    st.caption("ML models are not collection methods. CatBoost belongs to Predict.")
    from ontology.query import MODEL_PHASE_MAP

    phase_rows = [
        {"method": k, "phase": v["phase"], "purpose": v["purpose"]}
        for k, v in MODEL_PHASE_MAP.items()
    ]
    st.subheader("Method → phase (single source: ontology.query.MODEL_PHASE_MAP)")
    st.dataframe(pd.DataFrame(phase_rows), use_container_width=True)
    coef_path = DATA_DIR / "coefficients.csv"
    if coef_path.exists():
        st.subheader("Analyse phase — FE/RE coefficients")
        st.dataframe(pd.read_csv(coef_path), use_container_width=True)
        st.caption("How to read: one-year lagged predictors. Stars are in the ver02 tables, not here. Clustered SE by country.")
    vif_path = DATA_DIR / "vif_predictors.csv"
    if vif_path.exists():
        st.subheader("Analyse phase — VIF (with intercept)")
        st.dataframe(pd.read_csv(vif_path), use_container_width=True)
        st.caption("How to read: VIF > 10 flags collinearity among lagged predictors. It is not a DMI score.")
    ont = _read_json("ontology_graph.json")
    if ont:
        st.subheader("Ontology DecisionContext")
        st.json({"namespace": ont.get("namespace"), "n_indicators": len(ont.get("indicators") or []), "scenarios": ont.get("scenarios")})


def page_provenance(year: int, panel: pd.DataFrame) -> None:
    st.header("Provenance")
    prov = _read_df("provenance")
    if prov is None or prov.empty:
        st.info("No cell_provenance export. Run preprocessing then dashboard.export.")
        return
    if "year" in prov.columns:
        sub = prov[prov["year"] == year]
    else:
        sub = prov
    if "country_iso3" in sub.columns and "country_iso3" in panel.columns:
        sub = sub[sub["country_iso3"].astype(str).isin(set(panel["country_iso3"].astype(str)))]
    value_cols = [c for c in INDICATORS if c in sub.columns and c != "DMI"]
    if value_cols:
        long = sub.melt(
            id_vars=[c for c in ("country_iso3", "year") if c in sub.columns],
            value_vars=value_cols,
            var_name="indicator_id",
            value_name="provenance",
        )
        counts = long["provenance"].value_counts(dropna=False).rename_axis("state").reset_index(name="n")
        st.subheader(f"Cell states, {year}")
        st.dataframe(counts, use_container_width=True)
        st.bar_chart(counts.set_index("state")["n"])
        st.caption("How to read: official = published; carried_forward = ≤2-year fill; mice_imputed = reconstructed on v2.")
        by_ind = (
            long.groupby(["indicator_id", "provenance"]).size().unstack(fill_value=0)
        )
        st.subheader("By indicator")
        st.dataframe(by_ind, use_container_width=True)
        iso = st.selectbox("Country slice", sorted(sub["country_iso3"].unique()) if "country_iso3" in sub.columns else [])
        if iso:
            st.dataframe(long[long["country_iso3"] == iso], use_container_width=True)


def page_groups(panel: pd.DataFrame) -> None:
    st.header("Country groups")
    if "group_id" not in panel.columns:
        st.warning("Panel has no group_id.")
        return
    g = panel.groupby("group_id")["DMI"].agg(n="count", mean="mean", std="std", min="min", max="max").reset_index()
    st.dataframe(g, use_container_width=True)
    ts = panel.groupby(["year", "group_id"])["DMI"].mean().reset_index().pivot(index="year", columns="group_id", values="DMI")
    st.line_chart(ts)
    st.caption("How to read: group mean DMI over time. Developing and Balkan groups rise faster (σ-convergence).")


def page_predictions() -> None:
    st.header("Predictions (Predict phase)")
    comp = _read_df("model_comparison")
    if comp is not None and not comp.empty:
        show = comp.copy()
        if "status" in show.columns:
            show = show[show["status"] == "ok"]
        cols = [c for c in ("model_type", "rmse", "mae", "mape", "r2", "n_train", "n_test") if c in show.columns]
        st.dataframe(show[cols] if cols else show, use_container_width=True)
        if "rmse" in show.columns:
            st.bar_chart(show.dropna(subset=["rmse"]).set_index("model_type")[["rmse"]])
        st.caption(
            "How to read: hold-out RMSE for 2024–2025 on the point-imputed panel. "
            "Lower is better. CatBoost is lowest on this table. On the leakage-safe protocol "
            "(Hypotheses page) OLS beats CatBoost — frozen verdict, not a possibility."
        )
    preds = _read_df("predictions")
    if preds is not None and not preds.empty:
        st.dataframe(preds.head(200), use_container_width=True)


def page_explanations() -> None:
    st.header("Explanations (Explain phase)")
    xai = _read_json("xai_summary.json") or {}
    factors = xai.get("top_factors") or []
    if factors:
        fdf = pd.DataFrame(factors)
        st.dataframe(fdf, use_container_width=True)
        if {"feature", "importance"}.issubset(fdf.columns):
            st.bar_chart(fdf.set_index("feature")["importance"])
        st.caption("How to read: mean |SHAP|. Rank is predictive importance, not a causal FE coefficient.")
    ont = _read_json("ontology_graph.json") or {}
    semantics = ont.get("indicators") or []
    if semantics:
        st.subheader("Framework semantics (σ)")
        st.dataframe(pd.DataFrame(semantics)[["indicator_id", "name", "pillar", "collection_method", "uri"]], use_container_width=True)
    scenarios = ont.get("scenarios") or []
    if scenarios:
        st.subheader("Contrasting scenarios")
        st.json(scenarios)


def page_convergence(panel: pd.DataFrame) -> None:
    st.header("Convergence")
    n_c = int(panel["country_iso3"].nunique()) if "country_iso3" in panel.columns else 0
    if "DMI" in panel.columns and n_c and n_c < 12:
        g = (
            panel.groupby("year")["DMI"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        g["cv"] = g["std"] / g["mean"].replace(0, pd.NA)
        st.dataframe(g, use_container_width=True)
        st.line_chart(g.set_index("year")["cv"])
        st.caption(
            f"How to read: CV recomputed on the {n_c} countries selected in the sidebar. "
            "This is not the frozen 12-country σ series."
        )
        return
    conv = _read_df("convergence")
    if conv is not None and not conv.empty:
        st.dataframe(conv, use_container_width=True)
        y_col = next((c for c in ("cv", "sigma", "std") if c in conv.columns), None)
        if y_col and "year" in conv.columns:
            st.line_chart(conv.set_index("year")[[y_col]])
        st.caption("How to read: falling CV is σ-convergence (dispersion of DMI across countries declines). Frozen 12-country series.")


def page_ontology() -> None:
    st.header("Ontology / what is what")
    st.caption(
        "Framework explainability. SPARQL and σ are copied from frozen ontology artifacts. "
        "Reasoner is none: HermiT is not run."
    )
    layers = PROJECT_ROOT / "outputs" / "figures" / "processes" / "ontology_layers.png"
    if layers.exists():
        st.image(str(layers), use_container_width=True)
        st.caption(
            "How to read: three bands with TBox arrows (entry*, obtainedBy, explainsOutput, evaluates). "
            "Not a HermiT diagram."
        )
    cons = {}
    sparql = _read_json("sparql_answers.json") or {}
    if not sparql:
        alt = PROJECT_ROOT / "outputs" / "ontology" / "sparql_answers.json"
        if alt.exists():
            sparql = json.loads(alt.read_text(encoding="utf-8"))
    cons = (sparql.get("consistency") or {}) if isinstance(sparql, dict) else {}
    if cons:
        st.subheader("Structural consistency")
        st.json(cons)
        st.caption(
            f"reasoner = {cons.get('reasoner')!r}. {cons.get('note', '')} "
            "Competency questions CQ1–CQ6 are ABox individuals (one questionText each)."
        )
    sigma_path = DATA_DIR / "sigma.md"
    if not sigma_path.exists():
        sigma_path = PROJECT_ROOT / "ontology" / "sigma.md"
    if sigma_path.exists():
        st.subheader("σ mapping")
        st.markdown(sigma_path.read_text(encoding="utf-8"))
    cq = _read_json("competency_answers.json")
    if cq is None:
        alt = PROJECT_ROOT / "outputs" / "ontology" / "competency_answers.json"
        cq = json.loads(alt.read_text(encoding="utf-8")) if alt.exists() else None
    if cq:
        st.subheader("Competency questions (frozen answers)")
        st.json(cq)
    rq_dir = PROJECT_ROOT / "ontology" / "sparql"
    if rq_dir.is_dir():
        with st.expander("SPARQL .rq text"):
            for rq in sorted(rq_dir.glob("CQ*.rq")):
                st.code(rq.read_text(encoding="utf-8"), language="sparql")


def page_hypotheses() -> None:
    st.header("Hypotheses (frozen verdicts)")
    st.caption(
        "Read from results/hypotheses/hypothesis_evaluation.json. "
        "H1.6 headline JSON is the leaky point-panel; leakage-safe is the scientific verdict."
    )
    hyp_path = DATA_DIR / "hypothesis_evaluation.json"
    if not hyp_path.exists():
        hyp_path = PROJECT_ROOT / "results" / "hypotheses" / "hypothesis_evaluation.json"
    if not hyp_path.exists():
        st.info("Hypothesis JSON missing.")
        return
    payload = json.loads(hyp_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    hyps = (data.get("hypotheses") or {}) if isinstance(data, dict) else {}
    leak = _read_json("leakage_safe_ml_verdict.json")
    if leak is None:
        alt = PROJECT_ROOT / "results" / "imputation" / "leakage_safe_ml_verdict.json"
        leak = json.loads(alt.read_text(encoding="utf-8")) if alt.exists() else {}
    rows = []
    for hid in ["H1", "H1.1", "H1.2", "H1.3", "H1.4", "H1.5", "H1.6", "H1.7", "H1.8"]:
        rec = hyps.get(hid) or {}
        note = rec.get("status", "")
        if hid == "H1.6":
            ls = ((rec.get("robustness") or {}).get("leakage_safe") or {}).get("status")
            note = f"point-panel: {rec.get('status')}; leakage-safe: {ls}"
        if hid == "H1.4":
            cc = ((rec.get("robustness") or {}).get("complete_case") or {}).get("status")
            note = f"point-panel/MI-pooled: {rec.get('status')}; complete-case: {cc}"
        if hid == "H1.2":
            cc = ((rec.get("robustness") or {}).get("complete_case") or {}).get("status")
            note = f"point-panel: {rec.get('status')}; complete-case: {cc}"
        if hid == "H1.8":
            note = f"{rec.get('status')} (XAI artifact existence / driver overlap, not a statistical test)"
        rows.append({"id": hid, "status": note, "text": rec.get("text", "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if leak:
        st.subheader("Leakage-safe ML verdict file")
        st.json(leak)
        st.caption("Do not re-fit. This is the frozen 2026-08-11 verdict.")


def main() -> None:
    st.set_page_config(page_title="Digital Maturity VDA", layout="wide", initial_sidebar_state="expanded")
    st.title("Digital Maturity — visual data analysis")
    st.caption("Interactive VDA over official pipeline artifacts. Ontology: http://purl.org/dmi-framework#")
    if not _require_data():
        return
    panel_full = _read_df("panel")
    assert panel_full is not None
    year = _year_slider(panel_full)
    section = st.sidebar.radio(
        "View",
        [
            "Overview",
            "By country",
            "By indicator",
            "By obtaining method",
            "By process phase",
            "Processes",
            "Ontology / what is what",
            "Hypotheses",
            "Provenance",
            "Country groups",
            "Predictions",
            "Explanations",
            "Convergence",
        ],
    )
    grouped_views = {
        "Overview",
        "By country",
        "By indicator",
        "By obtaining method",
        "Provenance",
        "Country groups",
        "Convergence",
    }
    if section in grouped_views:
        panel = _group_filter(panel_full)
    else:
        panel = panel_full
        st.sidebar.caption(
            "Country-group filter is off on this view (sample-wide artifacts: phase map, "
            "processes, ontology, hypotheses, predictions, explanations)."
        )
    pages = {
        "Overview": lambda: page_overview(panel, year),
        "By country": lambda: page_country(panel, year),
        "By indicator": lambda: page_indicator(panel, year),
        "By obtaining method": lambda: page_obtaining(panel),
        "By process phase": page_phase,
        "Processes": page_processes,
        "Ontology / what is what": page_ontology,
        "Hypotheses": page_hypotheses,
        "Provenance": lambda: page_provenance(year, panel),
        "Country groups": lambda: page_groups(panel),
        "Predictions": page_predictions,
        "Explanations": page_explanations,
        "Convergence": lambda: page_convergence(panel),
    }
    pages[section]()


if __name__ == "__main__":
    main()
