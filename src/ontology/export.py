"""Materialize ABox individuals from YAML + parquet. OWL stores structure; Python computes values."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from catalog.data_catalog import DataCatalog
from common.io import ensure_dir, read_df, write_df, write_json
from common.logging_utils import get_logger
from ontology.query import MODEL_PHASE_MAP, answer_all

logger = get_logger("dmf.ontology")

NS = "http://purl.org/dmi-framework#"
PILLAR_URI = {
    "infrastructure": f"{NS}Pillar_infrastructure",
    "e_government": f"{NS}Pillar_e_government",
    "skills": f"{NS}Pillar_skills",
    "trust_innovation": f"{NS}Pillar_trust_innovation",
    "digital_economy": f"{NS}Pillar_digital_economy",
    "control": f"{NS}Pillar_control",
    "composite": f"{NS}Pillar_composite",
}
PHASE_URI = {
    "acquire": f"{NS}Phase_Acquire",
    "validate": f"{NS}Phase_Validate",
    "process": f"{NS}Phase_Process",
    "index": f"{NS}Phase_Index",
    "analyse": f"{NS}Phase_Analyse",
    "predict": f"{NS}Phase_Predict",
    "explain": f"{NS}Phase_Explain",
    "visualize": f"{NS}Phase_Visualize",
}
METHOD_URI = {
    "official_api": f"{NS}Method_official_api",
    "api": f"{NS}Method_official_api",
    "curated_snapshot": f"{NS}Method_curated_snapshot",
    "primary_survey": f"{NS}Method_primary_survey",
    "sensor_automated": f"{NS}Method_sensor_automated",
    "constructed": f"{NS}Method_constructed",
}
PROV_URI = {
    "official": f"{NS}Prov_official",
    "carried_forward": f"{NS}Prov_carried_forward",
    "mice_imputed": f"{NS}Prov_mice_imputed",
    "missing": f"{NS}Prov_missing",
    "constructed": f"{NS}Prov_constructed",
}
ACQUIRED_IDS = [f"X{i}" for i in range(1, 11)] + ["C1", "C2"]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _uri(local: str) -> str:
    return f"{NS}{local}"


def _load_panel(root: Path) -> Optional[pd.DataFrame]:
    for path in (
        root / "data" / "processed" / "analysis_panel.parquet",
        root / "data" / "processed" / "dmi_panel.parquet",
    ):
        if path.exists():
            return read_df(path)
    return None


def _load_provenance(root: Path) -> Optional[pd.DataFrame]:
    path = root / "data" / "processed" / "cell_provenance.parquet"
    if path.exists():
        return read_df(path)
    return None


def _countries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = (cfg.get("groups") or {})
    for gid, group in groups.items():
        if not isinstance(group, dict):
            continue
        for c in group.get("countries") or []:
            rows.append(
                {
                    "iso3": c.get("iso3"),
                    "iso2": c.get("iso2"),
                    "name": c.get("name"),
                    "name_mk": c.get("name_mk"),
                    "group_id": gid,
                    "group_name": group.get("name"),
                    "uri": _uri(f"Country_{c.get('iso3')}"),
                    "group_uri": _uri(f"Group_{gid}"),
                }
            )
    return rows


def _sources(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for key, spec in (cfg.get("sources") or {}).items():
        rows.append(
            {
                "source_id": key,
                "uri": _uri(f"Source_{key}"),
                "adapter": (spec or {}).get("adapter"),
                "base_url": (spec or {}).get("base_url"),
                "snapshot_file": (spec or {}).get("snapshot_file"),
            }
        )
    rows.append(
        {
            "source_id": "constructed",
            "uri": _uri("Source_constructed"),
            "adapter": None,
            "base_url": None,
            "snapshot_file": None,
        }
    )
    return rows


def _indicator_individuals(catalog: DataCatalog) -> List[Dict[str, Any]]:
    out = []
    for rec in catalog.list():
        iid = str(rec.get("indicator_id"))
        method = rec.get("collection_method") or rec.get("retrieval_method") or "official_api"
        phases = rec.get("process_phases") or []
        out.append(
            {
                "indicator_id": iid,
                "uri": rec.get("uri") or _uri(f"Ind_{iid}"),
                "name": rec.get("name"),
                "definition": rec.get("definition"),
                "role": rec.get("role"),
                "pillar": rec.get("pillar"),
                "pillar_uri": PILLAR_URI.get(str(rec.get("pillar")), PILLAR_URI["composite"]),
                "unit": rec.get("unit"),
                "polarity": rec.get("polarity") or "positive",
                "collection_method": method,
                "collection_method_uri": METHOD_URI.get(str(method), METHOD_URI["official_api"]),
                "source": rec.get("source"),
                "source_uri": _uri(f"Source_{rec.get('source')}"),
                "indicator_code": rec.get("indicator_code"),
                "same_as": rec.get("same_as") or rec.get("source_url"),
                "process_phases": list(phases),
                "process_phase_uris": [PHASE_URI[p] for p in phases if p in PHASE_URI],
            }
        )
    return out


def _score_entries(
    panel: Optional[pd.DataFrame],
    provenance: Optional[pd.DataFrame],
    catalog: DataCatalog,
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame(
            columns=[
                "uri",
                "indicator_id",
                "country_iso3",
                "year",
                "value",
                "provenance",
                "source_id",
                "phase_id",
                "collection_method",
            ]
        )
    recs = {r["indicator_id"]: r for r in catalog.list()}
    id_cols = [c for c in ACQUIRED_IDS + ["DMI"] if c in panel.columns]
    work = panel.copy()
    if "country_iso3" not in work.columns or "year" not in work.columns:
        return pd.DataFrame()

    prov_long = None
    if provenance is not None and not provenance.empty:
        p = provenance.copy()
        if "country_iso3" in p.columns and "year" in p.columns:
            pcols = [c for c in ACQUIRED_IDS if c in p.columns]
            if pcols:
                prov_long = p.melt(
                    id_vars=["country_iso3", "year"],
                    value_vars=pcols,
                    var_name="indicator_id",
                    value_name="provenance",
                )

    long = work.melt(
        id_vars=[c for c in ("country_iso3", "year", "group_id") if c in work.columns],
        value_vars=id_cols,
        var_name="indicator_id",
        value_name="value",
    )
    if prov_long is not None:
        long = long.merge(
            prov_long, on=["country_iso3", "year", "indicator_id"], how="left"
        )
    else:
        long["provenance"] = long["indicator_id"].map(
            lambda i: "constructed" if i == "DMI" else "official"
        )
    long["provenance"] = long["provenance"].fillna("missing")
    long.loc[long["indicator_id"] == "DMI", "provenance"] = long.loc[
        long["indicator_id"] == "DMI", "provenance"
    ].replace({"official": "constructed", "missing": "constructed"})

    long["source_id"] = long["indicator_id"].map(
        lambda i: (recs.get(i) or {}).get("source") or "constructed"
    )
    long["collection_method"] = long["indicator_id"].map(
        lambda i: (recs.get(i) or {}).get("collection_method")
        or (recs.get(i) or {}).get("retrieval_method")
        or "constructed"
    )
    long["phase_id"] = long["indicator_id"].map(
        lambda i: "index" if i == "DMI" else "acquire"
    )
    long["uri"] = (
        NS
        + "SE_"
        + long["indicator_id"].astype(str)
        + "_"
        + long["country_iso3"].astype(str)
        + "_"
        + long["year"].astype(int).astype(str)
    )
    long["indicator_uri"] = long["indicator_id"].map(lambda i: _uri(f"Ind_{i}"))
    long["country_uri"] = long["country_iso3"].map(lambda c: _uri(f"Country_{c}"))
    long["phase_uri"] = long["phase_id"].map(lambda p: PHASE_URI.get(p, PHASE_URI["acquire"]))
    long["source_uri"] = long["source_id"].map(lambda s: _uri(f"Source_{s}"))
    long["provenance_uri"] = long["provenance"].map(
        lambda p: PROV_URI.get(str(p), PROV_URI["missing"])
    )
    return long


def _ttl_esc(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', "'")


def _append_evaluation_runs(populated: Path, sparql_out: Dict[str, Any]) -> None:
    """Record SPARQL CQ outcomes as EvaluationRun individuals. Not a DL reasoner."""
    extra = ["", "# ----- evaluation runs (rdflib SPARQL, reasoner none) -----", ""]
    for cq_id in ("CQ1", "CQ2", "CQ3", "CQ4", "CQ5", "CQ6"):
        rec = sparql_out.get(cq_id) or {}
        passed = "true" if rec.get("passed") else "false"
        extra.append(
            f"dmi:Eval_{cq_id} a dmi:EvaluationRun ; "
            f'rdfs:label "rdflib SPARQL {cq_id}" ; '
            f"dmi:evaluates dmi:{cq_id} ; "
            f"dmi:evaluatedWith dmi:VQ_{cq_id} ; "
            f"dmi:validationPassed {passed} ."
        )
    text = populated.read_text(encoding="utf-8").rstrip()
    populated.write_text(text + "\n" + "\n".join(extra) + "\n", encoding="utf-8")


def _append_abox_ttl(
    tbox_path: Path,
    out_path: Path,
    indicators: List[Dict[str, Any]],
    countries: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    scores: pd.DataFrame,
    shap: List[Dict[str, Any]],
) -> None:
    lines = [tbox_path.read_text(encoding="utf-8").rstrip(), "", "# ----- generated ABox -----", ""]
    for src in sources:
        lines.append(
            f"dmi:Source_{src['source_id']} a dmi:DataSource ; "
            f'rdfs:label "{_ttl_esc(src["source_id"])}" .'
        )
    lines.append("")
    seen_groups = set()
    for c in countries:
        gid = c["group_id"]
        if gid not in seen_groups:
            lines.append(
                f'dmi:Group_{gid} a dmi:CountryGroup ; rdfs:label "{_ttl_esc(c["group_name"])}" .'
            )
            seen_groups.add(gid)
        lines.append(
            f"dmi:Country_{c['iso3']} a dmi:Country ; "
            f'dmi:iso3 "{c["iso3"]}" ; rdfs:label "{_ttl_esc(c["name"])}" ; '
            f"dmi:memberOf dmi:Group_{gid} ."
        )
    lines.append("")
    for ind in indicators:
        iid = ind["indicator_id"]
        label = _ttl_esc(ind["name"])
        method_local = str(ind["collection_method_uri"]).split("#")[-1]
        pillar_local = str(ind["pillar_uri"]).split("#")[-1]
        phase_refs = ", ".join(
            "dmi:" + u.split("#")[-1] for u in ind.get("process_phase_uris") or []
        )
        extra = f" ; dmi:usedInPhase {phase_refs}" if phase_refs else ""
        lines.append(
            f"dmi:Ind_{iid} a dmi:Indicator ; "
            f'dmi:indicatorId "{iid}" ; rdfs:label "{label}" ; '
            f"dmi:belongsToPillar dmi:{pillar_local} ; "
            f"dmi:obtainedBy dmi:{method_local} ; "
            f"dmi:hasSource dmi:Source_{ind['source']} ; "
            f'dmi:role "{_ttl_esc(ind.get("role"))}" ; dmi:unit "{_ttl_esc(ind.get("unit"))}" ; '
            f'dmi:polarity "{_ttl_esc(ind.get("polarity"))}"{extra} .'
        )
    lines.append("")
    phase_local = {
        "acquire": "Phase_Acquire",
        "validate": "Phase_Validate",
        "process": "Phase_Process",
        "index": "Phase_Index",
        "analyse": "Phase_Analyse",
        "predict": "Phase_Predict",
        "explain": "Phase_Explain",
        "visualize": "Phase_Visualize",
    }
    for mid, spec in MODEL_PHASE_MAP.items():
        local = "Model_" + str(mid).replace("-", "_")
        ph = phase_local.get(spec["phase"], "Phase_Predict")
        lines.append(
            f"dmi:{local} a dmi:ModelOutput ; "
            f'rdfs:label "{_ttl_esc(mid)}" ; '
            f'rdfs:comment "{_ttl_esc(spec["purpose"])}" ; '
            f"dmi:assignedToPhase dmi:{ph} ."
        )
    lines.append("")
    for s in shap:
        local = str(s["uri"]).split("#")[-1]
        iid = s.get("indicator_id")
        shap_v = s.get("mean_abs_shap")
        if iid is None or shap_v is None:
            continue
        lines.append(
            f"dmi:{local} a dmi:Explanation ; "
            f"dmi:aboutIndicator dmi:Ind_{iid} ; "
            f"dmi:explainsOutput dmi:Model_catboost ; "
            f"dmi:meanAbsShap {float(shap_v)} ."
        )
    lines.append("")
    if scores is not None and not scores.empty:
        for _, r in scores.iterrows():
            val = r["value"]
            if pd.isna(val):
                continue
            local = str(r["uri"]).split("#")[-1]
            year = int(r["year"])
            phase_local = str(r.get("phase_uri") or "").split("#")[-1] or "Phase_Acquire"
            source_local = str(r.get("source_uri") or "").split("#")[-1] or "Source_constructed"
            lines.append(
                f"dmi:{local} a dmi:ScoreEntry ; "
                f"dmi:entryIndicator dmi:Ind_{r['indicator_id']} ; "
                f"dmi:entryCountry dmi:Country_{r['country_iso3']} ; "
                f"dmi:entryPhase dmi:{phase_local} ; "
                f"dmi:entrySource dmi:{source_local} ; "
                f"dmi:year {year} ; "
                f"dmi:matrixValue {float(val)} ; "
                f"dmi:hasProvenance dmi:{str(r['provenance_uri']).split('#')[-1]} ."
            )
    lines.append("")
    lines.append(
        "dmi:CTX_overview_dashboard a dmi:DecisionContext ; "
        'rdfs:label "Dashboard overview slice" ; '
        'rdfs:comment "Not a Recommendation individual; VDA does not recommend a visualization technique." .'
    )
    lines.append(
        "dmi:CTX_prediction_xai a dmi:DecisionContext ; "
        'rdfs:label "SHAP attribution context" ; '
        'rdfs:comment "TreeExplainer on frozen CatBoost (results/xai/xai_shap_global.meta.json)." .'
    )
    lines.append("")
    from ontology.sparql import QUERIES

    cq_questions = {
        "CQ1": "By which collection method is each indicator obtained?",
        "CQ2": "What provenance states appear for MKD in 2025?",
        "CQ3": "In which process phase is CatBoost assigned?",
        "CQ4": "Which indicators have the highest mean |SHAP|?",
        "CQ5": "What ScoreEntry values exist for MKD 2025, with provenance, phase, and source?",
        "CQ6": "How many countries and indicators are in the graph?",
    }
    for cq_id, question in cq_questions.items():
        qbody = str(QUERIES[cq_id]).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        lines.append(
            f"dmi:{cq_id} a dmi:CompetencyQuestion ; "
            f'dmi:questionText "{_ttl_esc(question)}" ; '
            f"dmi:usesValidationQuery dmi:VQ_{cq_id} ."
        )
        lines.append(f"dmi:VQ_{cq_id} a dmi:ValidationQuery ; dmi:queryText \"\"\"{qbody}\"\"\" .")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shap_explanations(root: Path) -> List[Dict[str, Any]]:
    path = root / "results" / "xai" / "shap_global_importance.csv"
    if not path.exists():
        alt = root / "outputs" / "xai" / "shap_global_importance.csv"
        path = alt if alt.exists() else path
    if not path.exists():
        return []
    df = read_df(path)
    feat = "feature" if "feature" in df.columns else df.columns[0]
    val = "mean_abs_shap" if "mean_abs_shap" in df.columns else df.columns[1]
    rows = []
    for i, r in df.iterrows():
        feature = str(r[feat]).replace("_lag1", "")
        rows.append(
            {
                "uri": _uri(f"Expl_SHAP_{feature}_{i}"),
                "indicator_id": feature,
                "indicator_uri": _uri(f"Ind_{feature}"),
                "method": "SHAP",
                "mean_abs_shap": float(r[val]) if pd.notna(r[val]) else None,
                "rank": int(i) + 1 if isinstance(i, int) else None,
                "context_uri": _uri("CTX_prediction_xai"),
            }
        )
    return rows


def _contrasting_scenarios(
    panel: Optional[pd.DataFrame], shap: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if panel is None or panel.empty or "DMI" not in panel.columns:
        return []
    latest = int(panel["year"].max())
    slice_ = panel[panel["year"] == latest]
    picks: List[Tuple[str, str]] = []
    for iso, label in (("MKD", "balkan_north_macedonia"), ("DEU", "developed_eu_germany")):
        if "country_iso3" in slice_.columns and iso in set(slice_["country_iso3"]):
            picks.append((iso, label))
    top = [s["indicator_id"] for s in shap[:5]]
    scenarios = []
    for iso, key in picks:
        row = slice_[slice_["country_iso3"] == iso].iloc[0]
        binding = []
        for iid in top:
            if iid in row.index and pd.notna(row[iid]):
                binding.append({"indicator_id": iid, "value": float(row[iid])})
        scenarios.append(
            {
                "id": key,
                "country_iso3": iso,
                "year": latest,
                "DMI": float(row["DMI"]) if pd.notna(row["DMI"]) else None,
                "group_id": row.get("group_id"),
                "binding_indicators": binding,
                "context_uri": _uri("CTX_overview_dashboard"),
            }
        )
    return scenarios


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    """Export TBox+ABox JSON, ScoreEntry table, populated TTL, and CQ answers."""
    root = Path(project_root).resolve()
    out_dir = ensure_dir(root / "outputs" / "ontology")
    catalog = DataCatalog.from_project(root)
    countries_cfg = _load_yaml(root / "config" / "countries.yaml")
    sources_cfg = _load_yaml(root / "config" / "sources.yaml")
    index_cfg = _load_yaml(root / "config" / "index.yaml")

    indicators = _indicator_individuals(catalog)
    countries = _countries(countries_cfg)
    sources = _sources(sources_cfg)
    panel = _load_panel(root)
    provenance = _load_provenance(root)
    scores = _score_entries(panel, provenance, catalog)
    shap = _shap_explanations(root)
    scenarios = _contrasting_scenarios(panel, shap)

    graph: Dict[str, Any] = {
        "namespace": NS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layers": ["Ctax", "Cdec", "Ceval"],
        "indicators": indicators,
        "countries": countries,
        "sources": sources,
        "pillars": [
            {"id": k, "uri": v, "weight": (index_cfg.get("default_weights") or {}).get(k)}
            for k, v in PILLAR_URI.items()
        ],
        "phases": [{"id": k, "uri": v} for k, v in PHASE_URI.items()],
        "collection_methods": [
            {"id": k, "uri": v} for k, v in METHOD_URI.items() if k != "api"
        ],
        "model_phase_map": MODEL_PHASE_MAP,
        "explanations": shap,
        "scenarios": scenarios,
        "score_entry_n": int(len(scores)),
        "tbox": "ontology/dmi-framework.ttl",
        "sigma": "ontology/sigma.md",
    }

    write_json(out_dir / "graph.json", graph)
    if not scores.empty:
        write_df(scores, out_dir / "score_entries.parquet")
        write_df(scores, out_dir / "score_entries.csv")

    tbox = root / "ontology" / "dmi-framework.ttl"
    latest_year = int(scores["year"].max()) if not scores.empty else None
    populated = out_dir / "dmi-framework-populated.ttl"
    if tbox.exists():
        _append_abox_ttl(
            tbox,
            populated,
            indicators,
            countries,
            sources,
            scores,
            shap,
        )

    cq = answer_all(graph, scores, shap, default_iso3="MKD", default_year=latest_year)
    write_json(out_dir / "competency_answers.json", cq)

    sparql_out: Dict[str, Any] = {}
    if populated.exists():
        from ontology.sparql import answer_sparql_cqs, write_query_files

        write_query_files(root / "ontology" / "sparql")
        try:
            sparql_out = answer_sparql_cqs(populated)
            write_json(out_dir / "sparql_answers.json", sparql_out)
            _append_evaluation_runs(populated, sparql_out)
        except ImportError:
            sparql_out = {"error": "rdflib is not installed"}
            write_json(out_dir / "sparql_answers.json", sparql_out)

    logger.info(
        "Ontology export: %d indicators, %d countries, %d score entries",
        len(indicators),
        len(countries),
        len(scores),
    )
    sparql_ok = bool(sparql_out) and all(
        (v.get("passed") if isinstance(v, dict) and "passed" in v else True)
        for k, v in sparql_out.items()
        if str(k).startswith("CQ")
    )
    return {
        "output_dir": str(out_dir.relative_to(root)),
        "n_indicators": len(indicators),
        "n_countries": len(countries),
        "n_score_entries": int(len(scores)),
        "cq_passed": all(bool(v.get("passed")) for v in cq.values()),
        "sparql_passed": sparql_ok,
    }
