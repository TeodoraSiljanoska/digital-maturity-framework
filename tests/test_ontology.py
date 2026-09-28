"""Ontology coverage, competency questions, reusable catalog, process diagrams."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from catalog.data_catalog import DataCatalog
from common.io import read_json
from dashboard.powerbi_export import run as pb_run
from ontology.export import run as export_ontology
from ontology.query import MODEL_PHASE_MAP, answer_cq
from visualization.process_diagrams import run as run_diagrams

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_uris_and_facets():
    cat = DataCatalog.from_project(ROOT)
    for iid in ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "C1", "C2"]:
        assert iid in cat
        uri = cat.uri(iid)
        assert uri.startswith("http://purl.org/dmi-framework#Ind_")
        rec = cat.get(iid)
        assert rec.get("collection_method")
        assert rec.get("process_phases")
    assert len(cat.acquired_ids()) == 12
    assert "DMI" in cat
    assert cat.by_collection_method("official_api")
    assert cat.by_phase("acquire")
    frame = cat.reusable_frame()
    assert {"uri", "collection_method", "pillar"}.issubset(frame.columns)
    assert len(frame) >= 13


def test_catboost_is_predict_not_collect():
    assert MODEL_PHASE_MAP["catboost"]["phase"] == "predict"
    assert MODEL_PHASE_MAP["shap"]["phase"] == "explain"
    assert MODEL_PHASE_MAP["world_bank_api"]["phase"] == "acquire"


def test_ontology_export_coverage_and_cqs():
    result = export_ontology(ROOT)
    assert result["n_indicators"] >= 13
    assert result["n_countries"] == 12
    graph_path = ROOT / "outputs" / "ontology" / "graph.json"
    scores_path = ROOT / "outputs" / "ontology" / "score_entries.parquet"
    answers_path = ROOT / "outputs" / "ontology" / "competency_answers.json"
    assert graph_path.exists()
    assert (ROOT / "outputs" / "ontology" / "dmi-framework-populated.ttl").exists()
    graph = read_json(graph_path)
    ids = {i["indicator_id"] for i in graph["indicators"]}
    for iid in ["X1", "X2", "X10", "C1", "C2", "DMI"]:
        assert iid in ids
    isos = {c["iso3"] for c in graph["countries"]}
    assert isos >= {"MKD", "DEU", "USA", "IND"}
    scores = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
    cq1 = answer_cq("CQ1", graph, scores)
    assert cq1["passed"]
    cq3 = answer_cq("CQ3", graph)
    assert cq3["answer"]["phase"] == "predict"
    cq6 = answer_cq("CQ6", graph)
    assert cq6["passed"]
    answers = read_json(answers_path)
    assert answers["CQ6"]["passed"]
    if not scores.empty:
        assert result["n_score_entries"] >= 12 * 12
        cq2 = answer_cq("CQ2", graph, scores, iso3="MKD")
        assert cq2["passed"]


def test_process_diagrams_written():
    result = run_diagrams(ROOT)
    arts = result["artifacts"]
    assert any("assessment_pipeline.png" in a for a in arts)
    assert any("missing_data_workflow.png" in a for a in arts)
    assert any("model_to_phase.png" in a for a in arts)
    assert (ROOT / "docs" / "processes" / "assessment_pipeline.md").exists()
    assert (ROOT / "outputs" / "figures" / "processes" / "captions.json").exists()
    for name in (
        "assessment_pipeline.png",
        "missing_data_workflow.png",
        "model_to_phase.png",
        "acquisition_methods.png",
        "construction_p1_p8.png",
        "correlation_vif_process.png",
    ):
        assert (ROOT / "outputs" / "figures" / "processes" / name).exists()
    captions = read_json(ROOT / "outputs" / "figures" / "processes" / "captions.json")
    acq = captions.get("acquisition_methods.png") or ""
    assert "pending mentor confirmation" not in acq
    assert "traditional" in acq.lower()
    assert "constructed" in acq.lower()
    assert (ROOT / "outputs" / "figures" / "processes" / "ontology_layers.png").exists()
    from visualization.process_diagrams import (
        _draw_ontology_layers,
        _FIG_CLASSES,
        _FIG_PROPS,
        _layer_classes_from_ttl,
        _tbox_object_properties,
    )
    import inspect

    draw_src = inspect.getsource(_draw_ontology_layers)
    assert "_layer_classes_from_ttl" in draw_src
    assert "_tbox_object_properties" in draw_src
    ttl = ROOT / "ontology" / "dmi-framework.ttl"
    layers = _layer_classes_from_ttl(ttl)
    assert layers["Ctax"]
    assert layers["Cdec"]
    assert layers["Ceval"]
    for layer, names in _FIG_CLASSES.items():
        for name in names:
            assert name in layers[layer], name
    props = set(_tbox_object_properties(ttl))
    for triple in _FIG_PROPS:
        assert triple in props, triple
    ont_cap = captions.get("ontology_layers.png") or ""
    assert "dmi-framework.ttl" in ont_cap
    assert "layout" in ont_cap.lower() or "positions" in ont_cap.lower()


def test_sparql_cqs_over_populated_ttl():
    result = export_ontology(ROOT)
    ttl = ROOT / "outputs" / "ontology" / "dmi-framework-populated.ttl"
    assert ttl.exists()
    from ontology.sparql import answer_sparql_cqs

    answers = answer_sparql_cqs(ttl)
    assert answers["consistency"]["passed"]
    assert answers["consistency"]["reasoner"] == "none"
    assert answers["CQ1"]["passed"]
    ids_by_method = {}
    for row in answers["CQ1"]["answer"]:
        ids_by_method.setdefault(row["method"], []).append(row["id"])
    assert "X2" in ids_by_method.get("Official statistical API", [])
    assert answers["CQ3"]["passed"]
    assert answers["CQ4"]["passed"]
    assert answers["CQ6"]["passed"]
    assert result.get("sparql_passed") is True
    assert (ROOT / "ontology" / "sparql" / "CQ1.rq").exists()
    assert (ROOT / "outputs" / "ontology" / "sparql_answers.json").exists()
    cq5 = answers["CQ5"]["answer"]
    assert cq5
    assert {"id", "value", "prov", "phase", "source"} <= set(cq5[0].keys())
    from rdflib import RDF, URIRef

    from ontology.sparql import load_graph

    g = load_graph(ttl)
    ns = "http://purl.org/dmi-framework#"
    n_phase = sum(
        1
        for se in g.subjects(RDF.type, URIRef(ns + "ScoreEntry"))
        if list(g.objects(se, URIRef(ns + "entryPhase")))
    )
    n_src = sum(
        1
        for se in g.subjects(RDF.type, URIRef(ns + "ScoreEntry"))
        if list(g.objects(se, URIRef(ns + "entrySource")))
    )
    n_se = len(list(g.subjects(RDF.type, URIRef(ns + "ScoreEntry"))))
    assert n_se >= 12 * 12
    assert n_phase == n_se
    assert n_src == n_se
    assert len(list(g.subjects(RDF.type, URIRef(ns + "EvaluationRun")))) == 6
    assert len(list(g.subjects(RDF.type, URIRef(ns + "CompetencyQuestion")))) == 6
    assert len(list(g.subjects(RDF.type, URIRef(ns + "ValidationQuery")))) == 6
    eval_cq1 = URIRef(ns + "Eval_CQ1")
    assert list(g.objects(eval_cq1, URIRef(ns + "evaluates")))
    assert list(g.objects(eval_cq1, URIRef(ns + "evaluatedWith")))
    assert len(list(g.subjects(RDF.type, URIRef(ns + "Recommendation")))) == 0
    expl = next(g.subjects(RDF.type, URIRef(ns + "Explanation")))
    assert list(g.objects(expl, URIRef(ns + "explainsOutput")))


def test_score_entries_reusable_by_phase_country_method_provenance():
    path = ROOT / "outputs" / "ontology" / "score_entries.parquet"
    assert path.exists()
    se = pd.read_parquet(path)
    for col in ("indicator_id", "country_iso3", "phase_id", "collection_method", "provenance"):
        assert col in se.columns
    assert se["country_iso3"].nunique() == 12
    assert {"official_api", "curated_snapshot"}.issubset(set(se["collection_method"].unique()))
    assert {"official", "carried_forward", "mice_imputed"}.issubset(set(se["provenance"].unique()))


def test_powerbi_star_schema():
    out = pb_run(ROOT)
    folder = ROOT / "outputs" / "powerbi"
    for name in ("dim_indicator.csv", "dim_country.csv", "dim_phase.csv", "POWERBI_README.md"):
        assert (folder / name).exists()
    assert out["n_fact_rows"] >= 0
    assert (folder / "LoadStarSchema.pq").exists()
