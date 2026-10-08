"""Builds and checks the knowledge layer on top of the active pipeline outputs.

    python scripts/run_knowledge_layer.py            # use the stored context vintage
    python scripts/run_knowledge_layer.py --refresh  # download a new WDI vintage first
    python scripts/run_knowledge_layer.py --skip-faults

Writes everything under outputs/knowledge_layer/ and a summary of the quoted
numbers in outputs/knowledge_layer/knowledge_layer_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from rdflib import Graph  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

from common.io import ensure_dir, write_json  # noqa: E402
from knowledge_layer import association, context, cq, metadata, timeseries, validate  # noqa: E402
from knowledge_layer.graph import DMI, GraphBuilder, load_yaml, qb_structure  # noqa: E402

VERSION = "3.1.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-faults", action="store_true")
    args = ap.parse_args()
    out = ensure_dir(ROOT / "outputs" / "knowledge_layer")
    summary: dict = {"version": VERSION}
    timing: dict = {}

    t = time.perf_counter()
    summary["context"] = context.run(ROOT, refresh=args.refresh)
    timing["context"] = time.perf_counter() - t
    t = time.perf_counter()
    ts = timeseries.run(ROOT)
    timing["timeseries"] = time.perf_counter() - t
    t = time.perf_counter()
    summary["association"] = association.run(ROOT)
    timing["association"] = time.perf_counter() - t

    # ---------------------------------------------------------------- graph
    t = time.perf_counter()
    zcfg = context.load_config(ROOT)
    catalog = load_yaml(ROOT / "config/indicators.yaml")["indicators"]
    vcfg = load_yaml(ROOT / "config/vda_views.yaml")
    b = GraphBuilder(ROOT)
    b.load_tbox()
    tbox = Graph()
    for tr in b.g:
        tbox.add(tr)
    b.backbone(load_yaml(ROOT / "config/countries.yaml"), zcfg, catalog)
    built = b.cells(zcfg)
    cells = built["cells"]
    b.add_cells(cells)
    core_series = pd.read_csv(out / "timeseries/core_series.csv")
    ctx_series = pd.read_csv(out / "context/context_series.csv")
    series = pd.concat([core_series, ctx_series], ignore_index=True)
    countries = sorted(cells["country_iso3"].unique())
    b.add_series(series, ["DMI"], countries)
    b.add_models()
    assoc = pd.read_csv(out / "association/context_association.csv")
    b.add_estimates(assoc)
    scores = pd.read_csv(out / "timeseries/rolling_origin_scores.csv")
    b.add_forecast_scores(scores)
    b.add_dispersion()
    b.add_explanations()
    b.add_views(vcfg)
    b.add_competency_questions(cq.CQS)
    g = b.g
    timing["graph_build"] = time.perf_counter() - t

    od = ensure_dir(out / "ontology")
    t = time.perf_counter()
    g.serialize(od / "dmi-kg.ttl", format="turtle")
    g.serialize(od / "dmi-kg.owl", format="xml")
    tbox.serialize(od / "dmi-tbox.ttl", format="turtle")
    tbox.serialize(od / "dmi-tbox.owl", format="xml")
    qbs = qb_structure()
    qbs.serialize(ROOT / "ontology" / "dmi-qb-structure.ttl", format="turtle")
    timing["serialise"] = time.perf_counter() - t
    cells.to_csv(od / "cells.csv", index=False)

    def count(cls):
        return len(set(g.subjects(RDF.type, DMI[cls])))

    summary["graph"] = {
        "triples": len(g),
        "tbox_triples": len(tbox),
        "classes": len(set(tbox.subjects(RDF.type, URIRef_owl("Class")))),
        "object_properties": len(set(tbox.subjects(RDF.type, URIRef_owl("ObjectProperty")))),
        "datatype_properties": len(set(tbox.subjects(RDF.type, URIRef_owl("DatatypeProperty")))),
        "individuals": len(set(g.subjects(RDF.type, URIRef_owl("NamedIndividual")))),
        "score_entries": count("ScoreEntry"),
        "time_series": count("TimeSeries"),
        "estimates": count("Estimate"),
        "forecast_scores": count("ForecastScore"),
        "dispersion_readings": count("DispersionReading"),
        "explanations": count("Explanation"),
        "analytic_views": count("AnalyticView"),
        "provenance": cells["provenance"].value_counts().to_dict(),
        "cells_by_family": {
            "core": int(cells["indicator_id"].str.match(r"X\d+$").sum()),
            "control": int(cells["indicator_id"].isin(["C1", "C2"]).sum()),
            "composite": int((cells["indicator_id"] == "DMI").sum()),
            "context": int(cells["indicator_id"].str.startswith("Z").sum()),
        },
        "official_without_edition": int(((cells["provenance"] == "official") & cells["edition"].isna()).sum()),
        "builder_stats": b.stats,
    }

    # ------------------------------------------------------------- checks
    t = time.perf_counter()
    summary["hermit"] = validate.hermit(od / "dmi-kg.owl")
    timing["hermit_total"] = time.perf_counter() - t
    exp_counts = {
        "OfficialEntry": int((cells["provenance"] == "official").sum()),
        "CarriedEntry": int((cells["provenance"] == "carried_forward").sum()),
        "ReconstructedEntry": int((cells["provenance"] == "mice_imputed").sum()),
        "ConstructedEntry": int((cells["provenance"] == "constructed").sum()),
        "BackcastEntry": int(((cells["provenance"] == "mice_imputed") & (cells["scope"] == "backcast")).sum()),
        "ContextEntry": int(cells["indicator_id"].str.startswith("Z").sum()),
        "ComponentEntry": int(cells["indicator_id"].str.match(r"X\d+$").sum()),
        "BalkanEntry": int(cells["country_iso3"].isin(["MKD", "SRB", "SVN"]).sum()),
        "CompositeComponent": 10,
        "UnobservedSeries": int((series["n_official"] == 0).sum()),
        "SparseSeries": int((series["n_official"] < 7).sum()),
    }
    summary["hermit"]["expected_counts"] = exp_counts
    summary["hermit"]["agreement"] = {k: summary["hermit"]["inferred_counts"].get(k) == v for k, v in exp_counts.items()}

    t = time.perf_counter()
    summary["shacl"] = validate.shacl(g, ROOT / "ontology/shapes/dmi-shapes.ttl")
    timing["shacl"] = time.perf_counter() - t
    full = Graph()
    for tr in g:
        full.add(tr)
    for tr in qbs:
        full.add(tr)
    summary["qb_integrity"] = validate.qb_integrity(full)

    t = time.perf_counter()
    exp = cq.expected(ROOT, cells, series, scores, vcfg)
    summary["competency_questions"] = cq.run_all(g, exp)
    timing["competency_questions"] = time.perf_counter() - t

    summary["metadata"] = metadata.write(ROOT, out / "metadata" / "dataset.jsonld", catalog, zcfg, VERSION)

    if not args.skip_faults:
        t = time.perf_counter()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            summary["fault_injection"] = validate.fault_injection(g, ROOT / "ontology/shapes/dmi-shapes.ttl", Path(tmp))
        timing["fault_injection"] = time.perf_counter() - t

    summary["timing_s"] = {k: round(v, 2) for k, v in timing.items()}
    summary["rolling_origin"] = ts["scores"]
    write_json(out / "knowledge_layer_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rolling_origin", "association")}, indent=1, default=str)[:6000])


def URIRef_owl(local: str):
    from rdflib.namespace import OWL

    return OWL[local]


if __name__ == "__main__":
    main()
