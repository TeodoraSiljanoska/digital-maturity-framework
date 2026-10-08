"""JSON-LD description of the published dataset (schema.org Dataset + DCAT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = "https://github.com/TeodoraSiljanoska/digital-maturity-framework"


def dataset_jsonld(root: Path, indicators: List[Dict[str, Any]], zcfg: Dict[str, Any], version: str) -> Dict[str, Any]:
    countries = yaml.safe_load((root / "config/countries.yaml").read_text(encoding="utf-8"))["groups"]
    places = [
        {"@type": "Country", "name": c["name"], "identifier": c["iso3"]}
        for g in countries.values() for c in g["countries"]
    ]
    variables = [
        {"@type": "PropertyValue", "name": r["name"], "propertyID": f"http://purl.org/dmi-framework#Ind_{r['indicator_id']}",
         "alternateName": r["indicator_id"], "unitText": str(r.get("unit"))}
        for r in indicators
    ] + [
        {"@type": "PropertyValue", "name": s["name"], "propertyID": f"http://purl.org/dmi-framework#Ind_{z}",
         "alternateName": z, "unitText": s["unit"]}
        for z, s in zcfg["indicators"].items()
    ]
    dist = [
        ("Knowledge graph (Turtle)", "text/turtle", "outputs/knowledge_layer/ontology/dmi-kg.ttl"),
        ("Knowledge graph (RDF/XML, for Protégé)", "application/rdf+xml", "outputs/knowledge_layer/ontology/dmi-kg.owl"),
        ("SHACL shapes", "text/turtle", "ontology/shapes/dmi-shapes.ttl"),
        ("Star schema for Power BI (CSV)", "text/csv", "outputs/knowledge_layer/powerbi/"),
        ("Processed panel (Parquet)", "application/vnd.apache.parquet", "data/processed/analysis_panel.parquet"),
    ]
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
        },
        "@type": ["Dataset", "dcat:Dataset"],
        "@id": "https://doi.org/10.5281/zenodo.23043842",
        "name": "Digital Maturity Framework: provenance-tracked panel of digital-maturity time series (2012-2025)",
        "description": (
            "Annual country series of ten determinants (X1-X10), two controls (C1, C2), the constructed "
            "Digital Maturity Index and six context indicators (Z1-Z6) for twelve countries, 2012-2025, "
            "with the provenance state, edition and retrieval method of every cell, published as a "
            "knowledge graph aligned with RDF Data Cube, PROV-O, SKOS and DCAT."
        ),
        "version": version,
        "url": REPO,
        "identifier": "https://doi.org/10.5281/zenodo.23043842",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "isAccessibleForFree": True,
        "creator": [
            {"@type": "Person", "name": "Teodora Siljanoska Taskovska", "identifier": "https://orcid.org/0009-0008-9147-7408",
             "affiliation": {"@type": "Organization", "name": "University St. Kliment Ohridski - Bitola"}},
            {"@type": "Person", "name": "Snezana Savoska", "identifier": "https://orcid.org/0000-0002-0539-1771",
             "affiliation": {"@type": "Organization", "name": "University St. Kliment Ohridski - Bitola"}},
        ],
        "temporalCoverage": "2012/2025",
        "spatialCoverage": places,
        "variableMeasured": variables,
        "keywords": ["digital maturity", "time series", "data provenance", "ontology", "visual analytics", "knowledge graph"],
        "isBasedOn": [
            "https://api.worldbank.org/v2/",
            "https://publicadministration.un.org/egovkb/",
            "https://ncsi.ega.ee/",
            "https://www.wipo.int/global_innovation_index/",
            "https://oxfordinsights.com/ai-readiness/",
            "https://hdr.undp.org/",
        ],
        "distribution": [
            {"@type": ["DataDownload", "dcat:Distribution"], "name": n, "encodingFormat": f, "contentUrl": f"{REPO}/tree/main/{p}"}
            for n, f, p in dist
        ],
        "dct:conformsTo": [
            {"@id": "http://purl.org/linked-data/cube"},
            {"@id": "http://www.w3.org/ns/prov"},
        ],
    }


def write(root: Path, out: Path, indicators, zcfg, version: str) -> Dict[str, Any]:
    from rdflib import Graph

    doc = dataset_jsonld(root, indicators, zcfg, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    g = Graph().parse(data=json.dumps(doc), format="json-ld")
    return {"path": str(out.relative_to(root)), "triples": len(g), "variables": len(doc["variableMeasured"]), "distributions": len(doc["distribution"])}
