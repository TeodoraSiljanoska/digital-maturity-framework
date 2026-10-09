"""Quality checks of the knowledge graph.

* OWL 2 DL reasoning with HermiT (through Owlready2): consistency and the
  classification of the defined classes, compared with counts from the tables.
* SHACL validation (pySHACL) of the closed-world data contract.
* RDF Data Cube integrity constraints (selected IC queries of the W3C spec).
* Fault injection: ten known faults are planted in copies of the graph and
  each copy is checked by both the reasoner and the SHACL engine.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from knowledge_layer.graph import DMI, QB, dec

_JDK = r"C:\Program Files\Microsoft\jdk-11.0.12.7-hotspot\bin\java.exe"
JAVA = os.environ.get("DMF_JAVA") or (_JDK if os.path.exists(_JDK) else shutil.which("java") or "java")

DEFINED = [
    "OfficialEntry", "CarriedEntry", "ReconstructedEntry", "ConstructedEntry", "BackcastEntry",
    "ContextEntry", "ComponentEntry", "BalkanEntry", "CompositeComponent", "UnobservedSeries",
    "SparseSeries", "SignificantEstimate",
]


def hermit(owl_path: Path) -> Dict[str, Any]:
    import owlready2

    owlready2.JAVA_EXE = JAVA
    world = owlready2.World()
    t0 = time.perf_counter()
    with open(owl_path, "rb") as fh:
        onto = world.get_ontology(str(DMI)).load(fileobj=fh)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        with onto:
            owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        consistent = True
        err = None
    except owlready2.OwlReadyInconsistentOntologyError as exc:
        consistent, err = False, str(exc)[:300]
    t_reason = time.perf_counter() - t1
    counts: Dict[str, int] = {}
    if consistent:
        for name in DEFINED:
            cls = world[str(DMI[name])]
            counts[name] = len(list(cls.instances())) if cls is not None else -1
    return {"consistent": consistent, "error": err, "load_s": round(t_load, 2), "reason_s": round(t_reason, 2), "inferred_counts": counts}


def shacl(data: Graph, shapes_path: Path) -> Dict[str, Any]:
    from pyshacl import validate

    shapes = Graph().parse(shapes_path, format="turtle")
    t0 = time.perf_counter()
    conforms, report_graph, text = validate(data, shacl_graph=shapes, inference="none", advanced=True, allow_warnings=True)
    dt = time.perf_counter() - t0
    from rdflib.namespace import SH

    results = list(report_graph.subjects(RDF.type, SH.ValidationResult))
    by_shape: Dict[str, int] = {}
    for r in results:
        src = report_graph.value(r, SH.sourceShape)
        comp = report_graph.value(r, SH.sourceConstraintComponent)
        key = f"{str(comp).split('#')[-1]}"
        by_shape[key] = by_shape.get(key, 0) + 1
    return {"conforms": bool(conforms), "n_results": len(results), "by_component": by_shape, "seconds": round(dt, 2), "text": text[:2000] if not conforms else ""}


QB_IC = {
    "IC-1 unique data set": """PREFIX qb: <http://purl.org/linked-data/cube#>
ASK { { ?obs a qb:Observation . FILTER NOT EXISTS { ?obs qb:dataSet ?ds } } UNION
      { ?obs a qb:Observation ; qb:dataSet ?ds1, ?ds2 . FILTER (?ds1 != ?ds2) } }""",
    "IC-2 unique DSD": """PREFIX qb: <http://purl.org/linked-data/cube#>
ASK { { ?ds a qb:DataSet . FILTER NOT EXISTS { ?ds qb:structure ?dsd } } UNION
      { ?ds a qb:DataSet ; qb:structure ?d1, ?d2 . FILTER (?d1 != ?d2) } }""",
    "IC-3 DSD includes measure": """PREFIX qb: <http://purl.org/linked-data/cube#>
ASK { ?dsd a qb:DataStructureDefinition . FILTER NOT EXISTS { ?dsd qb:component [ qb:measure [] ] } }""",
    "IC-4 dimensions have range": """PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
ASK { ?dim a qb:DimensionProperty . FILTER NOT EXISTS { ?dim rdfs:range [] } }""",
    "IC-11 all dimensions required": """PREFIX qb: <http://purl.org/linked-data/cube#>
ASK { ?obs qb:dataSet/qb:structure/qb:component/qb:dimension ?dim .
      FILTER NOT EXISTS { ?obs ?dim [] } }""",
    "IC-12 no duplicate observations": """PREFIX dmi: <http://purl.org/dmi-framework#>
PREFIX qb: <http://purl.org/linked-data/cube#>
ASK { { SELECT ?i ?c ?y (COUNT(?o) AS ?n) WHERE { ?o qb:dataSet dmi:Dataset_panel ; dmi:entryIndicator ?i ; dmi:entryCountry ?c ; dmi:year ?y }
        GROUP BY ?i ?c ?y } FILTER (?n > 1) }""",
    "IC-14 all measures present": """PREFIX qb: <http://purl.org/linked-data/cube#>
ASK { ?obs qb:dataSet/qb:structure/qb:component/qb:measure ?m .
      FILTER NOT EXISTS { ?obs ?m [] } }""",
}


def qb_integrity(g: Graph) -> Dict[str, Any]:
    out = {}
    for name, q in QB_IC.items():
        t0 = time.perf_counter()
        violated = bool(g.query(q).askAnswer)
        out[name] = {"violated": violated, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    return out


# ---------------------------------------------------------------- faults
def _first(g: Graph, pred, obj=None):
    return next(iter(g.subjects(pred, obj)))


def faults(g: Graph) -> Dict[str, Tuple[str, Callable[[Graph], None]]]:
    e_mkd = DMI["SE_X2_MKD_2020"]
    e_ctx = DMI["SE_Z1_MKD_2020"]

    def dup(h: Graph):
        new = DMI["SE_X2_MKD_2020_dup"]
        for p, o in list(h.predicate_objects(e_mkd)):
            h.add((new, p, o))
        h.remove((new, DMI.matrixValue, None))
        h.add((new, DMI.matrixValue, dec(99.0)))

    def year_out(h: Graph):
        h.remove((e_mkd, DMI.year, None)); h.add((e_mkd, DMI.year, Literal(2031, datatype=XSD.integer)))

    def no_prov(h: Graph):
        h.remove((e_mkd, DMI.hasProvenance, None))

    def ctx_imputed(h: Graph):
        h.remove((e_ctx, DMI.hasProvenance, None)); h.add((e_ctx, DMI.hasProvenance, DMI.Prov_mice_imputed))
        h.add((e_ctx, DMI.hasReconstructionScope, DMI.Scope_interior))

    def ctx_in_pillar(h: Graph):
        h.add((DMI.Ind_Z4, DMI.belongsToPillar, DMI.Pillar_skills))

    def n_official(h: Graph):
        s = DMI["Series_X1_MKD"]
        h.remove((s, DMI.nOfficial, None)); h.add((s, DMI.nOfficial, Literal(3, datatype=XSD.integer)))

    def p_out(h: Graph):
        e = DMI["Est_mi_pooled_X2"]
        h.remove((e, DMI.pValue, None)); h.add((e, DMI.pValue, dec(1.7)))

    def two_types(h: Graph):
        h.add((DMI.Country_MKD, RDF.type, DMI.Indicator))

    def two_prov(h: Graph):
        h.add((e_mkd, DMI.hasProvenance, DMI.Prov_carried_forward))

    def wrong_series(h: Graph):
        h.remove((e_mkd, DMI.inSeries, None)); h.add((e_mkd, DMI.inSeries, DMI["Series_X2_SRB"]))

    return {
        "F1 duplicate cell with a different value": ("key", dup),
        "F2 year outside 2012-2025": ("range", year_out),
        "F3 cell without provenance": ("completeness", no_prov),
        "F4 context cell marked as reconstructed": ("policy", ctx_imputed),
        "F5 context indicator placed in a DMI pillar": ("policy", ctx_in_pillar),
        "F6 series count disagrees with its cells": ("aggregate", n_official),
        "F7 p-value outside [0, 1]": ("range", p_out),
        "F8 country also typed as indicator": ("typing", two_types),
        "F9 cell with two provenance states": ("functional", two_prov),
        "F10 cell linked to another country's series": ("cross-reference", wrong_series),
    }


def fault_injection(g: Graph, shapes_path: Path, workdir: Path) -> List[Dict[str, Any]]:
    rows = []
    for name, (kind, mutate) in faults(g).items():
        h = Graph()
        for t in g:
            h.add(t)
        mutate(h)
        owl = workdir / "fault.owl"
        h.serialize(owl, format="xml")
        r = hermit(owl)
        s = shacl(h, shapes_path)
        rows.append({"fault": name, "kind": kind, "hermit_detects": not r["consistent"], "shacl_detects": not s["conforms"],
                     "shacl_results": s["n_results"], "hermit_s": r["reason_s"], "shacl_s": s["seconds"]})
    return rows
