"""SPARQL competency questions over the populated Turtle graph (rdflib, not a DL reasoner)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

NS = "http://purl.org/dmi-framework#"

QUERIES: Dict[str, str] = {
    "CQ1": """
PREFIX dmi: <http://purl.org/dmi-framework#>
SELECT ?id ?method WHERE {
  ?ind a dmi:Indicator ;
       dmi:indicatorId ?id ;
       dmi:obtainedBy ?m .
  ?m rdfs:label ?method .
}
ORDER BY ?method ?id
""",
    "CQ2": """
PREFIX dmi: <http://purl.org/dmi-framework#>
SELECT ?prov (COUNT(*) AS ?n) WHERE {
  ?se a dmi:ScoreEntry ;
      dmi:entryCountry dmi:Country_MKD ;
      dmi:year 2025 ;
      dmi:hasProvenance ?p .
  ?p rdfs:label ?prov .
}
GROUP BY ?prov
""",
    "CQ3": """
PREFIX dmi: <http://purl.org/dmi-framework#>
SELECT ?phase ?purpose WHERE {
  dmi:Model_catboost dmi:assignedToPhase ?ph ;
                     rdfs:comment ?purpose .
  ?ph rdfs:label ?phase .
}
""",
    "CQ4": """
PREFIX dmi: <http://purl.org/dmi-framework#>
SELECT ?id ?shap WHERE {
  ?e a dmi:Explanation ;
     dmi:aboutIndicator ?ind ;
     dmi:meanAbsShap ?shap .
  ?ind dmi:indicatorId ?id .
}
ORDER BY DESC(?shap)
LIMIT 10
""",
    "CQ5": """
PREFIX dmi: <http://purl.org/dmi-framework#>
SELECT ?id ?value ?prov ?phase ?source WHERE {
  ?se a dmi:ScoreEntry ;
      dmi:entryCountry dmi:Country_MKD ;
      dmi:year 2025 ;
      dmi:entryIndicator ?ind ;
      dmi:matrixValue ?value ;
      dmi:hasProvenance ?p ;
      dmi:entryPhase ?ph ;
      dmi:entrySource ?src .
  ?ind dmi:indicatorId ?id .
  ?p rdfs:label ?prov .
  ?ph rdfs:label ?phase .
  ?src rdfs:label ?source .
}
ORDER BY ?id
""",
    "CQ6": """
PREFIX dmi: <http://purl.org/dmi-framework#>
SELECT (COUNT(DISTINCT ?c) AS ?n_countries) (COUNT(DISTINCT ?ind) AS ?n_indicators) WHERE {
  ?c a dmi:Country .
  ?ind a dmi:Indicator .
}
""",
}


def _bind_prefixes(query: str) -> str:
    if "PREFIX rdfs:" in query:
        return query
    return "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" + query


def load_graph(ttl_path: Path):
    from rdflib import Graph

    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def run_query(graph, cq_id: str) -> List[Dict[str, Any]]:
    from rdflib.term import Literal

    q = QUERIES.get(cq_id)
    if not q:
        raise KeyError(cq_id)
    rows = []
    results = graph.query(_bind_prefixes(q))
    vars_ = [str(v) for v in (results.vars or [])]
    for result in results:
        item: Dict[str, Any] = {}
        for name in vars_:
            v = result[name]
            if v is None:
                continue
            item[name] = v.toPython() if isinstance(v, Literal) else str(v)
        rows.append(item)
    return rows


def structural_consistency(graph) -> Dict[str, Any]:
    """RDFS-plus structural audit. This is not HermiT OWL DL reasoning."""
    from rdflib import RDF, URIRef

    ns = NS
    n_ind = len(list(graph.subjects(RDF.type, URIRef(ns + "Indicator"))))
    n_country = len(list(graph.subjects(RDF.type, URIRef(ns + "Country"))))
    n_se = len(list(graph.subjects(RDF.type, URIRef(ns + "ScoreEntry"))))
    missing_method = []
    for ind in graph.subjects(RDF.type, URIRef(ns + "Indicator")):
        if not list(graph.objects(ind, URIRef(ns + "obtainedBy"))):
            missing_method.append(str(ind))
    return {
        "reasoner": "none",
        "note": "HermiT is not run. The TBox is RDFS-plus; this check is structural.",
        "n_indicators": n_ind,
        "n_countries": n_country,
        "n_score_entries": n_se,
        "indicators_missing_obtainedBy": missing_method,
        "passed": n_ind >= 13 and n_country == 12 and not missing_method,
    }


def answer_sparql_cqs(ttl_path: Path) -> Dict[str, Any]:
    graph = load_graph(ttl_path)
    out: Dict[str, Any] = {"consistency": structural_consistency(graph)}
    for cq_id in ("CQ1", "CQ2", "CQ3", "CQ4", "CQ5", "CQ6"):
        rows = run_query(graph, cq_id)
        passed = bool(rows)
        if cq_id == "CQ3":
            passed = passed and any(
                str(r.get("phase", "")).lower() == "predict" for r in rows
            )
        if cq_id == "CQ6" and rows:
            n_c = int(rows[0].get("n_countries") or 0)
            n_i = int(rows[0].get("n_indicators") or 0)
            passed = n_c == 12 and n_i >= 13
        out[cq_id] = {"id": cq_id, "answer": rows, "passed": passed, "engine": "rdflib SPARQL"}
    return out


def write_query_files(directory: Path) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for cq_id, q in QUERIES.items():
        path = directory / f"{cq_id}.rq"
        path.write_text(_bind_prefixes(q).strip() + "\n", encoding="utf-8")
        written.append(str(path))
    return written
