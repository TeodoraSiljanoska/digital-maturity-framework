"""Competency questions CQ1–CQ14 over the knowledge graph.

Each question has a SPARQL query (run by rdflib on the asserted graph) and an
expected answer computed independently from the tabular artifacts with pandas.
A question passes when both paths give the same answer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from rdflib import Graph
from rdflib.term import Literal

P = """PREFIX dmi: <http://purl.org/dmi-framework#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

CQS: Dict[str, Dict[str, str]] = {
    "CQ1": {"question": "By which collection method is each indicator obtained?",
            "sparql": P + "SELECT ?id ?m WHERE { ?i a dmi:Indicator ; dmi:indicatorId ?id ; dmi:obtainedBy ?mm . BIND(STRAFTER(STR(?mm), 'Method_') AS ?m) } ORDER BY ?id"},
    "CQ2": {"question": "Which provenance states do the cells of North Macedonia in 2025 have?",
            "sparql": P + "SELECT ?p (COUNT(?e) AS ?n) WHERE { ?e a dmi:ScoreEntry ; dmi:entryCountry dmi:Country_MKD ; dmi:year 2025 ; dmi:hasProvenance ?pp . BIND(STRAFTER(STR(?pp), 'Prov_') AS ?p) } GROUP BY ?p ORDER BY ?p"},
    "CQ3": {"question": "In which process phase is each forecasting model assigned?",
            "sparql": P + "SELECT ?m ?ph WHERE { ?f a dmi:ForecastScore ; dmi:scoredModel ?mm . ?mm dmi:assignedToPhase ?p ; rdfs:label ?m . ?p rdfs:label ?ph } GROUP BY ?m ?ph ORDER BY ?m"},
    "CQ4": {"question": "Which five indicators have the highest mean |SHAP|, and for which model?",
            "sparql": P + "SELECT ?id ?model WHERE { ?e a dmi:Explanation ; dmi:aboutIndicator ?i ; dmi:meanAbsShap ?s ; dmi:explainsOutput ?m . ?i dmi:indicatorId ?id . ?m rdfs:label ?model } ORDER BY DESC(?s) LIMIT 5"},
    "CQ5": {"question": "What are the values, provenance, edition and source of the cells of North Macedonia in 2025?",
            "sparql": P + "SELECT ?id ?v ?p WHERE { ?e a dmi:ScoreEntry ; dmi:entryCountry dmi:Country_MKD ; dmi:year 2025 ; dmi:entryIndicator ?i ; dmi:matrixValue ?v ; dmi:hasProvenance ?pp ; dmi:entrySource ?s . ?i dmi:indicatorId ?id . BIND(STRAFTER(STR(?pp), 'Prov_') AS ?p) } ORDER BY ?id"},
    "CQ6": {"question": "How many countries, indicators and cells does the graph hold?",
            "sparql": P + "SELECT (COUNT(DISTINCT ?c) AS ?nc) (COUNT(DISTINCT ?i) AS ?ni) WHERE { ?c a dmi:Country . ?i a dmi:Indicator }"},
    "CQ7": {"question": "Which country series have no published value at all?",
            "sparql": P + "SELECT ?id ?iso WHERE { ?s a dmi:TimeSeries ; dmi:nOfficial 0 ; dmi:seriesIndicator/dmi:indicatorId ?id ; dmi:seriesCountry/dmi:iso3 ?iso } ORDER BY ?id ?iso"},
    "CQ8": {"question": "What share of the composite-component cells of the Balkan group is published in each year?",
            "sparql": P + """SELECT ?y (SUM(IF(?p = dmi:Prov_official, 1, 0)) AS ?off) (COUNT(?e) AS ?n) WHERE {
  ?e a dmi:ScoreEntry ; dmi:year ?y ; dmi:hasProvenance ?p ; dmi:entryCountry/dmi:memberOf dmi:Group_balkan ;
     dmi:entryIndicator/dmi:belongsToPillar ?pl .
  VALUES ?pl { dmi:Pillar_infrastructure dmi:Pillar_e_government dmi:Pillar_skills dmi:Pillar_trust_innovation dmi:Pillar_digital_economy }
} GROUP BY ?y ORDER BY ?y"""},
    "CQ9": {"question": "What is the latest published value of each context indicator for North Macedonia, and to which pillar is it related?",
            "sparql": P + """SELECT ?id ?pillar ?y ?v WHERE {
  ?i a dmi:ContextIndicator ; dmi:indicatorId ?id ; dmi:relatedPillar ?pp .
  BIND(STRAFTER(STR(?pp), 'Pillar_') AS ?pillar)
  ?e dmi:entryIndicator ?i ; dmi:entryCountry dmi:Country_MKD ; dmi:hasProvenance dmi:Prov_official ; dmi:year ?y ; dmi:matrixValue ?v .
  FILTER NOT EXISTS { ?e2 dmi:entryIndicator ?i ; dmi:entryCountry dmi:Country_MKD ; dmi:hasProvenance dmi:Prov_official ; dmi:year ?y2 . FILTER (?y2 > ?y) }
} ORDER BY ?id"""},
    "CQ10": {"question": "Which lagged indicators are significant at 5% in the single-imputation and in the pooled fixed-effects estimates?",
             "sparql": P + "SELECT ?arm ?id WHERE { ?e a dmi:Estimate ; dmi:estimationArm ?arm ; dmi:pValue ?p ; dmi:estimateFor/dmi:indicatorId ?id . FILTER (?p < 0.05) } ORDER BY ?arm ?id"},
    "CQ11": {"question": "How did the cross-country coefficient of variation of the DMI change between 2012 and 2025?",
             "sparql": P + "SELECT ?y ?cv WHERE { ?d a dmi:DispersionReading ; dmi:year ?y ; dmi:coefficientOfVariation ?cv . FILTER (?y IN (2012, 2025)) } ORDER BY ?y"},
    "CQ12": {"question": "Which models beat the persistence forecast with a bootstrap interval entirely below zero?",
             "sparql": P + "SELECT ?t ?m WHERE { ?f a dmi:ForecastScore ; dmi:forecastTarget ?t ; dmi:ciHigh ?hi ; dmi:scoredModel/rdfs:label ?m . FILTER (?hi < 0) } ORDER BY ?t ?m"},
    "CQ13": {"question": "From which editions were the published values of the AI Readiness Index read?",
             "sparql": P + "SELECT ?ed (COUNT(?e) AS ?n) WHERE { ?e dmi:entryIndicator dmi:Ind_X7 ; dmi:hasProvenance dmi:Prov_official ; dmi:edition ?ed } GROUP BY ?ed ORDER BY ?ed"},
    "CQ14": {"question": "Which analytic views show context indicators, and which slicer fields do they offer?",
             "sparql": P + "SELECT DISTINCT ?view ?field WHERE { ?v a dmi:AnalyticView ; dmi:showsIndicator ?i ; dmi:slicerField ?field ; rdfs:label ?view . ?i a dmi:ContextIndicator . FILTER (LANG(?view) = 'en') } ORDER BY ?view ?field"},
}


def _rows(g: Graph, q: str) -> List[Tuple]:
    out = []
    for r in g.query(q):
        out.append(tuple(v.toPython() if isinstance(v, Literal) else str(v) for v in r))
    return out


def _norm(rows: List[Tuple], nd: int = 6) -> List[Tuple]:
    def f(x):
        if isinstance(x, float):
            return round(x, nd)
        try:
            from decimal import Decimal
            if isinstance(x, Decimal):
                return round(float(x), nd)
        except Exception:  # noqa: BLE001
            pass
        return x
    return [tuple(f(x) for x in r) for r in rows]


def expected(root: Path, cells: pd.DataFrame, series: pd.DataFrame, scores: pd.DataFrame, views: Dict[str, Any]) -> Dict[str, List[Tuple]]:
    import yaml

    ind = yaml.safe_load((root / "config/indicators.yaml").read_text(encoding="utf-8"))["indicators"]
    exp: Dict[str, List[Tuple]] = {}
    m = {r["indicator_id"]: ("official_api" if r["collection_method"] in ("api", "official_api") else r["collection_method"]) for r in ind}
    for z in [f"Z{i}" for i in range(1, 7)]:
        m[z] = "official_api"
    exp["CQ1"] = sorted(m.items())
    mk = cells[(cells.country_iso3 == "MKD") & (cells.year == 2025)]
    exp["CQ2"] = sorted(mk["provenance"].value_counts().items())
    models = sorted(scores["model"].unique())
    exp["CQ3"] = [(mdl, "Predict") for mdl in models]
    shap = pd.read_csv(root / "results/xai/shap_global_importance.csv").sort_values("mean_abs_shap", ascending=False).head(5)
    import json
    meta = json.loads((root / "results/xai/xai_shap_global.meta.json").read_text(encoding="utf-8"))
    model = str(meta["meta"]["model_id"]).split("_20")[0]
    exp["CQ4"] = [(f.replace("_lag1", ""), model) for f in shap["feature"]]
    exp["CQ5"] = [(r.indicator_id, round(float(r.value), 6), r.provenance) for r in mk.sort_values("indicator_id").itertuples()]
    exp["CQ6"] = [(12, int(cells["indicator_id"].nunique()))]
    s0 = series[series["n_official"] == 0].sort_values(["indicator_id", "country_iso3"])
    exp["CQ7"] = [(r.indicator_id, r.country_iso3) for r in s0.itertuples()]
    bal = cells[cells.country_iso3.isin(["MKD", "SRB", "SVN"]) & cells.indicator_id.isin([f"X{i}" for i in range(1, 11)])]
    exp["CQ8"] = [(int(y), int((g.provenance == "official").sum()), int(len(g))) for y, g in bal.groupby("year")]
    zc = yaml.safe_load((root / "config/context_indicators.yaml").read_text(encoding="utf-8"))["indicators"]
    rows = []
    for z, spec in zc.items():
        f = cells[(cells.indicator_id == z) & (cells.country_iso3 == "MKD") & (cells.provenance == "official")]
        if len(f):
            last = f.sort_values("year").iloc[-1]
            rows.append((z, spec["related_pillar"], int(last.year), round(float(last.value), 6)))
    exp["CQ9"] = sorted(rows)
    fe = pd.read_csv(root / "results/econometrics/coefficients.csv")
    fe = fe[(fe.model == "FE") & fe.term.str.endswith("_lag1") & (fe.pvalue < 0.05)]
    mi = pd.read_csv(root / "results/imputation/mi_pooled_fe.csv")
    mi = mi[mi.term.str.endswith("_lag1") & (mi.pvalue < 0.05)]
    ctx = pd.read_csv(root / "outputs/knowledge_layer/association/context_association.csv")
    ctx = ctx[ctx.p_driscoll_kraay < 0.05]
    exp["CQ10"] = sorted([("single_imputation", t.replace("_lag1", "")) for t in fe.term]
                         + [("mi_pooled", t.replace("_lag1", "")) for t in mi.term]
                         + [("context_official_only", z) for z in ctx.indicator_id])
    sig = pd.read_csv(root / "results/convergence/sigma_convergence.csv")
    sig = sig[(sig["sample"] == "full") & sig.year.isin([2012, 2025])].sort_values("year")
    exp["CQ11"] = [(int(r.year), round(float(r.cv), 6)) for r in sig.itertuples()]
    b = scores[scores.delta_ci_high < 0].sort_values(["target", "model"])
    exp["CQ12"] = [(r.target, r.model) for r in b.itertuples()]
    x7 = cells[(cells.indicator_id == "X7") & (cells.provenance == "official")]
    exp["CQ13"] = sorted((str(k), int(v)) for k, v in x7["edition"].value_counts().items())
    rows = []
    for vid, spec in views["views"].items():
        if any(i.startswith("Z") for i in spec["indicators"]):
            rows += [(spec["title"], s) for s in spec["slicers"]]
    exp["CQ14"] = sorted(set(rows))
    return exp


def run_all(g: Graph, exp: Dict[str, List[Tuple]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cid, spec in CQS.items():
        t0 = time.perf_counter()
        got = _rows(g, spec["sparql"])
        ms = (time.perf_counter() - t0) * 1000
        gotn = _norm(got)
        expn = _norm(exp[cid])
        if cid in ("CQ1", "CQ2", "CQ7", "CQ9", "CQ10", "CQ12", "CQ13", "CQ14"):
            ok = sorted(gotn) == sorted(expn)
        else:
            ok = gotn == expn
        out[cid] = {"question": spec["question"], "passed": bool(ok), "n_rows": len(got), "ms": round(ms, 1),
                    "answer": [list(map(str, r)) for r in got[:20]]}
        if not ok:
            out[cid]["expected"] = [list(map(str, r)) for r in expn[:20]]
    return out
