"""Competency-question answers over the materialized ontology graph (Python, not a reasoner)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

MODEL_PHASE_MAP: Dict[str, Dict[str, str]] = {
    "world_bank_api": {
        "phase": "acquire",
        "purpose": "Obtain official World Bank WDI series via HTTP API",
    },
    "curated_snapshot_loaders": {
        "phase": "acquire",
        "purpose": "Load versioned UN EGDI, NCSI, GII, Oxford, UNDP snapshots",
    },
    "mice": {
        "phase": "process",
        "purpose": "Complete remaining missing cells; never overwrite official values",
    },
    "dmi_weighted_pillars": {
        "phase": "index",
        "purpose": "Construct dependent variable Y",
    },
    "ols": {"phase": "predict", "purpose": "Linear baseline for H1.6"},
    "ridge": {"phase": "predict", "purpose": "Regularised linear baseline for H1.6"},
    "linear_regression": {"phase": "predict", "purpose": "Linear baseline for H1.6"},
    "random_forest": {"phase": "predict", "purpose": "Non-linear prediction of DMI"},
    "xgboost": {"phase": "predict", "purpose": "Non-linear prediction of DMI"},
    "lightgbm": {"phase": "predict", "purpose": "Non-linear prediction of DMI"},
    "catboost": {"phase": "predict", "purpose": "Non-linear prediction of DMI"},
    "svr": {"phase": "predict", "purpose": "Non-linear prediction of DMI"},
    "mlp": {"phase": "predict", "purpose": "Non-linear prediction of DMI (unstable on this n)"},
    "shap": {"phase": "explain", "purpose": "Global and local factor attribution"},
    "lime": {"phase": "explain", "purpose": "Local model-agnostic explanation"},
    "pdp": {"phase": "explain", "purpose": "Marginal response of predicted DMI"},
    "streamlit": {"phase": "visualize", "purpose": "Interactive VDA by country/indicator/phase"},
    "power_bi": {"phase": "visualize", "purpose": "Star-schema VDA surface"},
}


def answer_cq(
    cq_id: str,
    graph: Dict[str, Any],
    scores: Optional[pd.DataFrame] = None,
    shap: Optional[List[Dict[str, Any]]] = None,
    *,
    iso3: str = "MKD",
    year: Optional[int] = None,
) -> Dict[str, Any]:
    indicators = graph.get("indicators") or []
    countries = graph.get("countries") or []
    shap = shap if shap is not None else graph.get("explanations") or []

    if cq_id in ("CQ1", "indicators_by_collection_method"):
        by_method: Dict[str, List[str]] = {}
        for ind in indicators:
            by_method.setdefault(str(ind.get("collection_method")), []).append(
                str(ind.get("indicator_id"))
            )
        return {
            "id": "CQ1",
            "answer": by_method,
            "passed": bool(indicators) and all("uri" in i for i in indicators),
        }

    if cq_id in ("CQ2", "provenance_for_country_year"):
        if scores is None or scores.empty:
            return {"id": "CQ2", "answer": {}, "passed": False}
        y = int(year if year is not None else scores["year"].max())
        sub = scores[(scores["country_iso3"] == iso3) & (scores["year"] == y)]
        counts = sub.groupby("provenance").size().to_dict() if not sub.empty else {}
        return {
            "id": "CQ2",
            "answer": {"country_iso3": iso3, "year": y, "counts": {str(k): int(v) for k, v in counts.items()}},
            "passed": not sub.empty,
        }

    if cq_id in ("CQ3", "model_to_phase"):
        cat = MODEL_PHASE_MAP["catboost"]
        return {
            "id": "CQ3",
            "answer": {"model": "catboost", **cat},
            "passed": cat["phase"] == "predict",
        }

    if cq_id in ("CQ4", "top_shap_features"):
        top = [
            {"indicator_id": s.get("indicator_id"), "mean_abs_shap": s.get("mean_abs_shap")}
            for s in shap[:10]
        ]
        return {"id": "CQ4", "answer": top, "passed": bool(top)}

    if cq_id in ("CQ5", "score_entries_country_latest"):
        if scores is None or scores.empty:
            return {"id": "CQ5", "answer": [], "passed": False}
        y = int(year if year is not None else scores["year"].max())
        sub = scores[(scores["country_iso3"] == iso3) & (scores["year"] == y)]
        cols = [
            c
            for c in ("indicator_id", "value", "provenance", "source_id", "phase_id")
            if c in sub.columns
        ]
        records = sub[cols].to_dict("records")
        return {"id": "CQ5", "answer": records, "passed": len(records) >= 12}

    if cq_id in ("CQ6", "coverage_countries_indicators"):
        acquired = [i for i in indicators if i.get("indicator_id") != "DMI"]
        n_c = len(countries)
        n_i = len(acquired)
        return {
            "id": "CQ6",
            "answer": {"n_countries": n_c, "n_acquired_indicators": n_i, "n_all_indicators": len(indicators)},
            "passed": n_c == 12 and n_i >= 12,
        }

    raise KeyError(f"Unknown competency question: {cq_id}")


def answer_all(
    graph: Dict[str, Any],
    scores: Optional[pd.DataFrame] = None,
    shap: Optional[List[Dict[str, Any]]] = None,
    *,
    default_iso3: str = "MKD",
    default_year: Optional[int] = None,
) -> Dict[str, Any]:
    out = {}
    for cq in ("CQ1", "CQ2", "CQ3", "CQ4", "CQ5", "CQ6"):
        out[cq] = answer_cq(
            cq, graph, scores, shap, iso3=default_iso3, year=default_year
        )
    return out
