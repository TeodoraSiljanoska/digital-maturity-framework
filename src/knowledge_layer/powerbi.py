"""Power BI project (PBIP) generated from the knowledge layer.

The star schema is written as CSV files; the semantic model is written in TMDL
and the report in the PBIR format, so the whole project is plain text that can
be reviewed and versioned. Pages, slicers and the indicators each page shows
come from config/vda_views.yaml, the same file that defines the
dmi:AnalyticView individuals of the knowledge graph.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

NAME = "DMI_VDA"
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
GROUP_ORDER = {"balkan": 1, "developed_eu": 2, "developed_non_eu": 3, "developing": 4}


def _guid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"dmi-vda/{seed}"))


def _vid(seed: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"dmi-vda-visual/{seed}").hex[:20]


# --------------------------------------------------------------- star schema
def star_schema(root: Path, cells: pd.DataFrame, out: Path) -> Dict[str, pd.DataFrame]:
    from knowledge_layer.context import cell_panel, load_config, load_vintage
    from knowledge_layer.graph import NAME_MK, PILLAR_MK, PRIMARY_PUBLISHER, PUBLISHERS, SOURCES

    out.mkdir(parents=True, exist_ok=True)
    zcfg = load_config(root)
    catalog = yaml.safe_load((root / "config/indicators.yaml").read_text(encoding="utf-8"))["indicators"]
    countries = yaml.safe_load((root / "config/countries.yaml").read_text(encoding="utf-8"))["groups"]

    dim_country = pd.DataFrame([
        {"country_iso3": c["iso3"], "country_name": c["name"], "country_name_mk": c["name_mk"],
         "group_id": gid, "group_name": g["name"], "group_name_mk": g["name_mk"], "group_order": GROUP_ORDER[gid]}
        for gid, g in countries.items() for c in g["countries"]
    ])
    rows = []
    for i, r in enumerate(catalog):
        iid = r["indicator_id"]
        fam = "Core (X)" if iid.startswith("X") else "Control (C)" if iid.startswith("C") else "Composite (DMI)"
        rows.append({"indicator_id": iid, "indicator_label": f"{iid} {r['name']}", "name_mk": NAME_MK[iid],
                     "family": fam, "pillar": PILLAR_MK[r["pillar"]][0], "pillar_mk": PILLAR_MK[r["pillar"]][1],
                     "in_composite": iid.startswith("X"), "unit": str(r.get("unit")), "source_id": r["source"],
                     "primary_publisher": PUBLISHERS.get(PRIMARY_PUBLISHER.get(iid, ""), "Digital Maturity Framework"),
                     "sort_order": i + 1})
    for j, (z, s) in enumerate(zcfg["indicators"].items()):
        rows.append({"indicator_id": z, "indicator_label": f"{z} {s['name']}", "name_mk": s["name_mk"],
                     "family": "Context (Z)", "pillar": PILLAR_MK[s["related_pillar"]][0], "pillar_mk": PILLAR_MK[s["related_pillar"]][1],
                     "in_composite": False, "unit": s["unit"], "source_id": "world_bank_context",
                     "primary_publisher": PUBLISHERS[PRIMARY_PUBLISHER[z]], "sort_order": 100 + j})
    dim_indicator = pd.DataFrame(rows)
    dim_year = pd.DataFrame({"year": list(range(2012, 2026))})
    dim_provenance = pd.DataFrame([
        {"state": "official", "label": "Published", "label_mk": "Објавена", "sort": 1},
        {"state": "carried_forward", "label": "Carried (<= 2 years)", "label_mk": "Пренесена", "sort": 2},
        {"state": "mice_imputed", "label": "Reconstructed (MICE)", "label_mk": "Реконструирана", "sort": 3},
        {"state": "constructed", "label": "Constructed (DMI)", "label_mk": "Конструирана", "sort": 4},
        {"state": "missing", "label": "Missing", "label_mk": "Недостасува", "sort": 5},
    ])
    dim_source = pd.DataFrame([
        {"source_id": sid, "source_label": lab, "publisher": PUBLISHERS.get(pub, "Digital Maturity Framework") if pub else "Digital Maturity Framework",
         "retrieved_at": vint or ""}
        for sid, (lab, pub, vint, _m) in SOURCES.items()
    ])

    fact = cells.copy()
    # context gaps are kept as rows without a value so that coverage views can show them
    zlong = load_vintage(root, zcfg)
    _zf, zprov = cell_panel(zlong, zcfg)
    gaps = zprov.melt(id_vars=["country_iso3", "year"], var_name="indicator_id", value_name="provenance")
    gaps = gaps[gaps["provenance"] == "missing"].assign(value=np.nan, scope=None, edition=None, method=None,
                                                        verification=None, source_id="world_bank_context", phase="Acquire")
    fact = pd.concat([fact, gaps[fact.columns]], ignore_index=True)
    lo = fact.groupby("indicator_id")["value"].transform("min")
    hi = fact.groupby("indicator_id")["value"].transform("max")
    fact["value_norm"] = (fact["value"] - lo) / (hi - lo) * 100.0
    fact["cell_iri"] = "http://purl.org/dmi-framework#SE_" + fact["indicator_id"] + "_" + fact["country_iso3"] + "_" + fact["year"].astype(int).astype(str)
    fact = fact.rename(columns={"method": "retrieval_method", "scope": "reconstruction_scope"})
    fact = fact[["cell_iri", "indicator_id", "country_iso3", "year", "value", "value_norm", "provenance",
                 "reconstruction_scope", "edition", "retrieval_method", "source_id"]]

    analysis = pd.read_parquet(root / "data/processed/analysis_panel.parquet")
    v3b = pd.read_parquet(root / "versions/v3b_official_only/data_processed/dmi_panel.parquet")[["country_iso3", "year", "DMI"]].rename(columns={"DMI": "DMI_official_only"})
    plus = pd.read_csv(root / "outputs/knowledge_layer/association/dmi_plus.csv")[["country_iso3", "year", "DMI_plus"]]
    fact_dmi = analysis[["country_iso3", "year", "DMI", "pillar_infrastructure", "pillar_e_government", "pillar_skills",
                         "pillar_trust_innovation", "pillar_digital_economy"]].merge(v3b, how="left").merge(plus, how="left")

    fe = pd.read_csv(root / "results/econometrics/coefficients.csv")
    fe = fe[(fe["model"] == "FE") & fe["term"].str.endswith("_lag1")].assign(arm="FE, single imputation")
    mi = pd.read_csv(root / "results/imputation/mi_pooled_fe.csv")
    mi = mi[mi["term"].str.endswith("_lag1")].assign(arm="FE, pooled over 10 imputations")
    est = pd.concat([fe[["term", "coefficient", "std_error", "pvalue", "arm"]], mi[["term", "coefficient", "std_error", "pvalue", "arm"]]])
    est["indicator_id"] = est["term"].str.replace("_lag1", "", regex=False)
    ctx = pd.read_csv(root / "outputs/knowledge_layer/association/context_association.csv")
    ctx = ctx.rename(columns={"beta_per_sd": "coefficient", "se_driscoll_kraay": "std_error", "p_driscoll_kraay": "pvalue"})
    ctx["arm"] = "Two-way FE, context (per SD)"
    est = pd.concat([est[["indicator_id", "arm", "coefficient", "std_error", "pvalue"]], ctx[["indicator_id", "arm", "coefficient", "std_error", "pvalue"]]])
    est["significant_5pct"] = est["pvalue"] < 0.05
    est["coefficient_txt"] = est["coefficient"].map(lambda v: f"{v:.3f}")
    est["p_txt"] = est["pvalue"].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")

    fc = pd.read_csv(root / "outputs/knowledge_layer/timeseries/rolling_origin_scores.csv")
    fc["target"] = fc["target"].map({"DMI_mice": "DMI (reconstructed panel)", "DMI_official_only": "DMI (published cells only)"})
    fc = fc.rename(columns={"delta_rmse_vs_persistence": "delta_rmse", "delta_ci_low": "ci_low", "delta_ci_high": "ci_high"})
    for c in ("rmse", "mae", "delta_rmse"):
        fc[c + "_txt"] = fc[c].map(lambda v: f"{v:.3f}")
    fc["ci_txt"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(fc["ci_low"], fc["ci_high"])]
    shap = pd.read_csv(root / "results/xai/shap_global_importance.csv")
    shap["indicator_id"] = shap["feature"].str.replace("_lag1", "", regex=False)
    shap = shap[["indicator_id", "mean_abs_shap"]]
    series = pd.concat([pd.read_csv(root / "outputs/knowledge_layer/timeseries/core_series.csv"),
                        pd.read_csv(root / "outputs/knowledge_layer/context/context_series.csv")], ignore_index=True)

    tables = {
        "dim_country": dim_country, "dim_indicator": dim_indicator, "dim_year": dim_year,
        "dim_provenance": dim_provenance, "dim_source": dim_source, "fact_cell": fact, "fact_dmi": fact_dmi,
        "fact_estimate": est, "fact_forecast": fc[["target", "model", "n_forecasts", "rmse", "mae", "delta_rmse", "ci_low", "ci_high", "origins_beating_persistence",
                              "rmse_txt", "mae_txt", "delta_rmse_txt", "ci_txt"]],
        "fact_shap": shap, "fact_series": series,
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    return tables


# ------------------------------------------------------------------- TMDL
def _tmdl_type(s: pd.Series) -> Tuple[str, str]:
    if pd.api.types.is_bool_dtype(s):
        return "boolean", "type logical"
    if pd.api.types.is_integer_dtype(s):
        return "int64", "Int64.Type"
    if pd.api.types.is_float_dtype(s):
        return "double", "type number"
    return "string", "type text"


def _q(name: str) -> str:
    return f"'{name}'" if any(c in name for c in " .=:'") else name


MEASURES: Dict[str, List[Tuple[str, str, str]]] = {
    "fact_cell": [
        ("Cells", "COUNTROWS(fact_cell)", "0"),
        ("Published cells", 'CALCULATE(COUNTROWS(fact_cell), fact_cell[provenance] = "official") + 0', "0"),
        ("Published share", "DIVIDE([Published cells], [Cells])", "0.0%"),
        ("Mean scaled value", "AVERAGE(fact_cell[value_norm])", "0.0"),
        ("Published share (en)", 'IF(ISBLANK([Published share]), BLANK(), FORMAT([Published share], "0.0%", "en-US"))', ""),
    ],
    "fact_dmi": [
        ("Average DMI", "AVERAGE(fact_dmi[DMI])", "0.00"),
        ("Average DMI (published cells only)", "AVERAGE(fact_dmi[DMI_official_only])", "0.00"),
        ("Average DMI+", "AVERAGE(fact_dmi[DMI_plus])", "0.00"),
        ("Average DMI (en)", 'IF(ISBLANK([Average DMI]), BLANK(), FORMAT([Average DMI], "0.00", "en-US"))', ""),
        ("Average DMI, published cells only (en)", 'IF(ISBLANK([Average DMI (published cells only)]), BLANK(), FORMAT([Average DMI (published cells only)], "0.00", "en-US"))', ""),
    ],
    "fact_forecast": [
        ("Mean RMSE", "AVERAGE(fact_forecast[rmse])", "0.000"),
        ("Mean RMSE change vs persistence", "AVERAGE(fact_forecast[delta_rmse])", "0.000"),
    ],
    "fact_shap": [("Mean |SHAP|", "SUM(fact_shap[mean_abs_shap])", "0.000")],
}
RELATIONSHIPS = [
    ("fact_cell", "indicator_id", "dim_indicator", "indicator_id"),
    ("fact_cell", "country_iso3", "dim_country", "country_iso3"),
    ("fact_cell", "year", "dim_year", "year"),
    ("fact_cell", "provenance", "dim_provenance", "state"),
    ("fact_cell", "source_id", "dim_source", "source_id"),
    ("fact_dmi", "country_iso3", "dim_country", "country_iso3"),
    ("fact_dmi", "year", "dim_year", "year"),
    ("fact_estimate", "indicator_id", "dim_indicator", "indicator_id"),
    ("fact_shap", "indicator_id", "dim_indicator", "indicator_id"),
    ("fact_series", "indicator_id", "dim_indicator", "indicator_id"),
    ("fact_series", "country_iso3", "dim_country", "country_iso3"),
]
SORT_BY = {("dim_country", "group_name"): "group_order", ("dim_indicator", "indicator_label"): "sort_order",
           ("dim_provenance", "label"): "sort"}


def write_semantic_model(proj: Path, tables: Dict[str, pd.DataFrame], data_dir: Path) -> None:
    sm = proj / f"{NAME}.SemanticModel"
    d = sm / "definition"
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (sm / "definition.pbism").write_text(json.dumps({
        "$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json", "version": "4.2", "settings": {}}, indent=2), encoding="utf-8")
    (sm / ".platform").write_text(json.dumps({
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "SemanticModel", "displayName": NAME},
        "config": {"version": "2.0", "logicalId": _guid("semantic-model")}}, indent=2), encoding="utf-8")
    (d / "database.tmdl").write_text("database\n\tcompatibilityLevel: 1600\n\n", encoding="utf-8")
    model = ["model Model", "\tculture: en-US", "\tdefaultPowerBIDataSourceVersion: powerBI_V3", "\tdiscourageImplicitMeasures",
             "\tsourceQueryCulture: en-US", "\tdataAccessOptions", "\t\tlegacyRedirects", "\t\treturnErrorValuesAsNull", "",
             "annotation __PBI_TimeIntelligenceEnabled = 0", ""]
    model += [f"ref table {_q(t)}" for t in tables]
    (d / "model.tmdl").write_text("\n".join(model) + "\n", encoding="utf-8")
    folder = str(data_dir.resolve()).rstrip("\\") + "\\"
    (d / "expressions.tmdl").write_text(
        f'expression DataFolder = "{folder}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n'
        f"\tlineageTag: {_guid('param-DataFolder')}\n\n\tannotation PBI_ResultType = Text\n\n", encoding="utf-8")
    for tname, df in tables.items():
        lines = [f"table {_q(tname)}", f"\tlineageTag: {_guid('t-' + tname)}", ""]
        for mname, expr, fmt in MEASURES.get(tname, []):
            lines += [f"\tmeasure {_q(mname)} = {expr}"] + ([f"\t\tformatString: {fmt}"] if fmt else [])
            lines += [f"\t\tlineageTag: {_guid('m-' + tname + mname)}", ""]
        types = []
        for col in df.columns:
            dt, mt = _tmdl_type(df[col])
            types.append(f'{{"{col}", {mt}}}')
            summ = "none" if dt in ("string", "boolean") or col in ("year", "sort", "sort_order", "group_order") else "sum"
            lines += [f"\tcolumn {_q(col)}", f"\t\tdataType: {dt}"]
            if dt == "double":
                lines.append("\t\tformatString: 0.000")
            lines += [f"\t\tlineageTag: {_guid('c-' + tname + col)}", f"\t\tsummarizeBy: {summ}", f"\t\tsourceColumn: {col}"]
            if (tname, col) in SORT_BY:
                lines.append(f"\t\tsortByColumn: {SORT_BY[(tname, col)]}")
            lines += ["", "\t\tannotation SummarizationSetBy = Automatic", ""]
        m = (
            "let\n"
            f'    Source = Csv.Document(File.Contents(DataFolder & "{tname}.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n'
            "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),\n"
            f'    Typed = Table.TransformColumnTypes(Promoted, {{{", ".join(types)}}}, "en-US")\n'
            "in\n    Typed"
        )
        lines += [f"\tpartition {_q(tname)} = m", "\t\tmode: import", "\t\tsource ="]
        lines += ["\t\t\t\t" + ln for ln in m.split("\n")]
        lines += ["", "\tannotation PBI_ResultType = Table", ""]
        (d / "tables" / f"{tname}.tmdl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rel = []
    for ft, fc, tt, tc in RELATIONSHIPS:
        rel += [f"relationship {_guid('r-' + ft + fc + tt)}", f"\tfromColumn: {ft}.{fc}", f"\ttoColumn: {tt}.{tc}", ""]
    (d / "relationships.tmdl").write_text("\n".join(rel) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- PBIR
def _col(entity: str, prop: str) -> Dict[str, Any]:
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


def _mea(entity: str, prop: str) -> Dict[str, Any]:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}}


DISPLAY = {"group_name": "Country group", "country_name": "Country", "year": "Year", "family": "Indicator family",
           "publisher": "Publisher", "label": "Provenance", "indicator_label": "Indicator", "target": "Forecast target",
           "model": "Model"}


def _proj(field: Dict[str, Any], display: Optional[str] = None, active: bool = False) -> Dict[str, Any]:
    kind = "Column" if "Column" in field else "Measure"
    ent = field[kind]["Expression"]["SourceRef"]["Entity"]
    prop = field[kind]["Property"]
    display = display or DISPLAY.get(prop)
    p = {"field": field, "queryRef": f"{ent}.{prop}", "nativeQueryRef": display or prop}
    if active:
        p["active"] = True
    if display:
        p["displayName"] = display
    return p


def _lit(v: str) -> Dict[str, Any]:
    return {"expr": {"Literal": {"Value": v}}}


def _title(text: str) -> Dict[str, Any]:
    return {"title": [{"properties": {"show": _lit("true"), "text": _lit("'" + text.replace("'", "''") + "'"),
                                      "fontSize": _lit("12D"), "bold": _lit("true")}}]}


def _visual(name: str, vtype: str, pos: Tuple[float, float, float, float], roles: Dict[str, List[Dict[str, Any]]],
            title: Optional[str] = None, objects: Optional[Dict[str, Any]] = None, sort: Optional[Dict[str, Any]] = None,
            z: int = 0) -> Dict[str, Any]:
    x, y, w, h = pos
    vis: Dict[str, Any] = {"visualType": vtype, "query": {"queryState": {r: {"projections": p} for r, p in roles.items()}}}
    if sort:
        vis["query"]["sortDefinition"] = sort
    if objects:
        vis["objects"] = objects
    if title:
        vis["visualContainerObjects"] = _title(title)
    vis["drillFilterOtherVisuals"] = True
    return {"$schema": f"{SCHEMA}/item/report/definition/visualContainer/2.4.0/schema.json", "name": name,
            "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z}, "visual": vis}


def _slicer(name: str, field: Dict[str, Any], pos, title: str, z: int, mode: str = "'Basic'",
            select: Optional[List[str]] = None, single: bool = False) -> Dict[str, Any]:
    objects: Dict[str, Any] = {"data": [{"properties": {"mode": _lit(mode)}}],
                               "header": [{"properties": {"show": _lit("false")}}]}
    general: Dict[str, Any] = {}
    if select:
        ent = field["Column"]["Expression"]["SourceRef"]["Entity"]
        prop = field["Column"]["Property"]
        general["filter"] = {"filter": {"Version": 2, "From": [{"Name": "s", "Entity": ent, "Type": 0}],
                                        "Where": [{"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": prop}}],
                                                                         "Values": [[{"Literal": {"Value": "'" + v.replace("'", "''") + "'"}}] for v in select]}}}]}}
    if general:
        objects["general"] = [{"properties": general}]
    if single:
        objects["selection"] = [{"properties": {"singleSelect": _lit("true")}}]
    return _visual(name, "slicer", pos, {"Values": [_proj(field, active=True)]}, title=title, objects=objects, z=z)


def report_pages() -> List[Dict[str, Any]]:
    C, I, Y, P, S = "dim_country", "dim_indicator", "dim_year", "dim_provenance", "dim_source"
    pages = []
    # 1 Monitoring --------------------------------------------------------
    v = [
        _slicer(_vid("p1s1"), _col(C, "group_name"), (10, 10, 300, 150), "Country group", 1),
        _slicer(_vid("p1s2"), _col(C, "country_name"), (10, 170, 300, 300), "Country", 2),
        _slicer(_vid("p1s3"), _col(Y, "year"), (10, 480, 300, 110), "Year", 3, mode="'Between'"),
        _visual(_vid("p1c1"), "card", (330, 10, 300, 110), {"Values": [_proj(_mea("fact_dmi", "Average DMI (en)"), display="Average DMI")]}, "Average DMI", z=4),
        _visual(_vid("p1c2"), "card", (650, 10, 300, 110), {"Values": [_proj(_mea("fact_cell", "Published share (en)"), display="Published share")]}, "Share of published cells", z=5),
        _visual(_vid("p1c3"), "card", (970, 10, 300, 110), {"Values": [_proj(_mea("fact_dmi", "Average DMI, published cells only (en)"), display="Average DMI, published cells only")]}, "Average DMI, published cells only", z=6),
        _visual(_vid("p1l1"), "lineChart", (330, 130, 620, 580), {
            "Category": [_proj(_col(Y, "year"), active=True)],
            "Y": [_proj(_mea("fact_dmi", "Average DMI"))],
            "Series": [_proj(_col(C, "group_name"))]}, "Average DMI by country group", z=7),
        _visual(_vid("p1b1"), "clusteredBarChart", (970, 130, 300, 580), {
            "Category": [_proj(_col(C, "country_name"), active=True)],
            "Y": [_proj(_mea("fact_dmi", "Average DMI"))]}, "Average DMI by country",
            sort={"sort": [{"field": _mea("fact_dmi", "Average DMI"), "direction": "Descending"}]}, z=8),
    ]
    pages.append(("monitoring", "Monitoring", v))
    # 2 Provenance --------------------------------------------------------
    v = [
        _slicer(_vid("p2s1"), _col(I, "family"), (10, 10, 300, 170), "Indicator family", 1),
        _slicer(_vid("p2s2"), _col(S, "publisher"), (10, 190, 300, 260), "Publisher", 2),
        _slicer(_vid("p2s3"), _col(C, "group_name"), (10, 460, 300, 150), "Country group", 3),
        _visual(_vid("p2k1"), "columnChart", (330, 10, 940, 300), {
            "Category": [_proj(_col(Y, "year"), active=True)],
            "Y": [_proj(_mea("fact_cell", "Cells"))],
            "Series": [_proj(_col(P, "label"))]}, "Cells by year and provenance state", z=4),
        _visual(_vid("p2m1"), "pivotTable", (330, 320, 940, 390), {
            "Rows": [_proj(_col(I, "indicator_label"), active=True)],
            "Columns": [_proj(_col(Y, "year"), active=True)],
            "Values": [_proj(_mea("fact_cell", "Published share (en)"), display="Published share")]}, "Share of published cells by indicator and year", z=5),
    ]
    pages.append(("provenance", "Provenance", v))
    # 3 Context indicators ------------------------------------------------
    v = [
        _slicer(_vid("p3s1"), _col(I, "family"), (10, 10, 300, 170), "Indicator family", 1, select=["Context (Z)"]),
        _slicer(_vid("p3s2"), _col(I, "indicator_label"), (10, 190, 300, 300), "Indicator", 2, select=["Z2 ICT goods exports"], single=True),
        _slicer(_vid("p3s3"), _col(C, "group_name"), (10, 500, 300, 150), "Country group", 3),
        _visual(_vid("p3l1"), "lineChart", (330, 10, 560, 350), {
            "Category": [_proj(_col(Y, "year"), active=True)],
            "Y": [_proj(_mea("fact_cell", "Mean scaled value"))],
            "Series": [_proj(_col(C, "group_name"))]}, "Selected indicator, scaled 0-100, by country group", z=4),
        _visual(_vid("p3x1"), "scatterChart", (330, 370, 560, 340), {
            "Category": [_proj(_col(C, "country_name"), active=True)],
            "X": [_proj(_mea("fact_cell", "Mean scaled value"))],
            "Y": [_proj(_mea("fact_dmi", "Average DMI"))],
            "Series": [_proj(_col(C, "group_name"))]}, "Selected indicator (x) against the DMI (y), by country", z=5),
        _visual(_vid("p3t1"), "tableEx", (900, 10, 370, 700), {
            "Values": [_proj(_col(I, "indicator_id"), display="Indicator"),
                       _proj(_col("fact_estimate", "arm"), display="Specification"),
                       _proj(_col("fact_estimate", "coefficient_txt"), display="Coefficient"),
                       _proj(_col("fact_estimate", "p_txt"), display="p")]}, "Panel estimates", z=6),
    ]
    pages.append(("context", "Context indicators", v))
    # 4 Models and explanations ----------------------------------------------
    v = [
        _slicer(_vid("p4s1"), _col("fact_forecast", "target"), (10, 10, 300, 130), "Forecast target", 1, select=["DMI (published cells only)"], single=True),
        _visual(_vid("p4b1"), "clusteredBarChart", (330, 10, 460, 350), {
            "Category": [_proj(_col("fact_forecast", "model"), active=True)],
            "Y": [_proj(_mea("fact_forecast", "Mean RMSE"))]}, "Rolling-origin RMSE, 2018-2025",
            sort={"sort": [{"field": _mea("fact_forecast", "Mean RMSE"), "direction": "Ascending"}]}, z=2),
        _visual(_vid("p4b2"), "clusteredBarChart", (810, 10, 460, 350), {
            "Category": [_proj(_col(I, "indicator_label"), active=True)],
            "Y": [_proj(_mea("fact_shap", "Mean |SHAP|"))]}, "Mean |SHAP| of the lagged predictors",
            sort={"sort": [{"field": _mea("fact_shap", "Mean |SHAP|"), "direction": "Descending"}]}, z=3),
        _visual(_vid("p4t1"), "tableEx", (330, 370, 940, 340), {
            "Values": [_proj(_col("fact_forecast", "model"), display="Model"),
                       _proj(_col("fact_forecast", "rmse_txt"), display="RMSE"),
                       _proj(_col("fact_forecast", "mae_txt"), display="MAE"),
                       _proj(_col("fact_forecast", "delta_rmse_txt"), display="RMSE minus persistence"),
                       _proj(_col("fact_forecast", "ci_txt"), display="95% bootstrap CI"),
                       _proj(_col("fact_forecast", "origins_beating_persistence"), display="Origins better than persistence")]},
            "Forecast accuracy against the persistence forecast",
            sort={"sort": [{"field": _col("fact_forecast", "rmse_txt"), "direction": "Ascending"}]}, z=4),
        _slicer(_vid("p4s2"), _col(I, "family"), (10, 150, 300, 170), "Indicator family", 5),
    ]
    pages.append(("models", "Models and explanations", v))
    return pages


def write_report(proj: Path, theme_src: Optional[Path], active: str = "monitoring") -> None:
    rp = proj / f"{NAME}.Report"
    d = rp / "definition"
    if rp.exists():
        shutil.rmtree(rp)
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (rp / "definition.pbir").write_text(json.dumps({
        "$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json", "version": "4.0",
        "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}}, indent=2), encoding="utf-8")
    (rp / ".platform").write_text(json.dumps({
        "$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "Report", "displayName": NAME},
        "config": {"version": "2.0", "logicalId": _guid("report")}}, indent=2), encoding="utf-8")
    (d / "version.json").write_text(json.dumps({"$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json", "version": "2.0.0"}, indent=2), encoding="utf-8")
    report: Dict[str, Any] = {"$schema": f"{SCHEMA}/item/report/definition/report/3.0.0/schema.json", "themeCollection": {}}
    if theme_src and theme_src.exists():
        tdir = rp / "StaticResources" / "SharedResources" / "BaseThemes"
        tdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(theme_src, tdir / theme_src.name)
        report["themeCollection"]["baseTheme"] = {"name": theme_src.stem,
                                                  "reportVersionAtImport": {"visual": "1.8.95", "report": "2.0.95", "page": "1.3.95"},
                                                  "type": "SharedResources"}
    report["settings"] = {"useStylableVisualContainerHeader": True, "defaultDrillFilterOtherVisuals": True}
    (d / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    order = []
    for pid, title, visuals in report_pages():
        pname = _vid("page-" + pid)
        order.append(pname)
        pd_ = d / "pages" / pname
        (pd_ / "visuals").mkdir(parents=True, exist_ok=True)
        (pd_ / "page.json").write_text(json.dumps({
            "$schema": f"{SCHEMA}/item/report/definition/page/2.0.0/schema.json", "name": pname, "displayName": title,
            "displayOption": "FitToPage", "height": 720, "width": 1280}, indent=2), encoding="utf-8")
        for vj in visuals:
            vd = pd_ / "visuals" / vj["name"]
            vd.mkdir(parents=True, exist_ok=True)
            (vd / "visual.json").write_text(json.dumps(vj, indent=2), encoding="utf-8")
    act = _vid("page-" + active)
    (d / "pages" / "pages.json").write_text(json.dumps({
        "$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json", "pageOrder": order, "activePageName": act}, indent=2), encoding="utf-8")


def write_project(root: Path, cells: pd.DataFrame, theme_src: Optional[Path] = None, active: str = "monitoring") -> Dict[str, Any]:
    proj = root / "outputs" / "knowledge_layer" / "powerbi"
    data_dir = proj / "data"
    tables = star_schema(root, cells, data_dir)
    write_semantic_model(proj, tables, data_dir)
    write_report(proj, theme_src, active)
    (proj / f"{NAME}.pbip").write_text(json.dumps({
        "$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
        "artifacts": [{"report": {"path": f"{NAME}.Report"}}], "settings": {"enableAutoRecovery": True}}, indent=2), encoding="utf-8")
    return {"tables": {k: int(len(v)) for k, v in tables.items()}, "pages": len(report_pages()),
            "visuals": sum(len(v) for _, _, v in report_pages()), "measures": sum(len(v) for v in MEASURES.values()),
            "relationships": len(RELATIONSHIPS)}
