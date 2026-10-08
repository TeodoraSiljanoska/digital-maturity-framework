"""Builds the knowledge graph (TBox 1.0.0 + extension 2.0.0 + generated ABox).

Every individual is derived from the pipeline's own artifacts: the processed
panel and its cell provenance, the per-value provenance of the publisher
snapshots, the context-indicator vintage, the econometric, convergence,
forecast and explanation results, and the analytic views of the VDA layer.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCAT, DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from knowledge_layer.context import cell_panel, load_config, load_vintage

DMI = Namespace("http://purl.org/dmi-framework#")
QB = Namespace("http://purl.org/linked-data/cube#")
PROV = Namespace("http://www.w3.org/ns/prov#")

CORE = [f"X{i}" for i in range(1, 11)]
CONTROLS = ["C1", "C2"]

NAME_MK = {
    "X1": "Мобилни претплати", "X2": "Фиксни широкопојасни претплати",
    "X3": "Индекс на развој на е-влада (EGDI)", "X4": "Корисници на интернет (прокси за вештини)",
    "X5": "Национален индекс за сајбер-безбедност (NCSI)", "X6": "Глобален индекс на иновации (GII)",
    "X7": "Индекс на подготвеност на владите за ВИ", "X8": "Расходи за истражување и развој",
    "X9": "Извоз на ИКТ-услуги", "X10": "Индекс на онлајн-услуги (OSI)",
    "C1": "БДП по жител, ППП (лог.)", "C2": "Индекс на образование", "DMI": "Индекс на дигитална зрелост",
}
PILLAR_MK = {
    "infrastructure": ("Infrastructure", "Инфраструктура"),
    "e_government": ("E-government", "Е-влада"),
    "skills": ("Skills", "Вештини"),
    "trust_innovation": ("Trust and innovation", "Доверба и иновации"),
    "digital_economy": ("Digital economy", "Дигитална економија"),
    "control": ("Control (not in DMI)", "Контролни (надвор од DMI)"),
    "composite": ("Composite DMI", "Композитен DMI"),
}
PUBLISHERS = {
    "world_bank": "World Bank",
    "un_desa": "United Nations Department of Economic and Social Affairs",
    "ega": "e-Governance Academy",
    "wipo": "World Intellectual Property Organization",
    "oxford_insights": "Oxford Insights",
    "undp_hdro": "UNDP Human Development Report Office",
    "itu": "International Telecommunication Union",
    "unesco_uis": "UNESCO Institute for Statistics",
    "imf": "International Monetary Fund",
    "netcraft": "Netcraft",
    "unctad": "United Nations Conference on Trade and Development",
    "un_statistics_division": "United Nations Statistics Division",
}
SOURCES = {
    # source id: (label, publisher id, vintage date or None, collection method)
    "world_bank": ("World Bank WDI API, vintage 2026-08-09", "world_bank", "2026-08-09T20:18:24+00:00", "official_api"),
    "world_bank_context": ("World Bank WDI API, vintage 2026-10-08 (context indicators)", "world_bank", "2026-10-08T20:09:01+00:00", "official_api"),
    "un_egdi": ("UN E-Government Survey country pages", "un_desa", None, "curated_snapshot"),
    "ncsi": ("NCSI 3.0 dated country pages", "ega", None, "curated_snapshot"),
    "gii": ("Global Innovation Index reports and data files", "wipo", None, "curated_snapshot"),
    "oxford_insights": ("Government AI Readiness Index reports and data file", "oxford_insights", None, "curated_snapshot"),
    "undp": ("Human Development Report 2025 time series", "undp_hdro", None, "curated_snapshot"),
    "constructed": ("Digital Maturity Framework (constructed index)", None, None, "constructed"),
}
PRIMARY_PUBLISHER = {
    "X1": "itu", "X2": "itu", "X3": "un_desa", "X4": "itu", "X5": "ega", "X6": "wipo",
    "X7": "oxford_insights", "X8": "unesco_uis", "X9": "imf", "X10": "un_desa",
    "C1": "world_bank", "C2": "undp_hdro",
    "Z1": "netcraft", "Z2": "unctad", "Z3": "un_statistics_division", "Z4": "unesco_uis",
    "Z5": "unesco_uis", "Z6": "world_bank",
}
SNAPSHOT_FILES = {
    "X3": "data/raw/un_egdi/egdi_panel_value_provenance.csv",
    "X10": "data/raw/un_egdi/egdi_panel_value_provenance.csv",
    "X5": "data/raw/ncsi/ncsi_panel_value_provenance.csv",
    "X6": "data/raw/gii/gii_panel_value_provenance.csv",
    "X7": "data/raw/oxford/ai_readiness_panel_value_provenance.csv",
    "C2": "data/raw/undp/education_index_panel_value_provenance.csv",
}
MODEL_EXTRA = {
    "persistence": ("predict", "Naive forecast: the DMI of the previous year"),
    "ridge_arx": ("predict", "Ridge regression with the lagged DMI as an extra input"),
    "catboost_arx": ("predict", "CatBoost with the lagged DMI as an extra input"),
    "two_way_fe": ("analyse", "Two-way fixed effects (country and year), Driscoll-Kraay standard errors"),
}


def dec(v: Any) -> Literal:
    return Literal(format(Decimal(repr(float(v))), "f"), datatype=XSD.decimal)


def u(local: str) -> URIRef:
    return DMI[local]


class GraphBuilder:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.g = Graph()
        for p, ns in (("dmi", DMI), ("qb", QB), ("prov", PROV), ("skos", SKOS), ("dcat", DCAT), ("dcterms", DCTERMS), ("owl", OWL), ("xsd", XSD)):
            self.g.bind(p, ns)
        self.stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ TBox
    def load_tbox(self) -> None:
        self.g.parse(self.root / "ontology" / "dmi-framework.ttl", format="turtle")
        self.g.parse(self.root / "ontology" / "dmi-framework-ext.ttl", format="turtle")
        ont = URIRef(str(DMI))
        self.g.remove((ont, OWL.versionInfo, None))
        self.g.add((ont, OWL.versionInfo, Literal("2.0.0")))
        self.g.add((ont, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
        self.g.add((ont, DCTERMS.source, URIRef("https://doi.org/10.5281/zenodo.23043842")))
        self.stats["tbox_triples"] = len(self.g)

    # ------------------------------------------------------------ backbone
    def backbone(self, countries_cfg: Dict[str, Any], zcfg: Dict[str, Any], catalog: List[Dict[str, Any]]) -> None:
        g = self.g
        for pid, label in PUBLISHERS.items():
            p = u(f"Publisher_{pid}")
            g.add((p, RDF.type, DMI.Publisher)); g.add((p, RDF.type, OWL.NamedIndividual))
            g.add((p, RDFS.label, Literal(label, lang="en")))
        for sid, (label, pub, vintage, method) in SOURCES.items():
            s = u(f"Source_{sid}")
            g.add((s, RDF.type, DMI.DataSource)); g.add((s, RDF.type, OWL.NamedIndividual))
            g.add((s, RDFS.label, Literal(sid)))
            g.add((s, DCTERMS.title, Literal(label, lang="en")))
            if pub:
                g.add((s, DMI.publishedBy, u(f"Publisher_{pub}")))
            if vintage:
                g.add((s, DMI.retrievedAt, Literal(vintage, datatype=XSD.dateTime)))
        for pid, (en, mk) in PILLAR_MK.items():
            p = u(f"Pillar_{pid}")
            g.add((p, SKOS.inScheme, DMI.PillarScheme))
            g.add((p, SKOS.prefLabel, Literal(en, lang="en")))
            g.add((p, SKOS.prefLabel, Literal(mk, lang="mk")))
        for gid, grp in (countries_cfg.get("groups") or {}).items():
            gu = u(f"Group_{gid}")
            g.add((gu, RDF.type, DMI.CountryGroup)); g.add((gu, RDF.type, OWL.NamedIndividual))
            g.add((gu, RDFS.label, Literal(grp["name"], lang="en")))
            g.add((gu, RDFS.label, Literal(grp["name_mk"], lang="mk")))
            for c in grp["countries"]:
                cu = u(f"Country_{c['iso3']}")
                g.add((cu, RDF.type, DMI.Country)); g.add((cu, RDF.type, OWL.NamedIndividual))
                g.add((cu, DMI.iso3, Literal(c["iso3"])))
                g.add((cu, RDFS.label, Literal(c["name"], lang="en")))
                g.add((cu, RDFS.label, Literal(c["name_mk"], lang="mk")))
                g.add((cu, DMI.memberOf, gu))
        for rec in catalog:
            iid = rec["indicator_id"]
            iu = u(f"Ind_{iid}")
            cls = DMI.CoreIndicator if iid in CORE else DMI.ControlIndicator if iid in CONTROLS else DMI.CompositeIndex
            # the superclass type is materialised so that plain SPARQL (no reasoner) finds every indicator
            g.add((iu, RDF.type, cls)); g.add((iu, RDF.type, DMI.Indicator)); g.add((iu, RDF.type, OWL.NamedIndividual))
            g.add((iu, DMI.indicatorId, Literal(iid)))
            g.add((iu, DMI.indicatorCode, Literal(str(rec.get("indicator_code")))))
            g.add((iu, RDFS.label, Literal(rec["name"], lang="en")))
            g.add((iu, RDFS.label, Literal(NAME_MK[iid], lang="mk")))
            g.add((iu, DMI.unit, Literal(str(rec.get("unit")))))
            g.add((iu, DMI.polarity, Literal(str(rec.get("polarity") or "positive"))))
            g.add((iu, DMI.role, Literal(str(rec.get("role")))))
            g.add((iu, DMI.belongsToPillar, u(f"Pillar_{rec['pillar']}")))
            g.add((iu, DMI.obtainedBy, u(f"Method_{'official_api' if rec['collection_method'] in ('api', 'official_api') else rec['collection_method']}")))
            g.add((iu, DMI.hasSource, u(f"Source_{rec['source']}")))
            if iid in PRIMARY_PUBLISHER:
                g.add((iu, DMI.primaryPublisher, u(f"Publisher_{PRIMARY_PUBLISHER[iid]}")))
            for ph in rec.get("process_phases") or []:
                g.add((iu, DMI.usedInPhase, u(f"Phase_{ph.capitalize()}")))
        for zid, spec in zcfg["indicators"].items():
            iu = u(f"Ind_{zid}")
            g.add((iu, RDF.type, DMI.ContextIndicator)); g.add((iu, RDF.type, DMI.Indicator)); g.add((iu, RDF.type, OWL.NamedIndividual))
            g.add((iu, DMI.indicatorId, Literal(zid)))
            g.add((iu, DMI.indicatorCode, Literal(spec["code"])))
            g.add((iu, RDFS.label, Literal(spec["name"], lang="en")))
            g.add((iu, RDFS.label, Literal(spec["name_mk"], lang="mk")))
            g.add((iu, DMI.unit, Literal(spec["unit"])))
            g.add((iu, DMI.polarity, Literal("positive")))
            g.add((iu, DMI.role, Literal("context")))
            g.add((iu, DMI.relatedPillar, u(f"Pillar_{spec['related_pillar']}")))
            g.add((iu, DMI.obtainedBy, DMI.Method_official_api))
            g.add((iu, DMI.hasSource, DMI.Source_world_bank_context))
            g.add((iu, DMI.primaryPublisher, u(f"Publisher_{PRIMARY_PUBLISHER[zid]}")))
            for ph in ("Acquire", "Process", "Analyse", "Visualize"):
                g.add((iu, DMI.usedInPhase, u(f"Phase_{ph}")))
        ds = DMI.Dataset_panel
        g.add((ds, RDF.type, QB.DataSet)); g.add((ds, RDF.type, OWL.NamedIndividual))
        g.add((ds, DCTERMS.title, Literal("Digital maturity panel: 12 countries, 2012-2025, indicator x country x year", lang="en")))

    # --------------------------------------------------------------- cells
    def _editions(self) -> pd.DataFrame:
        frames = []
        for iid, path in SNAPSHOT_FILES.items():
            f = pd.read_csv(self.root / path)
            f = f[f["indicator_id"] == iid]
            frames.append(f[["indicator_id", "country_iso3", "reference_year", "value", "edition", "method", "verification"]])
        return pd.concat(frames, ignore_index=True).rename(columns={"reference_year": "year", "value": "src_value"})

    def cells(self, zcfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        root = self.root
        panel = pd.read_parquet(root / "data/processed/analysis_panel.parquet")
        prov = pd.read_parquet(root / "data/processed/cell_provenance.parquet")
        scope = pd.read_parquet(root / "data/processed/cell_reconstruction_scope.parquet")
        raw = pd.read_parquet(root / "data/processed/panel_wide_imputed.parquet")
        ids = CORE + CONTROLS
        long = panel.melt(id_vars=["country_iso3", "year"], value_vars=ids + ["DMI"], var_name="indicator_id", value_name="value")
        pl = prov.melt(id_vars=["country_iso3", "year"], value_vars=ids, var_name="indicator_id", value_name="provenance")
        sl = scope.melt(id_vars=["country_iso3", "year"], value_vars=ids, var_name="indicator_id", value_name="scope")
        rl = raw.melt(id_vars=["country_iso3", "year"], value_vars=ids, var_name="indicator_id", value_name="raw_value")
        long = long.merge(pl, how="left").merge(sl, how="left").merge(rl, how="left")
        long.loc[long["indicator_id"] == "DMI", "provenance"] = "constructed"
        ed = self._editions()
        long = long.merge(ed, on=["indicator_id", "country_iso3", "year"], how="left")
        # keep the edition whose published value equals the panel value
        match = (long["src_value"] - long["raw_value"]).abs() <= 1e-6 * np.maximum(1.0, long["raw_value"].abs())
        long.loc[~match.fillna(False), ["edition", "method", "verification"]] = np.nan
        long = long.sort_values("edition").drop_duplicates(["indicator_id", "country_iso3", "year"], keep="first")
        wb = long["indicator_id"].isin(["X1", "X2", "X4", "X8", "X9", "C1"]) & (long["provenance"] == "official")
        long.loc[wb, "edition"] = "WDI vintage 2026-08-09"
        long.loc[wb, "method"] = "api"
        long.loc[(long["provenance"] != "official"), ["edition", "method", "verification"]] = np.nan
        catalog_src = {"X1": "world_bank", "X2": "world_bank", "X4": "world_bank", "X8": "world_bank", "X9": "world_bank", "C1": "world_bank",
                       "X3": "un_egdi", "X10": "un_egdi", "X5": "ncsi", "X6": "gii", "X7": "oxford_insights", "C2": "undp", "DMI": "constructed"}
        long["source_id"] = long["indicator_id"].map(catalog_src)
        long["phase"] = np.where(long["indicator_id"] == "DMI", "Index", "Acquire")
        # context cells: official and carried only; gaps stay absent
        zlong = load_vintage(root, zcfg)
        zfilled, zprov = cell_panel(zlong, zcfg)
        zids = list(zcfg["indicators"].keys())
        zl = zfilled.melt(id_vars=["country_iso3", "year"], value_vars=zids, var_name="indicator_id", value_name="value")
        zp = zprov.melt(id_vars=["country_iso3", "year"], value_vars=zids, var_name="indicator_id", value_name="provenance")
        zl = zl.merge(zp)
        zl = zl[zl["provenance"] != "missing"].copy()
        zl["edition"] = np.where(zl["provenance"] == "official", "WDI vintage 2026-10-08", None)
        zl["method"] = np.where(zl["provenance"] == "official", "api", None)
        zl["source_id"] = "world_bank_context"
        zl["phase"] = "Acquire"
        zl["scope"] = None
        allc = pd.concat([long[["indicator_id", "country_iso3", "year", "value", "provenance", "scope", "edition", "method", "verification", "source_id", "phase"]],
                          zl.assign(verification=None)[["indicator_id", "country_iso3", "year", "value", "provenance", "scope", "edition", "method", "verification", "source_id", "phase"]]],
                         ignore_index=True)
        allc = allc.dropna(subset=["value"])
        return {"cells": allc, "zprov": zprov, "prov": prov}

    def add_cells(self, cells: pd.DataFrame) -> None:
        g = self.g
        for r in cells.itertuples(index=False):
            e = u(f"SE_{r.indicator_id}_{r.country_iso3}_{int(r.year)}")
            g.add((e, RDF.type, DMI.ScoreEntry)); g.add((e, RDF.type, OWL.NamedIndividual))
            g.add((e, DMI.entryIndicator, u(f"Ind_{r.indicator_id}")))
            g.add((e, DMI.entryCountry, u(f"Country_{r.country_iso3}")))
            g.add((e, DMI.year, Literal(int(r.year), datatype=XSD.integer)))
            g.add((e, DMI.matrixValue, dec(r.value)))
            g.add((e, DMI.hasProvenance, u(f"Prov_{r.provenance}")))
            g.add((e, DMI.entrySource, u(f"Source_{r.source_id}")))
            g.add((e, DMI.entryPhase, u(f"Phase_{r.phase}")))
            g.add((e, DMI.inSeries, u(f"Series_{r.indicator_id}_{r.country_iso3}")))
            g.add((e, QB.dataSet, DMI.Dataset_panel))
            if isinstance(r.scope, str) and r.provenance == "mice_imputed":
                g.add((e, DMI.hasReconstructionScope, u(f"Scope_{r.scope}")))
            if isinstance(r.edition, str):
                g.add((e, DMI.edition, Literal(r.edition)))
            if isinstance(r.method, str):
                g.add((e, DMI.retrievalMethod, Literal(r.method)))
            if isinstance(r.verification, str):
                g.add((e, DMI.verification, Literal(r.verification)))
        self.stats["score_entries"] = int(len(cells))

    def add_series(self, series: pd.DataFrame, ids_without_counts: List[str], countries: List[str]) -> None:
        g = self.g
        seen = set()
        for r in series.itertuples(index=False):
            s = u(f"Series_{r.indicator_id}_{r.country_iso3}")
            seen.add((r.indicator_id, r.country_iso3))
            g.add((s, RDF.type, DMI.TimeSeries)); g.add((s, RDF.type, OWL.NamedIndividual))
            g.add((s, DMI.seriesIndicator, u(f"Ind_{r.indicator_id}")))
            g.add((s, DMI.seriesCountry, u(f"Country_{r.country_iso3}")))
            g.add((s, DMI.nOfficial, Literal(int(r.n_official), datatype=XSD.integer)))
            g.add((s, DMI.officialShare, dec(r.official_share)))
            g.add((s, DMI.longestGap, Literal(int(r.longest_non_official_run), datatype=XSD.integer)))
            if pd.notna(r.first_official_year):
                g.add((s, DMI.firstOfficialYear, Literal(int(r.first_official_year), datatype=XSD.integer)))
                g.add((s, DMI.lastOfficialYear, Literal(int(r.last_official_year), datatype=XSD.integer)))
        for iid in ids_without_counts:
            for iso in countries:
                s = u(f"Series_{iid}_{iso}")
                g.add((s, RDF.type, DMI.TimeSeries)); g.add((s, RDF.type, OWL.NamedIndividual))
                g.add((s, DMI.seriesIndicator, u(f"Ind_{iid}")))
                g.add((s, DMI.seriesCountry, u(f"Country_{iso}")))
        self.stats["series"] = int(len(series)) + len(ids_without_counts) * len(countries)

    # --------------------------------------------------------- model outputs
    def add_models(self) -> None:
        from ontology.query import MODEL_PHASE_MAP

        g = self.g
        models = {k: (v["phase"], v["purpose"]) for k, v in MODEL_PHASE_MAP.items()}
        models.update(MODEL_EXTRA)
        for mid, (phase, purpose) in models.items():
            m = u(f"Model_{mid}")
            g.add((m, RDF.type, DMI.ModelOutput)); g.add((m, RDF.type, OWL.NamedIndividual))
            g.add((m, RDFS.label, Literal(mid)))
            g.add((m, RDFS.comment, Literal(purpose)))
            g.add((m, DMI.assignedToPhase, u(f"Phase_{phase.capitalize()}")))

    def add_estimates(self, assoc: Optional[pd.DataFrame]) -> int:
        g = self.g
        n = 0
        fe = pd.read_csv(self.root / "results/econometrics/coefficients.csv")
        fe = fe[(fe["model"] == "FE") & fe["term"].str.endswith("_lag1")]
        mi = pd.read_csv(self.root / "results/imputation/mi_pooled_fe.csv")
        mi = mi[mi["term"].str.endswith("_lag1")]
        for arm, frame, se_col, est in (
            ("single_imputation", fe, "std_error", "FE, lag 1, clustered by country"),
            ("mi_pooled", mi, "std_error", "FE, lag 1, Rubin's rules over 10 imputations"),
        ):
            for r in frame.itertuples(index=False):
                iid = r.term.replace("_lag1", "")
                e = u(f"Est_{arm}_{iid}")
                g.add((e, RDF.type, DMI.Estimate)); g.add((e, RDF.type, OWL.NamedIndividual))
                g.add((e, DMI.estimateFor, u(f"Ind_{iid}")))
                g.add((e, DMI.coefficient, dec(r.coefficient)))
                g.add((e, DMI.stdError, dec(getattr(r, se_col))))
                g.add((e, DMI.pValue, dec(r.pvalue)))
                g.add((e, DMI.estimator, Literal(est)))
                g.add((e, DMI.estimationArm, Literal(arm)))
                g.add((e, DMI.assignedToPhase, DMI.Phase_Analyse))
                n += 1
        if assoc is not None:
            for r in assoc.itertuples(index=False):
                e = u(f"Est_context_{r.indicator_id}")
                g.add((e, RDF.type, DMI.Estimate)); g.add((e, RDF.type, OWL.NamedIndividual))
                g.add((e, DMI.estimateFor, u(f"Ind_{r.indicator_id}")))
                g.add((e, DMI.coefficient, dec(r.beta_per_sd)))
                g.add((e, DMI.stdError, dec(r.se_driscoll_kraay)))
                g.add((e, DMI.pValue, dec(r.p_driscoll_kraay)))
                g.add((e, DMI.estimator, Literal("Two-way FE, lag 1, per SD of the indicator, Driscoll-Kraay SE")))
                g.add((e, DMI.estimationArm, Literal("context_official_only")))
                g.add((e, DMI.assignedToPhase, DMI.Phase_Analyse))
                n += 1
        self.stats["estimates"] = n
        return n

    def add_forecast_scores(self, scores: pd.DataFrame) -> None:
        g = self.g
        for r in scores.itertuples(index=False):
            f = u(f"FS_{r.target}_{r.model}")
            g.add((f, RDF.type, DMI.ForecastScore)); g.add((f, RDF.type, OWL.NamedIndividual))
            g.add((f, DMI.scoredModel, u(f"Model_{r.model}")))
            g.add((f, DMI.forecastTarget, Literal(r.target)))
            g.add((f, DMI.rmse, dec(r.rmse)))
            g.add((f, DMI.mae, dec(r.mae)))
            g.add((f, DMI.deltaRmseVsPersistence, dec(r.delta_rmse_vs_persistence)))
            g.add((f, DMI.ciLow, dec(r.delta_ci_low)))
            g.add((f, DMI.ciHigh, dec(r.delta_ci_high)))
            g.add((f, DMI.assignedToPhase, DMI.Phase_Predict))
        self.stats["forecast_scores"] = int(len(scores))

    def add_dispersion(self) -> None:
        g = self.g
        sig = pd.read_csv(self.root / "results/convergence/sigma_convergence.csv")
        sig = sig[sig["sample"] == "full"]
        for r in sig.itertuples(index=False):
            d = u(f"DR_{int(r.year)}")
            g.add((d, RDF.type, DMI.DispersionReading)); g.add((d, RDF.type, OWL.NamedIndividual))
            g.add((d, DMI.year, Literal(int(r.year), datatype=XSD.integer)))
            g.add((d, DMI.coefficientOfVariation, dec(r.cv)))
            g.add((d, DMI.assignedToPhase, DMI.Phase_Analyse))
        self.stats["dispersion_readings"] = int(len(sig))

    def add_explanations(self) -> None:
        import json

        g = self.g
        meta = json.loads((self.root / "results/xai/xai_shap_global.meta.json").read_text(encoding="utf-8"))
        model_id = str(meta["meta"]["model_id"]).split("_20")[0]
        shap = pd.read_csv(self.root / "results/xai/shap_global_importance.csv")
        for rank, r in enumerate(shap.itertuples(index=False), start=1):
            iid = r.feature.replace("_lag1", "")
            e = u(f"Expl_SHAP_{iid}")
            g.add((e, RDF.type, DMI.Explanation)); g.add((e, RDF.type, OWL.NamedIndividual))
            g.add((e, DMI.aboutIndicator, u(f"Ind_{iid}")))
            g.add((e, DMI.explainsOutput, u(f"Model_{model_id}")))
            g.add((e, DMI.meanAbsShap, dec(r.mean_abs_shap)))
        self.stats["explanations"] = int(len(shap))
        self.stats["shap_model"] = model_id  # type: ignore[assignment]

    def add_views(self, vcfg: Dict[str, Any]) -> None:
        g = self.g
        for cid, spec in (vcfg.get("contexts") or {}).items():
            c = u(cid)
            g.add((c, RDF.type, DMI.DecisionContext)); g.add((c, RDF.type, OWL.NamedIndividual))
            g.add((c, RDFS.label, Literal(spec["label"])))
            g.add((c, RDFS.comment, Literal(spec["comment"])))
        for vid, spec in vcfg["views"].items():
            v = u(f"View_{vid}")
            g.add((v, RDF.type, DMI.AnalyticView)); g.add((v, RDF.type, OWL.NamedIndividual))
            g.add((v, RDFS.label, Literal(spec["title"], lang="en")))
            g.add((v, RDFS.label, Literal(spec["title_mk"], lang="mk")))
            g.add((v, DMI.servesContext, u(spec["context"])))
            for t in spec["tasks"]:
                g.add((v, DMI.supportsTask, u(f"Task_{t}")))
            for i in spec["indicators"]:
                g.add((v, DMI.showsIndicator, u(f"Ind_{i}")))
            for s in spec["slicers"]:
                g.add((v, DMI.slicerField, Literal(s)))
        self.stats["views"] = len(vcfg["views"])

    def add_competency_questions(self, cqs: Dict[str, Dict[str, str]]) -> None:
        g = self.g
        for cid, spec in cqs.items():
            q = u(cid)
            g.add((q, RDF.type, DMI.CompetencyQuestion)); g.add((q, RDF.type, OWL.NamedIndividual))
            g.add((q, DMI.questionText, Literal(spec["question"])))
            g.add((q, DMI.usesValidationQuery, u(f"VQ_{cid}")))
            vq = u(f"VQ_{cid}")
            g.add((vq, RDF.type, DMI.ValidationQuery)); g.add((vq, RDF.type, OWL.NamedIndividual))
            g.add((vq, DMI.queryText, Literal(spec["sparql"])))


def qb_structure() -> Graph:
    """Data structure definition of the cube, published next to the graph."""
    g = Graph()
    g.bind("qb", QB); g.bind("dmi", DMI)
    dsd = DMI.DSD_panel
    g.add((dsd, RDF.type, QB.DataStructureDefinition))
    g.add((DMI.Dataset_panel, QB.structure, dsd))
    comps = [
        (DMI.entryIndicator, QB.dimension, QB.DimensionProperty, 1),
        (DMI.entryCountry, QB.dimension, QB.DimensionProperty, 2),
        (DMI.year, QB.dimension, QB.DimensionProperty, 3),
        (DMI.matrixValue, QB.measure, QB.MeasureProperty, None),
        (DMI.hasProvenance, QB.attribute, QB.AttributeProperty, None),
        (DMI.edition, QB.attribute, QB.AttributeProperty, None),
    ]
    for prop, role, kind, order in comps:
        c = BNode()
        g.add((dsd, QB.component, c))
        g.add((c, role, prop))
        if order is not None:
            g.add((c, QB.order, Literal(order)))
        if role == QB.attribute:
            g.add((c, QB.componentRequired, Literal(prop == DMI.hasProvenance)))
        g.add((prop, RDF.type, kind))
    g.add((DMI.entryIndicator, RDFS.range, DMI.Indicator))
    g.add((DMI.entryCountry, RDFS.range, DMI.Country))
    g.add((DMI.year, RDFS.range, XSD.integer))
    g.add((DMI.matrixValue, RDFS.range, XSD.decimal))
    return g


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
