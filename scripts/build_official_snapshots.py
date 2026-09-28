#!/usr/bin/env python3
"""Build curated official snapshot CSVs for non-API research indicators.

Superseded by scripts/build_snapshots_v3.py from v3 onwards; kept so that the
v1/v2 snapshots (tag v2-frozen) remain reproducible.

Prefer downloading from official/public sources when available. Where direct
machine-readable downloads are unreliable, embed carefully researched published
scores from UN E-Government Survey (via QoG UN egov series), WIPO GII reports,
eGA NCSI, Oxford Insights Government AI Readiness Index, and UNDP HDR.

Years without an official publication are left as empty/NaN (never fabricated).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = [
    "MKD",
    "SRB",
    "SVN",
    "FRA",
    "GBR",
    "DEU",
    "USA",
    "CAN",
    "AUS",
    "BRA",
    "ARG",
    "IND",
]
START_YEAR = 2012
END_YEAR = 2025

# ---------------------------------------------------------------------------
# Official published values (documented in sidecar metadata JSON)
# ---------------------------------------------------------------------------

# UN EGDI / OSI from UN E-Government Survey editions, retrieved via the Quality
# of Government (QoG) Standard Time-Series dataset (egov_egov / egov_osi), which
# archives UN DESA published scores. QoG year = survey year - 1 for these series.
# Survey years: 2012, 2014, 2016, 2018, 2020, 2022.
EGDI_OFFICIAL: Dict[int, Dict[str, float]] = {
    2012: {
        "ARG": 0.62279,
        "AUS": 0.83900,
        "BRA": 0.61673,
        "CAN": 0.84299,
        "DEU": 0.80788,
        "FRA": 0.86349,
        "GBR": 0.89603,
        "IND": 0.38287,
        "MKD": 0.55865,
        "SRB": 0.63119,
        "SVN": 0.74921,
        "USA": 0.86873,
    },
    2014: {
        "ARG": 0.63059,
        "AUS": 0.91034,
        "BRA": 0.60082,
        "CAN": 0.84177,
        "DEU": 0.78640,
        "FRA": 0.89384,
        "GBR": 0.86948,
        "IND": 0.38343,
        "MKD": 0.47198,
        "SRB": 0.54715,
        "SVN": 0.65054,
        "USA": 0.87483,
    },
    2016: {
        "ARG": 0.69780,
        "AUS": 0.91428,
        "BRA": 0.63769,
        "CAN": 0.82847,
        "DEU": 0.82099,
        "FRA": 0.84559,
        "GBR": 0.91928,
        "IND": 0.46375,
        "MKD": 0.58855,
        "SRB": 0.71308,
        "SVN": 0.77691,
        "USA": 0.84201,
    },
    2018: {
        "ARG": 0.73350,
        "AUS": 0.90530,
        "BRA": 0.73270,
        "CAN": 0.82580,
        "DEU": 0.87650,
        "FRA": 0.87900,
        "GBR": 0.89990,
        "IND": 0.56690,
        "MKD": 0.63120,
        "SRB": 0.71550,
        "SVN": 0.77140,
        "USA": 0.87690,
    },
    2020: {
        "ARG": 0.82790,
        "AUS": 0.94320,
        "BRA": 0.76770,
        "CAN": 0.84200,
        "DEU": 0.85240,
        "FRA": 0.87180,
        "GBR": 0.93580,
        "IND": 0.59640,
        "MKD": 0.70830,
        "SRB": 0.74740,
        "SVN": 0.85460,
        "USA": 0.92970,
    },
    2022: {
        "ARG": 0.81980,
        "AUS": 0.94050,
        "BRA": 0.79100,
        "CAN": 0.85110,
        "DEU": 0.87700,
        "FRA": 0.88320,
        "GBR": 0.91380,
        "IND": 0.58830,
        "MKD": 0.70000,
        "SRB": 0.82370,
        "SVN": 0.87810,
        "USA": 0.91510,
    },
}

OSI_OFFICIAL: Dict[int, Dict[str, float]] = {
    2012: {
        "ARG": 0.52941,
        "AUS": 0.86274,
        "BRA": 0.67320,
        "CAN": 0.88888,
        "DEU": 0.75163,
        "FRA": 0.87581,
        "GBR": 0.97385,
        "IND": 0.53594,
        "MKD": 0.45098,
        "SRB": 0.57516,
        "SVN": 0.66666,
        "USA": 1.00000,
    },
    2014: {
        "ARG": 0.55118,
        "AUS": 0.92913,
        "BRA": 0.59842,
        "CAN": 0.91338,
        "DEU": 0.66929,
        "FRA": 1.00000,
        "GBR": 0.89763,
        "IND": 0.54330,
        "MKD": 0.24409,
        "SRB": 0.39370,
        "SVN": 0.42519,
        "USA": 0.94488,
    },
    2016: {
        "ARG": 0.71014,
        "AUS": 0.97826,
        "BRA": 0.73188,
        "CAN": 0.95652,
        "DEU": 0.84058,
        "FRA": 0.94203,
        "GBR": 1.00000,
        "IND": 0.74638,
        "MKD": 0.60870,
        "SRB": 0.81884,
        "SVN": 0.84783,
        "USA": 0.92754,
    },
    2018: {
        "ARG": 0.75000,
        "AUS": 0.97220,
        "BRA": 0.92360,
        "CAN": 0.93060,
        "DEU": 0.93060,
        "FRA": 0.97920,
        "GBR": 0.97920,
        "IND": 0.95140,
        "MKD": 0.71530,
        "SRB": 0.73610,
        "SVN": 0.79860,
        "USA": 0.98610,
    },
    2020: {
        "ARG": 0.84710,
        "AUS": 0.94710,
        "BRA": 0.87060,
        "CAN": 0.84120,
        "DEU": 0.73530,
        "FRA": 0.88240,
        "GBR": 0.95880,
        "IND": 0.85290,
        "MKD": 0.74120,
        "SRB": 0.79410,
        "SVN": 0.85290,
        "USA": 0.94710,
    },
    2022: {
        "ARG": 0.80890,
        "AUS": 0.93800,
        "BRA": 0.89640,
        "CAN": 0.85040,
        "DEU": 0.79050,
        "FRA": 0.87680,
        "GBR": 0.88590,
        "IND": 0.79340,
        "MKD": 0.70200,
        "SRB": 0.85140,
        "SVN": 0.86660,
        "USA": 0.93040,
    },
}

# WIPO Global Innovation Index overall scores from published annual reports.
# Gaps left empty where a clean published score was not compiled into this snapshot.
GII_OFFICIAL: Dict[int, Dict[str, float]] = {
    2013: {
        "USA": 60.31,
        "GBR": 61.25,
        "CAN": 57.60,
        "DEU": 55.83,
        "AUS": 53.07,
        "FRA": 52.83,
        "SVN": 47.32,
        "MKD": 38.18,
        "SRB": 37.87,
        "ARG": 37.66,
        "IND": 36.17,
        "BRA": 36.33,
    },
    2019: {
        "USA": 61.73,
        "GBR": 61.30,
        "DEU": 58.19,
        "FRA": 54.25,
        "CAN": 53.88,
        "AUS": 50.34,
        "SVN": 45.25,
        "IND": 36.58,
        "BRA": 33.82,
        "SRB": 35.71,
        "MKD": 35.29,
        "ARG": 31.95,
    },
    2020: {
        "USA": 60.56,
        "GBR": 59.78,
        "DEU": 56.55,
        "FRA": 53.66,
        "CAN": 52.26,
        "AUS": 48.35,
        "SVN": 42.91,
        "IND": 35.59,
        "BRA": 31.94,
        "SRB": 34.33,
        "MKD": 33.43,
        "ARG": 28.33,
    },
    2021: {
        "USA": 61.3,
        "GBR": 59.8,
        "DEU": 57.3,
        "FRA": 55.0,
        "CAN": 53.1,
        "AUS": 48.3,
        "SVN": 44.1,
        "IND": 36.4,
        "SRB": 35.0,
        "BRA": 34.2,
        "MKD": 34.1,
        "ARG": 29.8,
    },
    2022: {
        "USA": 61.8,
        "GBR": 59.7,
        "DEU": 57.2,
        "FRA": 55.0,
        "CAN": 50.8,
        "AUS": 47.1,
        "SVN": 40.6,
        "IND": 36.6,
        "BRA": 32.5,
        "SRB": 32.3,
        "MKD": 28.8,
        "ARG": 28.6,
    },
    2023: {
        "USA": 63.5,
        "GBR": 62.4,
        "DEU": 58.8,
        "FRA": 56.0,
        "CAN": 53.8,
        "AUS": 49.7,
        "SVN": 42.2,
        "IND": 38.1,
        "BRA": 33.6,
        "SRB": 33.1,
        "MKD": 33.0,
        "ARG": 28.0,
    },
}

# NCSI (e-Governance Academy). Live index is continuously updated; archive=2022
# captures the 2022 edition. Live scores recorded at snapshot build are assigned
# to 2023 (panel end year). Earlier years left NaN unless archived.
NCSI_OFFICIAL: Dict[int, Dict[str, float]] = {
    2022: {
        "DEU": 90.91,
        "GBR": 89.61,
        "FRA": 84.42,
        "SRB": 80.52,
        "CAN": 70.13,
        "IND": 67.53,
        "SVN": 67.53,
        "AUS": 66.23,
        "USA": 64.94,
        "ARG": 63.64,
        "MKD": 58.44,
        "BRA": 51.95,
    },
    2023: {
        "CAN": 96.67,
        "DEU": 90.83,
        "SVN": 89.17,
        "FRA": 89.17,
        "AUS": 85.00,
        "MKD": 84.17,
        "USA": 84.17,
        "SRB": 80.83,
        "GBR": 80.00,
        "IND": 75.83,
        "BRA": 71.67,
        "ARG": 70.00,
    },
}

# Oxford Insights Government AI Readiness Index.
# 2019 published on a 0–10 scale; 2020+ editions use 0–100. Values stored as
# published (see metadata notes). Only years with verified published annex
# scores are populated; other years are left empty/NaN.
OXFORD_OFFICIAL: Dict[int, Dict[str, float]] = {
    2019: {
        "GBR": 9.069,
        "DEU": 8.810,
        "USA": 8.804,
        "CAN": 8.674,
        "FRA": 8.608,
        "AUS": 8.126,
        "IND": 7.515,
        "SVN": 6.232,
        "BRA": 6.157,
        "ARG": 5.684,
        "SRB": 5.364,
        "MKD": 5.284,
    },
    2020: {
        "USA": 85.479,
        "GBR": 81.124,
        "DEU": 78.974,
        "FRA": 73.767,
        "AUS": 73.577,
        "CAN": 73.158,
        "SVN": 55.986,
        "IND": 55.983,
        "SRB": 53.431,
        "ARG": 50.754,
        "BRA": 47.464,
        "MKD": 42.995,
    },
    2022: {
        "USA": 85.72,
        "GBR": 78.54,
        "CAN": 77.39,
        "FRA": 75.78,
        "AUS": 75.29,
        "DEU": 72.64,
        "IND": 63.67,
        "BRA": 62.37,
        "SVN": 61.45,
        "ARG": 57.39,
        "SRB": 52.96,
        "MKD": 46.11,
    },
    2023: {
        "USA": 84.80,
        "GBR": 78.57,
        "CAN": 77.07,
        "FRA": 76.07,
        "DEU": 75.26,
        "AUS": 73.89,
        "BRA": 63.70,
        "SVN": 62.63,
        "IND": 62.58,
        "ARG": 57.72,
        "SRB": 55.57,
        "MKD": 45.40,
    },
}

# UNDP Human Development Report Education Index (0–1).
# Values from HDR statistical annex / HDR data centre time series for published years.
EDU_INDEX_OFFICIAL: Dict[int, Dict[str, float]] = {
    2012: {
        "AUS": 0.927,
        "DEU": 0.928,
        "CAN": 0.890,
        "USA": 0.900,
        "GBR": 0.886,
        "SVN": 0.886,
        "FRA": 0.816,
        "ARG": 0.830,
        "SRB": 0.760,
        "BRA": 0.661,
        "MKD": 0.680,
        "IND": 0.492,
    },
    2015: {
        "AUS": 0.929,
        "DEU": 0.928,
        "CAN": 0.894,
        "USA": 0.900,
        "GBR": 0.896,
        "SVN": 0.894,
        "FRA": 0.831,
        "ARG": 0.839,
        "SRB": 0.770,
        "BRA": 0.681,
        "MKD": 0.692,
        "IND": 0.535,
    },
    2018: {
        "AUS": 0.923,
        "DEU": 0.943,
        "CAN": 0.899,
        "USA": 0.903,
        "GBR": 0.922,
        "SVN": 0.907,
        "FRA": 0.840,
        "ARG": 0.855,
        "SRB": 0.783,
        "BRA": 0.694,
        "MKD": 0.698,
        "IND": 0.555,
    },
    2019: {
        "DEU": 0.943,
        "AUS": 0.924,
        "GBR": 0.928,
        "SVN": 0.910,
        "CAN": 0.899,
        "USA": 0.900,
        "ARG": 0.855,
        "FRA": 0.817,
        "SRB": 0.783,
        "MKD": 0.698,
        "BRA": 0.694,
        "IND": 0.555,
    },
    2021: {
        "DEU": 0.943,
        "AUS": 0.924,
        "GBR": 0.928,
        "SVN": 0.910,
        "CAN": 0.899,
        "USA": 0.900,
        "ARG": 0.855,
        "FRA": 0.840,
        "SRB": 0.778,
        "MKD": 0.704,
        "BRA": 0.704,
        "IND": 0.556,
    },
    2022: {
        "DEU": 0.943,
        "AUS": 0.927,
        "GBR": 0.930,
        "SVN": 0.912,
        "CAN": 0.902,
        "USA": 0.903,
        "ARG": 0.858,
        "FRA": 0.841,
        "SRB": 0.780,
        "MKD": 0.706,
        "BRA": 0.707,
        "IND": 0.570,
    },
}


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "dmf-snapshot-builder/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def try_refresh_egdi_osi_from_qog() -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], str]:
    """Optionally refresh EGDI/OSI from QoG Standard TS if download succeeds."""
    url = "https://www.qogdata.pol.gu.se/data/qog_std_ts_jan24.csv"
    try:
        raw = _http_get(url, timeout=180)
        from io import BytesIO

        df = pd.read_csv(
            BytesIO(raw),
            usecols=["ccodealp", "year", "egov_egov", "egov_osi"],
            low_memory=False,
        )
    except Exception as exc:  # noqa: BLE001
        return EGDI_OFFICIAL, OSI_OFFICIAL, f"QoG download skipped/failed ({exc}); using embedded UN/QoG values"

    qog_to_survey = {2011: 2012, 2013: 2014, 2015: 2016, 2017: 2018, 2019: 2020, 2021: 2022}
    sub = df[df["ccodealp"].isin(COUNTRIES) & df["year"].isin(qog_to_survey.keys())].copy()
    sub["survey_year"] = sub["year"].map(qog_to_survey)
    egdi: Dict[int, Dict[str, float]] = {}
    osi: Dict[int, Dict[str, float]] = {}
    for y, grp in sub.groupby("survey_year"):
        egdi[int(y)] = {
            str(r.ccodealp): round(float(r.egov_egov), 5)
            for r in grp.itertuples()
            if pd.notna(r.egov_egov)
        }
        osi[int(y)] = {
            str(r.ccodealp): round(float(r.egov_osi), 5)
            for r in grp.itertuples()
            if pd.notna(r.egov_osi)
        }
    return egdi, osi, f"Refreshed from QoG Standard TS ({url})"


def try_refresh_ncsi_live() -> Tuple[Dict[int, Dict[str, float]], str]:
    """Scrape current NCSI ranking page; map to 2023 panel year."""
    data = {k: dict(v) for k, v in NCSI_OFFICIAL.items()}
    try:
        html = _http_get("https://ncsi.ega.ee/ncsi-index/", timeout=60).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return data, f"NCSI live scrape failed ({exc}); using embedded archive/live values"

    name_to_iso = {
        "North Macedonia": "MKD",
        "Serbia": "SRB",
        "Slovenia": "SVN",
        "France": "FRA",
        "United Kingdom": "GBR",
        "Germany": "DEU",
        "United States": "USA",
        "Canada": "CAN",
        "Australia": "AUS",
        "Brazil": "BRA",
        "Argentina": "ARG",
        "India": "IND",
    }
    live: Dict[str, float] = {}
    for name, iso in name_to_iso.items():
        m = re.search(
            rf"{re.escape(name)}</a></td>\s*<td[^>]*>\s*<strong[^>]*>\s*([\d.]+)",
            html,
            flags=re.I,
        )
        if not m:
            m = re.search(
                rf"{re.escape(name)}</a>.*?<strong[^>]*>\s*([\d.]+)",
                html,
                flags=re.I | re.S,
            )
        if m:
            live[iso] = float(m.group(1))
    if live:
        data[2023] = live
        return data, "2023 scores refreshed from https://ncsi.ega.ee/ncsi-index/; 2022 from NCSI archive"
    return data, "NCSI parse produced no rows; using embedded values"


def wide_from_nested(
    nested: Dict[int, Dict[str, float]],
    value_col: str,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    years = years or list(range(START_YEAR, END_YEAR + 1))
    rows: List[Dict[str, Any]] = []
    for iso in COUNTRIES:
        for year in years:
            val = nested.get(year, {}).get(iso)
            rows.append({"country_iso3": iso, "year": year, value_col: val})
    return pd.DataFrame(rows)


def write_snapshot(
    path: Path,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if path.suffix == ".csv":
        meta_path = path.with_name(path.stem + "_metadata.json")
    metadata = dict(metadata)
    metadata["written_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata["rows"] = int(len(df))
    metadata["path"] = str(path.relative_to(ROOT))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {path} ({len(df)} rows) + {meta_path.name}")


def build_egdi_panel(egdi: Dict[int, Dict[str, float]], osi: Dict[int, Dict[str, float]], note: str) -> None:
    rows = []
    for iso in COUNTRIES:
        for year in range(START_YEAR, END_YEAR + 1):
            rows.append(
                {
                    "country_iso3": iso,
                    "year": year,
                    "EGDI": egdi.get(year, {}).get(iso),
                    "OSI": osi.get(year, {}).get(iso),
                }
            )
    df = pd.DataFrame(rows)
    write_snapshot(
        ROOT / "data/raw/un_egdi/egdi_panel.csv",
        df,
        {
            "dataset": "UN_EGOV",
            "indicators": ["EGDI", "OSI"],
            "source_url": "https://publicadministration.un.org/egovkb/",
            "secondary_source_url": "https://www.gu.se/en/quality-government/qog-data",
            "notes": (
                "Biennial UN E-Government Survey scores for EGDI and OSI. "
                "Values sourced from UN DESA via QoG Standard Time-Series "
                "(egov_egov, egov_osi). Non-survey years are empty/NaN. "
                + note
            ),
            "publication_years": sorted(egdi.keys()),
            "countries": COUNTRIES,
            "panel_years": [START_YEAR, END_YEAR],
        },
    )


def build_gii_panel() -> None:
    df = wide_from_nested(GII_OFFICIAL, "GII_SCORE")
    write_snapshot(
        ROOT / "data/raw/gii/gii_panel.csv",
        df,
        {
            "dataset": "GII",
            "indicators": ["GII_SCORE"],
            "source_url": "https://www.wipo.int/global_innovation_index/",
            "notes": (
                "WIPO Global Innovation Index overall scores from published annual "
                "reports (2013, 2019–2023 compiled). Other years left empty because "
                "scores are not linearly comparable across all editions without the "
                "official report value."
            ),
            "publication_years": sorted(GII_OFFICIAL.keys()),
            "countries": COUNTRIES,
        },
    )


def build_ncsi_panel(ncsi: Dict[int, Dict[str, float]], note: str) -> None:
    df = wide_from_nested(ncsi, "NCSI_SCORE")
    write_snapshot(
        ROOT / "data/raw/ncsi/ncsi_panel.csv",
        df,
        {
            "dataset": "NCSI",
            "indicators": ["NCSI_SCORE"],
            "source_url": "https://ncsi.ega.ee/",
            "notes": (
                "National Cyber Security Index (e-Governance Academy). "
                "2022 from NCSI archive edition; 2023 from live ranking snapshot. "
                "Earlier years empty (no stable public machine archive for all years). "
                + note
            ),
            "publication_years": sorted(ncsi.keys()),
            "countries": COUNTRIES,
        },
    )


def build_oxford_panel() -> None:
    df = wide_from_nested(OXFORD_OFFICIAL, "AI_READY")
    write_snapshot(
        ROOT / "data/raw/oxford/ai_readiness_panel.csv",
        df,
        {
            "dataset": "AI_READINESS",
            "indicators": ["AI_READY"],
            "source_url": "https://oxfordinsights.com/ai-readiness/",
            "notes": (
                "Oxford Insights Government AI Readiness Index. "
                "2019 scores are on the original 0–10 published scale; "
                "2020/2022/2023 scores are on the 0–100 published scale from "
                "official report annex tables. 2021 left empty pending a "
                "verified annex extract (not fabricated). "
                "Do not mix scales without renormalisation."
            ),
            "publication_years": sorted(OXFORD_OFFICIAL.keys()),
            "countries": COUNTRIES,
            "scale_notes": {"2019": "0-10", "2020_2023": "0-100"},
            "source_reports": {
                "2019": "https://oxfordinsights.com/ (Government AI Readiness Index 2019)",
                "2020": "https://oxfordinsights.com/wp-content/uploads/2023/11/AIReadinessReport.pdf",
                "2022": "https://oxfordinsights.com/wp-content/uploads/2023/11/Government_AI_Readiness_2022_FV.pdf",
                "2023": "https://oxfordinsights.com/wp-content/uploads/2023/12/2023-Government-AI-Readiness-Index-2.pdf",
            },
        },
    )


def build_undp_panel() -> None:
    df = wide_from_nested(EDU_INDEX_OFFICIAL, "EDU_INDEX")
    write_snapshot(
        ROOT / "data/raw/undp/education_index_panel.csv",
        df,
        {
            "dataset": "HDR",
            "indicators": ["EDU_INDEX"],
            "source_url": "https://hdr.undp.org/",
            "notes": (
                "UNDP Human Development Report Education Index (0–1). "
                "Values compiled from HDR statistical annex / data centre "
                "published time series for selected years. Years without a "
                "compiled official observation are left empty/NaN."
            ),
            "publication_years": sorted(EDU_INDEX_OFFICIAL.keys()),
            "countries": COUNTRIES,
        },
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use embedded official tables only (no network refresh)",
    )
    args = parser.parse_args(argv)

    if args.skip_download:
        egdi, osi, egdi_note = EGDI_OFFICIAL, OSI_OFFICIAL, "Using embedded UN/QoG values (--skip-download)"
        ncsi, ncsi_note = NCSI_OFFICIAL, "Using embedded NCSI values (--skip-download)"
    else:
        print("Refreshing EGDI/OSI from QoG (optional)...")
        egdi, osi, egdi_note = try_refresh_egdi_osi_from_qog()
        print(egdi_note)
        print("Refreshing NCSI live ranking (optional)...")
        ncsi, ncsi_note = try_refresh_ncsi_live()
        print(ncsi_note)

    build_egdi_panel(egdi, osi, egdi_note)
    build_gii_panel()
    build_ncsi_panel(ncsi, ncsi_note)
    build_oxford_panel()
    build_undp_panel()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
