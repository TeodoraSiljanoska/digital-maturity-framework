#!/usr/bin/env python3
"""Build the v3 snapshots for indicators that have no statistical API.

Each value is written twice: into the wide snapshot CSV that the ingestion
adapters read, and into a long provenance sidecar that records the edition,
the reference year, the publication or version date, the exact source URL, the
retrieval channel, a checksum of the downloaded file and how the value was
verified. Values are read from the publishers' files by code; nothing is typed
in and nothing is invented. A value that cannot be retrieved stays empty.

Sources:
  X3, X10  UN E-Government Survey (EGDI, OSI) 2012-2024, UN DESA country pages,
           cross-checked against the UN_EGDI dataset served by the World Bank
           Data360 API (surveys up to 2020)
  C2       UNDP Human Development Report 2025 time series (education index
           computed with the HDR formula; the published HDI is reproduced as a check)
  X6       WIPO Global Innovation Index editions
  X7       Oxford Insights Government AI Readiness Index editions
  X5       e-Governance Academy NCSI 3.0 versioned country assessments

Usage:
    python scripts/build_snapshots_v3.py            # download or reuse cache, rebuild all
    python scripts/build_snapshots_v3.py --only un undp
    python scripts/build_snapshots_v3.py --offline  # use data/external_cache only
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external_cache"
RAW = ROOT / "data" / "raw"
START_YEAR, END_YEAR = 2012, 2025
COUNTRIES = ["MKD", "SRB", "SVN", "FRA", "GBR", "DEU", "USA", "CAN", "AUS", "BRA", "ARG", "IND"]
USER_AGENT = "Mozilla/5.0 (research data retrieval; digital-maturity-framework)"
OFFLINE = False

SIDE_CAR_COLUMNS = [
    "indicator_id",
    "indicator_code",
    "country_iso3",
    "reference_year",
    "value",
    "edition",
    "vintage_date",
    "source_url",
    "method",
    "retrieved_at",
    "file_sha256",
    "verification",
    "note",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, dest: Path, *, timeout: int = 180, retries: int = 4, params=None) -> Dict[str, str]:
    """Download ``url`` into the cache once; later runs reuse the cached copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = dest.with_name(dest.name + ".retrieved_at")
    if dest.exists() and stamp.exists():
        return {"path": str(dest), "sha256": sha256_of(dest), "retrieved_at": stamp.read_text().strip()}
    if OFFLINE:
        raise FileNotFoundError(f"offline mode and no cached copy of {url}")
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            stamp.write_text(utc_now())
            return {"path": str(dest), "sha256": sha256_of(dest), "retrieved_at": stamp.read_text().strip()}
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last


def page_text(path: Path) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(raw)))


def write_snapshot(
    folder: str,
    csv_name: str,
    records: List[Dict[str, Any]],
    value_columns: Dict[str, str],
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    """Write the wide adapter CSV, the long provenance sidecar and a metadata JSON."""
    out_dir = RAW / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(records, columns=SIDE_CAR_COLUMNS)
    long = long.sort_values(["indicator_code", "country_iso3", "reference_year"]).reset_index(drop=True)
    long.to_csv(out_dir / f"{Path(csv_name).stem}_value_provenance.csv", index=False)

    skeleton = pd.DataFrame(
        [(c, y) for c in COUNTRIES for y in range(START_YEAR, END_YEAR + 1)],
        columns=["country_iso3", "year"],
    )
    wide = skeleton.copy()
    for code, column in value_columns.items():
        part = long.loc[long["indicator_code"] == code, ["country_iso3", "reference_year", "value"]]
        part = part.rename(columns={"reference_year": "year", "value": column})
        wide = wide.merge(part, on=["country_iso3", "year"], how="left")
    wide.to_csv(out_dir / csv_name, index=False)

    meta = dict(metadata)
    meta.update(
        {
            "written_at": utc_now(),
            "rows": int(len(wide)),
            "values": int(long["value"].notna().sum()),
            "path": f"data/raw/{folder}/{csv_name}",
            "provenance_sidecar": f"data/raw/{folder}/{Path(csv_name).stem}_value_provenance.csv",
            "countries": COUNTRIES,
            "panel_years": [START_YEAR, END_YEAR],
            "fabricated_values": False,
        }
    )
    (out_dir / f"{Path(csv_name).stem}_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return wide


# ---------------------------------------------------------------------------
# UN E-Government Survey: EGDI (X3) and OSI (X10)
# ---------------------------------------------------------------------------

UN_PAGE_IDS = {
    "ARG": "7-Argentina",
    "AUS": "9-Australia",
    "BRA": "24-Brazil",
    "CAN": "31-Canada",
    "FRA": "61-France",
    "DEU": "65-Germany",
    "IND": "77-India",
    "SRB": "151-Serbia",
    "SVN": "156-Slovenia",
    "MKD": "170-North-Macedonia",
    "GBR": "182-United-Kingdom-of-Great-Britain-and-Northern-Ireland",
    "USA": "184-United-States-of-America",
}
UN_SURVEYS = [2012, 2014, 2016, 2018, 2020, 2022, 2024]


def _un_history(text: str, label: str) -> Dict[int, float]:
    """Read the '<label> 2024 2022 ... (Value) v2024 v2022 ...' history row of a country page."""
    match = re.search(
        label
        + r"\s+((?:(?:19|20)\d\d\s+){5,})(?:(?!\(Value\)).)*?\(Value\)\s+((?:[0-9]\.[0-9]+\s+){5,})",
        text,
        flags=re.S,
    )
    if not match:
        return {}
    years = [int(y) for y in match.group(1).split()]
    values = [float(v) for v in match.group(2).split()]
    return dict(zip(years, values))


def _data360(indicator: str, iso: str) -> Dict[int, float]:
    dest = CACHE / "un_egov" / f"data360_{indicator}_{iso}.json"
    info = fetch(
        f"https://data360api.worldbank.org/data360/data?DATABASE_ID=UN_EGDI&INDICATOR={indicator}&REF_AREA={iso}",
        dest,
        timeout=90,
    )
    payload = json.loads(Path(info["path"]).read_text(encoding="utf-8"))
    return {int(v["TIME_PERIOD"]): float(v["OBS_VALUE"]) for v in payload.get("value", [])}


def build_un() -> None:
    records: List[Dict[str, Any]] = []
    for iso, slug in UN_PAGE_IDS.items():
        url = f"https://publicadministration.un.org/egovkb/en-us/Data/Country-Information/id/{slug}"
        info = fetch(url, CACHE / "un_egov" / f"country_{iso}.html", timeout=240)
        text = page_text(Path(info["path"]))
        series = {
            ("EGDI", "X3", "UN_EGDI_EGDI"): _un_history(text, "E-Government Development Index"),
            ("OSI", "X10", "UN_EGDI_OSI"): _un_history(text, "Online Service Index"),
        }
        for (code, indicator_id, d360_code), history in series.items():
            check = _data360(d360_code, iso)
            for year in UN_SURVEYS:
                value = history.get(year)
                if value is None:
                    continue
                if year in check:
                    same = math.isclose(value, check[year], abs_tol=5e-5)
                    verification = "matches World Bank Data360 UN_EGDI" if same else (
                        f"MISMATCH with Data360 ({check[year]})"
                    )
                else:
                    verification = "UN country page only (Data360 series ends 2020)"
                records.append(
                    {
                        "indicator_id": indicator_id,
                        "indicator_code": code,
                        "country_iso3": iso,
                        "reference_year": year,
                        "value": value,
                        "edition": f"UN E-Government Survey {year}",
                        "vintage_date": str(year),
                        "source_url": url,
                        "method": "html",
                        "retrieved_at": info["retrieved_at"],
                        "file_sha256": info["sha256"],
                        "verification": verification,
                        "note": "",
                    }
                )
    mismatches = [r for r in records if r["verification"].startswith("MISMATCH")]
    if mismatches:
        raise RuntimeError(f"UN values disagree with Data360: {mismatches[:3]}")
    write_snapshot(
        "un_egdi",
        "egdi_panel.csv",
        records,
        {"EGDI": "EGDI", "OSI": "OSI"},
        {
            "dataset": "UN_EGOV",
            "indicators": ["EGDI", "OSI"],
            "publisher": "UN DESA, Division for Public Institutions and Digital Government",
            "source_url": "https://publicadministration.un.org/egovkb/en-us/data-center",
            "cross_check": "https://data360api.worldbank.org/data360/data?DATABASE_ID=UN_EGDI",
            "publication_years": UN_SURVEYS,
            "notes": "Biennial surveys; years between surveys are left empty.",
        },
    )
    print(f"UN EGDI/OSI: {len(records)} values ({sum('Data360' in r['verification'] and 'matches' in r['verification'] for r in records)} cross-checked)")


# ---------------------------------------------------------------------------
# UNDP HDR 2025: education index (C2)
# ---------------------------------------------------------------------------

HDR_URL = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"


def build_undp() -> None:
    info = fetch(HDR_URL, CACHE / "undp" / "HDR25_Composite_indices_complete_time_series.csv")
    hdr = pd.read_csv(info["path"], encoding="latin-1").set_index("iso3")
    records: List[Dict[str, Any]] = []
    worst = 0.0
    for iso in COUNTRIES:
        row = hdr.loc[iso]
        for year in range(START_YEAR, END_YEAR + 1):
            eys, mys = row.get(f"eys_{year}"), row.get(f"mys_{year}")
            if pd.isna(eys) or pd.isna(mys):
                continue
            # HDR technical notes: goalposts 18 (expected) and 15 (mean years of schooling).
            education = (min(float(mys), 15.0) / 15.0 + min(float(eys), 18.0) / 18.0) / 2.0
            # Check the formula by rebuilding the published HDI from its three dimensions.
            health = (float(row[f"le_{year}"]) - 20.0) / (85.0 - 20.0)
            income = (math.log(float(row[f"gnipc_{year}"])) - math.log(100.0)) / (
                math.log(75000.0) - math.log(100.0)
            )
            hdi = (min(health, 1.0) * education * min(income, 1.0)) ** (1.0 / 3.0)
            gap = abs(hdi - float(row[f"hdi_{year}"]))
            worst = max(worst, gap)
            records.append(
                {
                    "indicator_id": "C2",
                    "indicator_code": "EDU_INDEX",
                    "country_iso3": iso,
                    "reference_year": year,
                    "value": round(education, 6),
                    "edition": "Human Development Report 2025 (data to 2023)",
                    "vintage_date": "2025",
                    "source_url": HDR_URL,
                    "method": "csv",
                    "retrieved_at": info["retrieved_at"],
                    "file_sha256": info["sha256"],
                    "verification": f"published HDI reproduced within {gap:.4f}",
                    "note": "education index = (MYS/15 + EYS/18)/2 per HDR technical notes",
                }
            )
    if worst > 0.001:
        raise RuntimeError(f"HDI reproduction off by {worst:.4f}; education-index formula suspect")
    write_snapshot(
        "undp",
        "education_index_panel.csv",
        records,
        {"EDU_INDEX": "EDU_INDEX"},
        {
            "dataset": "HDR",
            "indicators": ["EDU_INDEX"],
            "publisher": "UNDP Human Development Report Office",
            "source_url": HDR_URL,
            "vintage": "HDR 2025 time series (single vintage for all years)",
            "notes": "Years after the last HDR data year are left empty.",
            "max_hdi_reproduction_error": worst,
        },
    )
    print(f"UNDP education index: {len(records)} values; max HDI reproduction error {worst:.5f}")


# ---------------------------------------------------------------------------
# WIPO Global Innovation Index (X6)
# ---------------------------------------------------------------------------

GII_DATA = "https://www.wipo.int/gii-ranking/data"
GII_REPORTS = {
    2012: "https://www.wipo.int/edocs/pubdocs/en/economics/gii/gii_2012.pdf",
    2013: "https://www.wipo.int/edocs/pubdocs/en/economics/gii/gii_2013.pdf",
    2014: "https://www.wipo.int/edocs/pubdocs/en/economics/gii/gii_2014.pdf",
    2015: "https://www.wipo.int/edocs/pubdocs/en/wipo_gii_2015.pdf",
    2016: "https://www.wipo.int/edocs/pubdocs/en/wipo_pub_gii_2016.pdf",
    2017: "https://www.wipo.int/edocs/pubdocs/en/wipo_pub_gii_2017.pdf",
    2018: "https://www.wipo.int/edocs/pubdocs/en/wipo_pub_gii_2018.pdf",
    2019: "https://www.wipo.int/edocs/pubdocs/en/wipo_pub_gii_2019.pdf",
}
GII_NAMES = {
    "MKD": [
        "North Macedonia",
        "TFYR of Macedonia",
        "The former Yugoslav Republic of Macedonia",
        "The Former Yugoslav Republic of Macedonia",
        "Macedonia, FYR",
        "FYR Macedonia",
    ],
    "SRB": ["Serbia"],
    "SVN": ["Slovenia"],
    "FRA": ["France"],
    "GBR": ["United Kingdom"],
    "DEU": ["Germany"],
    "USA": ["United States of America", "United States"],
    "CAN": ["Canada"],
    "AUS": ["Australia"],
    "BRA": ["Brazil"],
    "ARG": ["Argentina"],
    "IND": ["India"],
}


def _gii_official_ranks() -> Dict[tuple, int]:
    info = fetch(f"{GII_DATA}/economy_id_2025.csv", CACHE / "gii" / "economy_id_2025.csv")
    table = pd.read_csv(info["path"]).set_index("ISO3")
    ranks: Dict[tuple, int] = {}
    for iso in COUNTRIES:
        for year in range(2011, 2026):
            value = table.loc[iso].get(f"GII{str(year)[2:]}")
            if pd.notna(value):
                ranks[(iso, year)] = int(value)
    return ranks


def _gii_from_report(year: int, text: str, official_rank: Dict[tuple, int]) -> Dict[str, float]:
    """Score of each country in the report's rankings table, accepted only if its rank is WIPO's."""
    found: Dict[str, float] = {}
    for iso, names in GII_NAMES.items():
        expected = official_rank.get((iso, year))
        for name in names:
            words = r"\s+".join(re.escape(w) for w in name.split())
            for match in re.finditer(words + r"\s*\n\s*(\d{1,3}\.\d{1,2})\s*\n\s*(\d{1,3})\s*\n", text):
                score, rank = float(match.group(1)), int(match.group(2))
                if expected is not None and rank == expected and 0 < score <= 100:
                    found.setdefault(iso, score)
            if iso in found:
                break
    return found


def build_gii() -> None:
    import pymupdf

    ranks = _gii_official_ranks()
    records: List[Dict[str, Any]] = []

    for year, url in GII_REPORTS.items():
        info = fetch(url, CACHE / "gii" / f"gii_{year}.pdf", timeout=900)
        with pymupdf.open(info["path"]) as doc:
            text = "\n".join(page.get_text() for page in doc)
        scores = _gii_from_report(year, text, ranks)
        for iso in COUNTRIES:
            if iso not in scores:
                print(f"  GII {year}: no rank-verified score for {iso}")
                continue
            records.append(
                {
                    "indicator_id": "X6",
                    "indicator_code": "GII_SCORE",
                    "country_iso3": iso,
                    "reference_year": year,
                    "value": scores[iso],
                    "edition": f"Global Innovation Index {year}",
                    "vintage_date": str(year),
                    "source_url": url,
                    "method": "pdf",
                    "retrieved_at": info["retrieved_at"],
                    "file_sha256": info["sha256"],
                    "verification": f"rank {ranks[(iso, year)]} matches WIPO economy_id_2025.csv",
                    "note": "overall score from the report's rankings table",
                }
            )

    info = fetch(f"{GII_DATA}/model_pageone_2025.csv", CACHE / "gii" / "model_pageone_2025.csv")
    model = pd.read_csv(info["path"])
    index = model.loc[(model["CODE"] == "Index") & model["ISO3"].isin(COUNTRIES)]
    for row in index.itertuples():
        year, iso = int(row.GIIYR), str(row.ISO3)
        expected = ranks.get((iso, year))
        if expected is not None and int(row.RANK) != expected:
            raise RuntimeError(f"GII {year} {iso}: rank {row.RANK} disagrees with economy_id {expected}")
        records.append(
            {
                "indicator_id": "X6",
                "indicator_code": "GII_SCORE",
                "country_iso3": iso,
                "reference_year": year,
                "value": round(float(row.SCORE), 4),
                "edition": f"Global Innovation Index {year}",
                "vintage_date": str(year),
                "source_url": f"{GII_DATA}/model_pageone_2025.csv",
                "method": "csv",
                "retrieved_at": info["retrieved_at"],
                "file_sha256": info["sha256"],
                "verification": f"rank {int(row.RANK)} matches WIPO economy_id_2025.csv",
                "note": "GII Innovation Ecosystems & Data Explorer 2025 data file",
            }
        )

    write_snapshot(
        "gii",
        "gii_panel.csv",
        records,
        {"GII_SCORE": "GII_SCORE"},
        {
            "dataset": "GII",
            "indicators": ["GII_SCORE"],
            "publisher": "World Intellectual Property Organization (WIPO)",
            "source_url": "https://www.wipo.int/global_innovation_index/",
            "editions": sorted({r["reference_year"] for r in records}),
            "notes": (
                "Editions 2012-2019 from the report rankings tables, 2020-2025 from the "
                "GII data explorer file. The publisher cautions that scores are not fully "
                "comparable across editions because the framework and data are revised yearly."
            ),
        },
    )
    print(f"GII: {len(records)} values over {len({r['reference_year'] for r in records})} editions")


# ---------------------------------------------------------------------------
# Oxford Insights Government AI Readiness Index (X7)
# ---------------------------------------------------------------------------

OXFORD = "https://oxfordinsights.com/wp-content/uploads"
OXFORD_FILES = {
    2019: f"{OXFORD}/2023/12/ai-gov-readiness-report_v08.pdf",
    2020: f"{OXFORD}/2023/11/AIReadinessReport.pdf",
    2021: f"{OXFORD}/2023/11/Government_AI_Readiness_21.pdf",
    2022: f"{OXFORD}/2023/11/Government_AI_Readiness_2022_FV.pdf",
    2023: f"{OXFORD}/2023/12/2023-Government-AI-Readiness-Index-2.pdf",
    2024: f"{OXFORD}/2024/12/2024-Government-AI-Readiness-Index-2.pdf",
}
OXFORD_2023_DATA = f"{OXFORD}/2024/02/2023-Government-AI-Readiness-Index-Public-Indicator-Data.xlsx"
OXFORD_NAMES = {
    "MKD": ["North Macedonia", "Republic of North Macedonia"],
    "SRB": ["Serbia"],
    "SVN": ["Slovenia"],
    "FRA": ["France"],
    "GBR": ["United Kingdom of Great Britain and Northern Ireland", "United Kingdom"],
    "DEU": ["Germany"],
    "USA": ["United States of America", "United States"],
    "CAN": ["Canada"],
    "AUS": ["Australia"],
    "BRA": ["Brazil"],
    "ARG": ["Argentina"],
    "IND": ["India"],
}
NUM = r"\s*\n\s*(\d{1,3}\.\d{2,3})"


def _name_pattern(name: str) -> str:
    return r"\s+".join(re.escape(w) for w in name.split())


def _oxford_pillar_rows(text: str, with_rank: bool) -> Dict[str, Dict[str, float]]:
    """Total plus three pillar scores; a row is accepted only if the total is the pillar mean."""
    found: Dict[str, Dict[str, float]] = {}
    for iso, names in OXFORD_NAMES.items():
        for name in names:
            lead = r"\n\s*(\d{1,3})\s*\n\s*" if with_rank else r"\n\s*"
            pattern = lead + _name_pattern(name) + NUM * 4
            for m in re.finditer(pattern, text):
                values = [float(v) for v in m.groups()[1 if with_rank else 0:]]
                total, pillars = values[0], values[1:]
                if abs(total - sum(pillars) / 3.0) <= 0.011:
                    found[iso] = {"total": total, "pillars": pillars}
                    break
            if iso in found:
                break
    return found


def _oxford_ranked_list(text: str, *, decimals: int, scale_max: float) -> Dict[str, float]:
    """'rank / country / score' listing (2019 report table, 2020 Annex 1)."""
    found: Dict[str, float] = {}
    for iso, names in OXFORD_NAMES.items():
        for name in names:
            pattern = _name_pattern(name) + r"\s*\n\s*(\d{1,3}\.\d{" + str(decimals) + r"})\s*\n"
            for m in re.finditer(pattern, text):
                value = float(m.group(1))
                if 0 < value <= scale_max:
                    found[iso] = value
                    break
            if iso in found:
                break
    return found


def build_oxford() -> None:
    import openpyxl
    import pymupdf

    records: List[Dict[str, Any]] = []

    def add(iso: str, year: int, value: float, url: str, method: str, info: Dict[str, str], check: str, note: str = "") -> None:
        records.append(
            {
                "indicator_id": "X7",
                "indicator_code": "AI_READY",
                "country_iso3": iso,
                "reference_year": year,
                "value": value,
                "edition": f"Government AI Readiness Index {year}",
                "vintage_date": str(year),
                "source_url": url,
                "method": method,
                "retrieved_at": info["retrieved_at"],
                "file_sha256": info["sha256"],
                "verification": check,
                "note": note,
            }
        )

    texts: Dict[int, str] = {}
    infos: Dict[int, Dict[str, str]] = {}
    for year, url in OXFORD_FILES.items():
        infos[year] = fetch(url, CACHE / "oxford" / f"oxford_{year}.pdf", timeout=900)
        with pymupdf.open(infos[year]["path"]) as doc:
            pages = [page.get_text() for page in doc]
        if year == 2020:
            # The contents page also names Annex 1, so start at the heading that
            # is followed by ranked rows ("rank / country / score").
            ranked_row = re.compile(r"\n\d{1,3}\n[^\n]+\n\d{1,3}\.\d{3}\n")
            start = next(
                i for i, t in enumerate(pages)
                if "Annex 1: Full Rankings" in t and len(ranked_row.findall(t)) >= 10
            )
            stop = next((i for i, t in enumerate(pages) if i > start and "Annex 2" in t), len(pages))
            pages = pages[start:stop]
        texts[year] = "\n" + "\n".join(pages)

    # 2019: 0-10 scale, eleven metrics in four clusters (excluded later by the edition filter).
    for iso, value in _oxford_ranked_list(texts[2019], decimals=3, scale_max=10.0).items():
        add(iso, 2019, value, OXFORD_FILES[2019], "pdf", infos[2019],
            "report ranking table", "0-10 scale; framework differs from 2020 onwards")

    # 2020: Annex 1 full rankings, three decimals.
    annex = _oxford_ranked_list(texts[2020], decimals=3, scale_max=100.0)
    for iso, value in annex.items():
        add(iso, 2020, value, OXFORD_FILES[2020], "pdf", infos[2020], "Annex 1 full rankings")

    # 2021-2022 and 2024: total alongside the three pillar scores.
    for year, with_rank in ((2021, True), (2022, True), (2024, False)):
        for iso, row in _oxford_pillar_rows(texts[year], with_rank=with_rank).items():
            add(iso, year, row["total"], OXFORD_FILES[year], "pdf", infos[year],
                "total equals mean of the three pillar scores")

    # 2023: published data file (full precision), cross-checked against the report table.
    info = fetch(OXFORD_2023_DATA, CACHE / "oxford" / "oxford_2023_data.xlsx")
    sheet = openpyxl.load_workbook(info["path"], read_only=True, data_only=True)["Global rankings"]
    totals = {str(r[1]).strip(): float(r[2]) for r in sheet.iter_rows(min_row=2, values_only=True) if r[1] and r[2] is not None}
    report_2023 = _oxford_pillar_rows(texts[2023], with_rank=True)
    for iso, names in OXFORD_NAMES.items():
        name = next((n for n in names if n in totals), None)
        if name is None:
            continue
        value = round(totals[name], 4)
        in_report = report_2023.get(iso, {}).get("total")
        check = (
            f"matches report table ({in_report})"
            if in_report is not None and abs(in_report - value) <= 0.006
            else "data file only"
        )
        add(iso, 2023, value, OXFORD_2023_DATA, "xlsx", info, check)

    missing = [(iso, y) for y in (2019, 2020, 2021, 2022, 2023, 2024) for iso in COUNTRIES
               if not any(r["country_iso3"] == iso and r["reference_year"] == y for r in records)]
    for iso, year in missing:
        print(f"  Oxford {year}: no verified value for {iso}")

    write_snapshot(
        "oxford",
        "ai_readiness_panel.csv",
        records,
        {"AI_READY": "AI_READY"},
        {
            "dataset": "AI_READINESS",
            "indicators": ["AI_READY"],
            "publisher": "Oxford Insights",
            "source_url": "https://oxfordinsights.com/ai-readiness/",
            "editions": sorted({r["reference_year"] for r in records}),
            "notes": (
                "2019 edition: 0-10 scale, four clusters. 2020-2024 editions: 0-100 scale, "
                "three pillars (Government, Technology Sector, Data and Infrastructure). "
                "2025 edition (six pillars, no overall score in the report tables) not retrieved."
            ),
        },
    )
    print(f"Oxford Insights: {len(records)} values")


# ---------------------------------------------------------------------------
# e-Governance Academy National Cyber Security Index (X5)
# ---------------------------------------------------------------------------

NCSI = "https://ncsi.ega.ee"
NCSI_ISO2 = {
    "MKD": ("mk", "North Macedonia"),
    "SRB": ("rs", "Serbia"),
    "SVN": ("si", "Slovenia"),
    "FRA": ("fr", "France"),
    "GBR": ("gb", "United Kingdom"),
    "DEU": ("de", "Germany"),
    "USA": ("us", "United States"),
    "CAN": ("ca", "Canada"),
    "AUS": ("au", "Australia"),
    "BRA": ("br", "Brazil"),
    "ARG": ("ar", "Argentina"),
    "IND": ("in", "India"),
}


def _ncsi_timeline(raw_html: str, variable: str = "ncsi_country_timeline") -> List[Dict[str, Any]]:
    match = re.search(r"var " + variable + r"\s*=\s*(\[[^\]]*\])", raw_html)
    if not match:
        return []
    entries = json.loads(match.group(1))
    return [
        {"date": e["date"], "value": float(e["value"])}
        for e in entries
        if not e.get("hidden") and e.get("value") not in (None, "")
    ]


def _first_ranked(raw_html: str) -> Optional[str]:
    """Date on which the country first appears in the ranking; before it the page holds a placeholder."""
    ranks = sorted(_ncsi_timeline(raw_html, "ncsi_country_ranking_timeline"), key=lambda e: e["date"])
    return ranks[0]["date"] if ranks else None


def build_ncsi() -> None:
    records: List[Dict[str, Any]] = []
    for iso, (iso2, name) in NCSI_ISO2.items():
        url = f"{NCSI}/country/{iso2}/"
        info = fetch(url, CACHE / "ncsi" / f"country_{iso}.html", timeout=120)
        raw_html = Path(info["path"]).read_text(encoding="utf-8", errors="replace")
        timeline = sorted(_ncsi_timeline(raw_html), key=lambda e: e["date"])
        if not timeline:
            print(f"  NCSI: no dated versions for {iso}")
            continue
        # The page headline must equal the latest dated version.
        headline = re.search(r"\d+\.\s*" + re.escape(name) + r"\s+(\d+\.\d+)", page_text(Path(info["path"])))
        latest = timeline[-1]["value"]
        if headline is None or abs(float(headline.group(1)) - latest) > 0.005:
            raise RuntimeError(f"NCSI {iso}: headline score disagrees with latest dated version")
        first_ranked = _first_ranked(raw_html)
        for year in (2023, 2024, 2025):
            year_end = f"{year}-12-31"
            in_force = [e for e in timeline if e["date"] <= year_end]
            if not in_force:
                continue
            if first_ranked is None or first_ranked > year_end:
                # e.g. a 0.00 shown before the country's first assessment: not a publication
                print(f"  NCSI {iso} {year}: not yet ranked (first ranked {first_ranked}); left empty")
                continue
            version = in_force[-1]
            records.append(
                {
                    "indicator_id": "X5",
                    "indicator_code": "NCSI_SCORE",
                    "country_iso3": iso,
                    "reference_year": year,
                    "value": version["value"],
                    "edition": "NCSI 3.0",
                    "vintage_date": version["date"],
                    "source_url": url,
                    "method": "html",
                    "retrieved_at": info["retrieved_at"],
                    "file_sha256": info["sha256"],
                    "verification": "dated version history; latest version equals page headline",
                    "note": f"value in force on {year}-12-31 (version of {version['date']})",
                }
            )

    # Legacy index (2016-2023 methodology), archived by the publisher on 1 September 2023.
    archive_url = f"{NCSI}/ncsi-index/?archive=1"
    info = fetch(archive_url, CACHE / "ncsi" / "legacy_archive.html", timeout=120)
    archive_html = Path(info["path"]).read_text(encoding="utf-8", errors="replace")
    for iso, (_, name) in NCSI_ISO2.items():
        match = re.search(
            re.escape(name) + r"</a></td>\s*<td[^>]*>\s*<strong[^>]*>\s*([\d.]+)", archive_html, flags=re.I
        ) or re.search(re.escape(name) + r"</a>.*?<strong[^>]*>\s*([\d.]+)", archive_html, flags=re.I | re.S)
        if not match:
            print(f"  NCSI legacy: no value for {iso}")
            continue
        records.append(
            {
                "indicator_id": "X5",
                "indicator_code": "NCSI_LEGACY_SCORE",
                "country_iso3": iso,
                "reference_year": 2023,
                "value": float(match.group(1)),
                "edition": "NCSI 2016-2023 (legacy methodology)",
                "vintage_date": "2023-09-01",
                "source_url": archive_url,
                "method": "html",
                "retrieved_at": info["retrieved_at"],
                "file_sha256": info["sha256"],
                "verification": "publisher archive page",
                "note": "not part of the NCSI 3.0 series; kept for the edition-break comparison",
            }
        )

    write_snapshot(
        "ncsi",
        "ncsi_panel.csv",
        records,
        {"NCSI_SCORE": "NCSI_SCORE", "NCSI_LEGACY_SCORE": "NCSI_LEGACY_SCORE"},
        {
            "dataset": "NCSI",
            "indicators": ["NCSI_SCORE"],
            "publisher": "e-Governance Academy",
            "source_url": NCSI,
            "notes": (
                "NCSI_SCORE: NCSI 3.0 value in force on 31 December of each year, from the dated "
                "version history on each country page. NCSI_LEGACY_SCORE: legacy index as archived "
                "on 1 September 2023; different methodology, not read by the ingestion adapter."
            ),
        },
    )
    print(f"NCSI: {sum(r['indicator_code'] == 'NCSI_SCORE' for r in records)} NCSI 3.0 values, "
          f"{sum(r['indicator_code'] == 'NCSI_LEGACY_SCORE' for r in records)} legacy values")


BUILDERS: Dict[str, Callable[[], None]] = {
    "un": build_un,
    "undp": build_undp,
    "gii": build_gii,
    "oxford": build_oxford,
    "ncsi": build_ncsi,
}


def main(argv: Optional[Iterable[str]] = None) -> int:
    global OFFLINE
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", choices=sorted(BUILDERS), help="subset of sources")
    parser.add_argument("--offline", action="store_true", help="use cached downloads only")
    args = parser.parse_args(list(argv) if argv is not None else None)
    OFFLINE = args.offline
    for name in args.only or list(BUILDERS):
        BUILDERS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
