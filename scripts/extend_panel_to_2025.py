#!/usr/bin/env python3
"""Extend curated snapshots to 2025 with official values only.

Superseded by scripts/build_snapshots_v3.py from v3 onwards; kept so that the
v2 snapshots (tag v2-frozen) remain reproducible.

- Adds years 2024–2025 to all snapshot panels.
- Inserts UN EGDI/OSI 2024 from UN E-Government Survey 2024 (official).
- Leaves 2025 EGDI/OSI empty (no survey).
- Does NOT invent GII/NCSI/AI/Education values for 2024–2025.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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

# UN E-Government Survey 2024 — OSI, EGDI (and EGDI 2022 cross-check in comments)
# Sources: UN DESA E-Government Survey 2024 tables / Data Center.
EGDI_2024 = {
    "MKD": 0.7070,
    "SRB": 0.8618,
    "SVN": 0.8759,
    "FRA": 0.8744,
    "GBR": 0.9577,
    "DEU": 0.9382,
    "USA": 0.9195,
    "CAN": 0.8452,
    "AUS": 0.9577,
    "BRA": 0.8403,
    "ARG": 0.8573,
    "IND": 0.6678,
}
OSI_2024 = {
    "MKD": 0.6642,
    "SRB": 0.8540,
    "SVN": 0.8640,
    "FRA": 0.8440,
    "GBR": 0.9535,
    "DEU": 0.9238,
    "USA": 0.9136,
    "CAN": 0.8552,
    "AUS": 0.9222,
    "BRA": 0.9063,
    "ARG": 0.7965,
    "IND": 0.8184,
}


def _skeleton(value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for iso in COUNTRIES:
        for year in range(START_YEAR, END_YEAR + 1):
            row = {"country_iso3": iso, "year": year}
            for c in value_cols:
                row[c] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


def extend_wide(path: Path, value_cols: list[str], updates: dict[int, dict[str, dict[str, float]]] | None = None) -> pd.DataFrame:
    """Merge existing snapshot into full 2012–2025 skeleton; apply official updates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base = _skeleton(value_cols)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["year"].between(START_YEAR, END_YEAR)].copy()
        for col in value_cols:
            if col not in old.columns:
                old[col] = pd.NA
        old = old[["country_iso3", "year", *value_cols]]
        merged = base.merge(old, on=["country_iso3", "year"], how="left", suffixes=("", "_old"))
        for col in value_cols:
            merged[col] = merged[f"{col}_old"].combine_first(merged[col])
            merged = merged.drop(columns=[f"{col}_old"])
    else:
        merged = base

    if updates:
        for year, ind_map in updates.items():
            for col, country_vals in ind_map.items():
                for iso, val in country_vals.items():
                    mask = (merged["country_iso3"] == iso) & (merged["year"] == year)
                    merged.loc[mask, col] = val

    merged = merged.sort_values(["country_iso3", "year"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def write_meta(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    egdi_path = ROOT / "data/raw/un_egdi/egdi_panel.csv"
    egdi = extend_wide(
        egdi_path,
        ["EGDI", "OSI"],
        updates={2024: {"EGDI": EGDI_2024, "OSI": OSI_2024}},
    )
    write_meta(
        ROOT / "data/raw/un_egdi/egdi_panel_metadata.json",
        {
            "dataset": "UN_EGOV",
            "indicators": ["EGDI", "OSI"],
            "source_url": "https://publicadministration.un.org/egovkb/",
            "secondary_source_url": "https://desapublications.un.org/publications/un-e-government-survey-2024",
            "notes": (
                "Biennial UN E-Government Survey EGDI/OSI. "
                "2024 values from UN E-Government Survey 2024 official tables/Data Center. "
                "2025 left empty (no survey). Non-survey years remain NaN."
            ),
            "publication_years": [2012, 2014, 2016, 2018, 2020, 2022, 2024],
            "panel_years": [START_YEAR, END_YEAR],
            "countries": COUNTRIES,
            "rows": int(len(egdi)),
            "path": str(egdi_path.relative_to(ROOT)),
            "egdi_2024_non_null": int(egdi.loc[egdi.year == 2024, "EGDI"].notna().sum()),
        },
    )

    # Other snapshots: extend years only; do not invent new scores
    for rel, cols, meta_name, notes in [
        (
            "data/raw/gii/gii_panel.csv",
            ["GII_SCORE"],
            "gii_panel_metadata.json",
            "WIPO GII overall scores. 2024–2025 left empty unless previously compiled official values exist.",
        ),
        (
            "data/raw/ncsi/ncsi_panel.csv",
            ["NCSI_SCORE"],
            "ncsi_panel_metadata.json",
            "NCSI scores. 2024–2025 left empty unless previously compiled official values exist.",
        ),
        (
            "data/raw/oxford/ai_readiness_panel.csv",
            ["AI_READY"],
            "ai_readiness_panel_metadata.json",
            "Oxford Insights AI Readiness. 2024–2025 left empty unless previously compiled official values exist.",
        ),
        (
            "data/raw/undp/education_index_panel.csv",
            ["EDU_INDEX"],
            "education_index_panel_metadata.json",
            "UNDP Education Index. 2023–2025 left empty unless previously compiled official values exist.",
        ),
    ]:
        path = ROOT / rel
        df = extend_wide(path, cols, updates=None)
        write_meta(
            path.with_name(meta_name),
            {
                "dataset": path.parent.name.upper(),
                "indicators": cols,
                "notes": notes,
                "panel_years": [START_YEAR, END_YEAR],
                "countries": COUNTRIES,
                "rows": int(len(df)),
                "path": str(path.relative_to(ROOT)),
                "extended_to": END_YEAR,
                "fabricated_values": False,
            },
        )
        print(f"Extended {rel}: rows={len(df)} years={df.year.min()}-{df.year.max()}")

    print(
        "EGDI 2024 non-null:",
        int(egdi.loc[egdi.year == 2024, "EGDI"].notna().sum()),
        "EGDI 2025 non-null:",
        int(egdi.loc[egdi.year == 2025, "EGDI"].notna().sum()),
    )


if __name__ == "__main__":
    main()
