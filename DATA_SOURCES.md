# Data sources, editions and terms of use

Every value that enters the panel comes from one of the sources below. The
per-value record (edition, reference year, vintage date, URL, retrieval channel,
SHA-256 of the downloaded file, verification) is in
`data/raw/<source>/*_value_provenance.csv`; World Bank values carry their
retrieval time in `data/raw/world_bank/wdi_vintage_20260809.*`.

Third-party values remain under their publishers' terms. The licence in
`LICENSE-DATA` covers only what this project derives from them (the Digital
Maturity Index, cell provenance, reconstruction scope, processed panels and
results), to the extent of the project's rights.

| Code | Indicator | Publisher and channel | Editions / years used | Publisher's terms |
|---|---|---|---|---|
| X1, X2, X4, X8, X9, C1 | WDI series `IT.CEL.SETS.P2`, `IT.NET.BBND.P2`, `IT.NET.USER.ZS`, `GB.XPD.RSDV.GD.ZS`, `BX.GSR.CCIS.ZS`, `NY.GDP.PCAP.PP.KD` | World Bank, World Development Indicators API v2 (`https://api.worldbank.org/v2`) | vintage retrieved 2026-08-09, years 2012–2025 | CC BY 4.0 (World Bank data) |
| X3, X10 | E-Government Development Index, Online Service Index | UN DESA, E-Government Survey country pages (`publicadministration.un.org/egovkb`); surveys 2012–2020 cross-checked against the World Bank Data360 API, dataset `UN_EGDI` | surveys 2012, 2014, 2016, 2018, 2020, 2022, 2024 | United Nations terms of use; cite UN DESA |
| X5 | National Cyber Security Index (NCSI 3.0) | e-Governance Academy, dated version history on each country page (`ncsi.ega.ee/country/<iso2>/`) | value in force on 31 December 2023, 2024, 2025; legacy index (archived 1 Sep 2023) kept as `NCSI_LEGACY_SCORE`, not used | no licence stated on the pages; reproduced for non-commercial research with attribution |
| X6 | Global Innovation Index, overall score | WIPO: report ranking tables 2012–2019 (PDF), GII Innovation Ecosystems & Data Explorer file `model_pageone_2025.csv` for 2020–2025 | editions 2012–2025 | report editions: CC BY-NC-ND 3.0 IGO (Cornell University, INSEAD, WIPO); cite WIPO |
| X7 | Government AI Readiness Index | Oxford Insights reports 2019–2024 (PDF) and 2023 public indicator data (XLSX) | 2020–2024 (three-pillar framework); 2019 and 2025 editions excluded | CC BY-SA 4.0 (Oxford Insights) |
| C2 | Education index | UNDP Human Development Report Office, `HDR25_Composite_indices_complete_time_series.csv` (EYS, MYS) | 2012–2023, single HDR 2025 vintage | CC BY 3.0 IGO (UNDP HDRO) |

Edition families, excluded editions and the reasons are declared in
`config/editions.yaml`. Downloaded publisher files are cached in
`data/external_cache/` and are not redistributed; `scripts/build_snapshots_v3.py`
downloads them again from the addresses above.
