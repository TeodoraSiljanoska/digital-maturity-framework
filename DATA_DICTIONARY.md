# Data dictionary

## Keys and grid

All panels are keyed by `country_iso3` (ISO 3166-1 alpha-3) and `year`
(2012–2025): 12 countries × 14 years = 168 rows. Groups (`group_id`):
`balkan` (MKD, SRB, SVN), `developed_eu` (FRA, GBR, DEU), `developed_non_eu`
(USA, CAN, AUS), `developing` (BRA, ARG, IND).

## Indicators

| Column | Meaning | Unit |
|---|---|---|
| X1 | Mobile cellular subscriptions | per 100 people |
| X2 | Fixed broadband subscriptions | per 100 people |
| X3 | UN E-Government Development Index | 0–1 |
| X4 | Individuals using the Internet (digital-skills proxy) | % of population |
| X5 | National Cyber Security Index (NCSI 3.0) | 0–100 |
| X6 | Global Innovation Index, overall score | 0–100 |
| X7 | Government AI Readiness Index | 0–100 |
| X8 | Research and development expenditure | % of GDP |
| X9 | ICT service exports | % of service exports |
| X10 | UN Online Service Index | 0–1 |
| C1 | GDP per capita, PPP (constant international $); natural log in processed panels | log |
| C2 | Education index (UNDP HDR) | 0–1 |
| DMI | Digital Maturity Index: min–max 0–100 per indicator, mean within five pillars, pillars weighted 0.2 (renormalised over available pillars) | 0–100 |

`*_lag1` columns hold the value of the previous year for the same country;
`*_yoy` columns hold year-on-year growth; `pillar_*` columns hold pillar scores.

## Long raw format (`data/raw/integrated_raw.*`, `data/raw/*/raw_long.*`)

`country_iso3, year, indicator_id, indicator_code, value, source, retrieved_at`

## Per-value provenance of snapshot sources (`data/raw/<source>/*_value_provenance.csv`)

| Column | Meaning |
|---|---|
| indicator_id / indicator_code | research variable (X…, C…) and publisher code |
| country_iso3, reference_year, value | the published value and the panel year it is assigned to |
| edition | publisher edition or version family |
| vintage_date | publication year, or version date for continuously updated indices |
| source_url | exact address the value was read from |
| method | `html`, `pdf`, `csv`, `xlsx` |
| retrieved_at | UTC time of download |
| file_sha256 | checksum of the downloaded file |
| verification | how the value was checked (second channel, official rank, pillar mean, HDI reproduction, dated history) |

## Cell provenance (`data/processed/cell_provenance.parquet`)

One column per indicator, one row per country–year, with states:

| State | Meaning |
|---|---|
| `official` | value published for that country and year |
| `carried_forward` | copied from the same country's nearest publication at most two years away |
| `mice_imputed` | reconstructed by chained-equation imputation (MICE arm only) |
| `missing` | no value (official-only arm) |

`data/processed/cell_reconstruction_scope.parquet` locates each reconstructed
cell in its country series: `interior` (between two publications), `backcast`
(before the first), `forecast` (after the last), `unobserved_series` (no
publication at all).

## Result tracks (`versions/`)

| Track | Configuration and data | Missing-data arm |
|---|---|---|
| `v1_official_unbalanced` | original v1 (historical) | none |
| `v1b_official_corrected` | v2 data and corrections | official only |
| `v2_mice_imputed` | frozen v2 (doctoral project, August 2026) | MICE |
| `v2r_reproduction` | v2 configuration and data, current code and environment | MICE |
| `v2rb_official_only` | as v2r | official only |
| `v3_edition_harmonised` | edition-aware snapshots (primary) | MICE |
| `v3b_official_only` | as v3 | official only |

Each track holds `outputs/`, `results/`, `data_processed/`, `data_raw/`,
`config/` and a `VERSION.txt` generated from its own artifacts.
