# Changelog

## v3.1.0 — knowledge layer and visual analytics (9 October 2026)

The data layer, the DMI and all v3.0.0 results are unchanged; everything new
sits in `src/knowledge_layer/`, `ontology/`, `config/context_indicators.yaml`,
`config/vda_views.yaml` and `outputs/knowledge_layer/`.

- Context indicators Z1–Z6 from the WDI API (vintage 2026-10-08) with per-cell
  provenance (official / carried forward ≤ 2 years / missing); never imputed.
- Rolling-origin evaluation, 2017–2024 origins, 96 forecasts per model, against
  persistence; country-cluster bootstrap (2,000 draws). Ridge with the lagged DMI
  is the only model whose interval lies below zero on the reconstructed panel
  (ΔRMSE −0.283 [−0.417, −0.135]).
- Two-way fixed-effects association of Z1–Z6 with the DMI and the extended
  index DMI+.
- Ontology 2.0 (OWL 2 DL) aligned with RDF Data Cube, PROV-O, SKOS and DCAT;
  nominals replaced by value partitions and `owl:hasKey` moved to SHACL so that
  HermiT realises the 43,919-triple graph in about 20 s. SHACL shapes, seven
  Data Cube integrity constraints, 14 competency questions and a fault-injection
  test (SHACL detects 10 of 10 faults, HermiT 6 of 10).
- Power BI project (TMDL + PBIR) generated from `config/vda_views.yaml`; the
  views are also written to the graph as `dmi:AnalyticView`. The published
  project's `DataFolder` parameter is `C:\digital-maturity-framework\...`
  (`build_powerbi_project.py --data-folder`).
- `scripts/export_knowledge_figures.py`; record of the Windows re-run of
  v3.0.0 (`outputs/knowledge_layer/reproduction_v3.0.0_windows.json`).
- Dependencies: `owlready2`, `pyshacl`.

## v3.0.1 — release metadata (29 September 2026)

- Citation metadata only: release date and repository address in
  `CITATION.cff`, licence identifier in `.zenodo.json` (`mit`). Code, data
  and results are identical to v3.0.0.

## v3.0.0 — edition-aware re-harmonisation (September 2026)

Data
- All non-API indicators are rebuilt by `scripts/build_snapshots_v3.py` from the
  publishers' own files, with a per-value provenance record
  (`data/raw/<source>/*_value_provenance.csv`).
- GII: all editions 2012–2025 (previously 2013 and 2019–2023), each score accepted
  only with WIPO's official rank.
- AI Readiness: editions 2020–2024 of the three-pillar framework; the 2019 edition
  (0–10 scale, four clusters) and the 2025 edition (six pillars) are excluded
  instead of rescaled (`config/editions.yaml`).
- NCSI: NCSI 3.0 value in force on 31 December 2023–2025 from the dated version
  history. v2 had labelled the legacy index (archived 1 Sep 2023) as 2022 and a
  live 2026 score as 2023.
- Education index: one HDR 2025 vintage for 2012–2023 instead of values compiled
  from several reports.
- World Bank: the vintage retrieved on 2026-08-09 is frozen and served from disk.

Code
- Fix: cell-provenance labels were combined by row position after a re-sort and
  landed on other countries when the configured country order was not
  alphabetical (32 of 2016 cells in v2). Estimates were not affected.
- Edition filter, edition-boundary diagnostic, reconstruction scope
  (interior/backcast/forecast), official-only masked-cell benchmark,
  `scripts/run_track.py` and `scripts/export_preprocessing_evidence.py`.

Results
- New tracks `v2r_reproduction`, `v2rb_official_only`, `v3_edition_harmonised`
  (primary) and `v3b_official_only`; frozen v1, v1b and v2 are unchanged
  (git tag `v2-frozen`). The active `outputs/`, `results/` and `data/processed/`
  now hold the v3 run.
- Generated metadata (`reproducibility.json`, `final_audit.json`,
  `framework_status.json`, `source_metadata.json`) records directory and file
  names relative to the project instead of absolute local paths.

Reproducibility check
- Re-running `python run_pipeline.py --from RAW_DATA_ACQUIRED` at the project
  root reproduced `versions/v3_edition_harmonised`: processed panels, DMI, cell
  provenance and reconstruction scope agree to within 1e-13, fixed-effects
  coefficients to within 1e-8, and the multiple-imputation estimates exactly.
  In the hold-out comparison only the random forest differs (RMSE by at most
  0.007); the other seven models are identical.
