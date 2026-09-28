# Changelog

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
