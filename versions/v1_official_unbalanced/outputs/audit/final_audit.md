# Final framework audit

_Generated: 2026-08-09T20:19:03.638716+00:00_

## Definition of Complete

- Criteria: **24** pass / **0** fail (of 24)
- Pass rate: **100.0%**
- Critical OK: **True**
- Fully complete: **True**

## Criteria

| ID | Layer | Status | Critical | Description | Artifact |
|---|---|---|---|---|---|
| `DATA_RAW` | data | **pass** | yes | Integrated raw panel exists | `data/raw/integrated_raw.parquet` |
| `DATA_VALIDATED` | data | **pass** | no | Validation report produced | `outputs/audit/validation_report.json` |
| `DATA_INTERIM` | data | **pass** | no | Validated interim parquet exists | `data/interim/` |
| `PROC_WIDE` | processing | **pass** | no | Processed wide panel exists | `data/processed/panel_wide.parquet` |
| `PROC_FEATURES` | processing | **pass** | no | Feature panel exists | `data/processed/panel_features.parquet` |
| `INDEX_DMI_PANEL` | index | **pass** | yes | DMI panel artifact exists | `data/processed/dmi_panel.parquet|outputs/data/dmi_panel.parquet` |
| `INDEX_ANALYSIS_PANEL` | index | **pass** | yes | Analysis panel ready for modeling | `data/processed/analysis_panel.parquet` |
| `INDEX_RELIABILITY` | index | **pass** | no | DMI reliability / sensitivity artifacts | `outputs/tables/dmi_*` |
| `STATS_DESCRIPTIVE` | statistics | **pass** | no | Descriptive statistics results present | `results/descriptive/` |
| `ECO_COEFFICIENTS` | econometric | **pass** | no | Econometric coefficients table | `results/econometrics/coefficients.csv` |
| `ECO_DIAGNOSTICS` | econometric | **pass** | no | Econometric diagnostics / Hausman | `results/econometrics/` |
| `ML_MODEL_COMPARISON` | ml | **pass** | yes | ML model comparison CSV | `results/machine_learning/model_comparison.csv` |
| `ML_BEST_MODEL` | ml | **pass** | no | Best model selection recorded | `outputs/models/best_model.json` |
| `ML_PREDICTIONS` | ml | **pass** | no | Prediction artifacts exported | `outputs/predictions/` |
| `XAI_ARTIFACTS` | xai | **pass** | no | Explainability artifacts under results/xai or outputs/xai | `results/xai|outputs/xai` |
| `CONV_ARTIFACTS` | convergence | **pass** | no | Convergence results available (or sigma series derived) | `results/convergence|outputs/figures/convergence_sigma.png` |
| `VIZ_FIGURES` | visualization | **pass** | no | Figures written to outputs/figures | `outputs/figures/` |
| `HYP_EVALUATION` | framework | **pass** | no | Hypothesis evaluation JSON | `results/hypotheses/hypothesis_evaluation.json` |
| `COMPARATIVE` | framework | **pass** | no | Comparative group analysis | `results/comparative/` |
| `DASHBOARD_EXPORT` | framework | **pass** | no | Dashboard data package exported | `outputs/dashboard/` |
| `FRAMEWORK_STATUS` | framework | **pass** | no | Intelligent framework status JSON | `outputs/dashboard/framework_status.json` |
| `REPORTS` | reporting | **pass** | no | Markdown reports generated | `outputs/reports/` |
| `REPRODUCIBILITY` | audit | **pass** | no | Reproducibility metadata captured | `outputs/audit/reproducibility.json` |
| `CONFIG_SNAPSHOT` | audit | **pass** | no | Config snapshot stored | `outputs/audit/config_snapshot/` |
