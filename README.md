# Digital Maturity Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23043842.svg)](https://doi.org/10.5281/zenodo.23043842)

Archived on Zenodo: concept DOI [10.5281/zenodo.23043842](https://doi.org/10.5281/zenodo.23043842) (all versions); version 3.0.0, the state reported in the AIIT 2026 paper: [10.5281/zenodo.23043843](https://doi.org/10.5281/zenodo.23043843).

Intelligent AI framework for **visual analytics, monitoring, and prediction** of digital transformation (Digital Maturity Index — DMI).

The project builds a multi-country panel (2012–2025), constructs a composite DMI, runs descriptive / econometric / ML analyses, evaluates research hypotheses, and exposes results through figures, reports, and a Streamlit dashboard.

## Version 3: edition-aware, provenance-tracked data layer

- Every value from a source without a statistical API is read from the publisher's
  own files by `scripts/build_snapshots_v3.py` and stored with its edition,
  reference year, vintage date, URL, retrieval channel, file checksum and a
  verification note (`data/raw/<source>/*_value_provenance.csv`).
- Editions built under different frameworks are kept apart
  (`config/editions.yaml`); an edition-boundary diagnostic reports declared and
  detected breaks (`outputs/audit/edition_breaks_{raw,filtered}.json`).
- Every cell is labelled `official`, `carried_forward`, `mice_imputed` or
  `missing` (`data/processed/cell_provenance.parquet`), and reconstructed cells are
  located as interior, backcast or forecast (`cell_reconstruction_scope.parquet`).
- The World Bank vintage of 2026-08-09 is frozen, so re-runs reproduce the same
  values (`config/sources.yaml`, `use_cached_raw`).

See `CHANGELOG.md`, `DATA_SOURCES.md` (publishers and terms of use) and
`DATA_DICTIONARY.md` (columns, provenance states, tracks).

### Reproduce the tracks

```bash
python scripts/build_snapshots_v3.py            # re-download and rebuild the snapshots (optional)
python run_pipeline.py --from RAW_DATA_ACQUIRED  # v3 into outputs/ and results/
python scripts/run_track.py --name v3_edition_harmonised
python scripts/run_track.py --name v3b_official_only --arm official_only
python scripts/run_track.py --name v2r_reproduction --config-ref v2-frozen --raw-ref v2-frozen --from DATA_VALIDATED
python scripts/run_track.py --name v2rb_official_only --config-ref v2-frozen --raw-ref v2-frozen --arm official_only --from DATA_VALIDATED
python scripts/export_preprocessing_evidence.py  # tables, figures and quoted numbers
python -m pytest
```

### Licence and citation

Code: MIT (`LICENSE`). Derived data: CC BY 4.0 (`LICENSE-DATA`); third-party
values stay under their publishers' terms. Citation metadata: `CITATION.cff`.

This work is based upon work from COST Action HiTEc, CA21163, supported by COST
(European Cooperation in Science and Technology).

## Requirements

- Python **3.9+** (Active/Maintenance LTS recommended: 3.10–3.12)
- Dependencies in `requirements.txt` / `pyproject.toml`
- macOS: OpenMP runtime for XGBoost/LightGBM (`brew install libomp`)

## Setup

```bash
cd digital-maturity-framework
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# macOS if xgboost fails to load:
# brew install libomp
```

Put `src` on `PYTHONPATH` (the CLI does this automatically):

```bash
export PYTHONPATH=src
```

## Result versions (important)

| Version | Location | Missing data | Role |
|---|---|---|---|
| **v1 (preserved)** | [`versions/v1_official_unbalanced/`](versions/v1_official_unbalanced/) | Official gaps retained | Historical snapshot; predates the Aug-2026 corrections |
| **v1b (robustness arm)** | [`versions/v1b_official_corrected/`](versions/v1b_official_corrected/) | Official gaps retained | Corrected official-only track of v2 |
| **v2 (frozen)** | [`versions/v2_mice_imputed/`](versions/v2_mice_imputed/), git tag `v2-frozen` | MICE | Results of the doctoral project (August 2026) |
| **v2r / v2rb** | [`versions/v2r_reproduction/`](versions/v2r_reproduction/), [`versions/v2rb_official_only/`](versions/v2rb_official_only/) | MICE / official only | v2 configuration and data re-run with the current code |
| **v3 (primary, active `outputs/`)** | project root + [`versions/v3_edition_harmonised/`](versions/v3_edition_harmonised/) | MICE; observed values never overwritten | Edition-aware snapshots |
| **v3b** | [`versions/v3b_official_only/`](versions/v3b_official_only/) | Official only | Robustness arm of v3 |

v1b and v2 differ in exactly one respect — whether a missing cell is reconstructed — so
any divergence between them is attributable to the missing-data policy alone. Reproduce
the v1b arm without touching the active outputs:

```bash
python scripts/run_v1b_track.py
python scripts/build_results_discussion_v1b_docx.py
```

See [`versions/README.md`](versions/README.md).

## Run the pipeline

End-to-end orchestration (writes **v3 / MICE** results into `outputs/`):

```bash
python run_pipeline.py
```

Partial runs:

```bash
python run_pipeline.py --from INDEX_CREATED --to ML_COMPLETED
python run_pipeline.py --from VISUALIZATION_COMPLETED --skip-completed
```

Stage names match `PipelineStage` values (e.g. `RAW_DATA_ACQUIRED`, `DATA_VALIDATED`, `ML_COMPLETED`, `VISUALIZATION_COMPLETED`, `REPORTING_COMPLETED`, `FRAMEWORK_VALIDATED`).

### Stage modules (each exposes `run(project_root, config)`)

| Stage | Module |
|---|---|
| Acquire / validate | `ingestion.acquire`, `validation.validate` |
| Process | `preprocessing.pipeline`, `feature_engineering.pipeline` |
| Index (DMI) | `index.build` |
| Statistics | `statistics.descriptive` |
| Econometrics | `econometrics.models` |
| Machine learning | `machine_learning.train` |
| XAI | `explainability.explain` |
| Convergence | `convergence.analysis` |
| Visualization | `visualization.plots` |
| Reporting bundle | `imputation.robustness`, `hypotheses.testing`, `reporting.comparative`, `dashboard.export`, `framework.outputs`, `reporting.generate` |
| Audit | `audit.validate` |

Example single-stage call:

```bash
PYTHONPATH=src python -c "from visualization.plots import run; run('.')"
```

## Dashboard

Export pipeline outputs (no synthetic data):

```bash
PYTHONPATH=src python -c "from dashboard.export import run; run('.')"
```

Launch Streamlit from the **project root**:

```bash
streamlit run src/dashboard/app.py
```

Sections: overview, country groups, predictions, explanations, convergence. Data is read only from `outputs/dashboard/`.

## Architecture overview

```
config/          YAML research, countries, models, viz, dashboard
data/
  raw/           Integrated source panel
  interim/       Validated / cleaned
  processed/     Wide panel, features, DMI, analysis panel
src/
  pipeline/      Orchestrator, config, state
  common/        IO, errors, logging, seeds
  registries/    Result + model registries
  */             Domain stages (index, econometrics, ML, viz, …)
results/         Canonical analysis artifacts + registry
outputs/
  figures/       Matplotlib / Plotly figures
  predictions/   Holdout predictions
  models/        Joblib models + best_model.json
  dashboard/     Streamlit data package + framework_status.json
  reports/       Markdown summaries
  hypothesis_testing/
  audit/         Reproducibility, validation, final_audit
experiments/     Experiment tracker metadata
```

### Intelligent framework facade

`framework.IntelligentFramework` packages monitoring / prediction / explanation over registries:

- `latest_maturity()` — latest DMI by country/group
- `predict_path_info()` — best-model prediction artifacts
- `explanations_summary()` — top SHAP / importance factors
- `health_check()` — critical artifact readiness

`framework.outputs.run` writes `outputs/dashboard/framework_status.json`.

### Hypotheses

`hypotheses.testing` evaluates **H1** and **H1.1–H1.8** from econometric coefficients, group tests, ML vs baseline RMSE, and XAI overlap. Results: `results/hypotheses/hypothesis_evaluation.json`.

When the panel contains imputed cells, each verdict is additionally reported under multiple-imputation pooling, complete cases, and a leakage-safe predictive protocol (`robustness_summary` node), so specification-dependent conclusions are visible rather than hidden.

### Imputation robustness (v2 track)

`imputation.robustness` runs before hypothesis evaluation and writes `results/imputation/`:

| Artifact | Question it answers |
|---|---|
| `cell_provenance_by_variable.csv` | How much of each indicator is published, carried forward, or reconstructed? |
| `imputation_validation.csv` | How accurately does the imputer reconstruct hidden official cells versus naive fills? |
| `mi_pooled_fe.csv` | What are the fixed-effects estimates once imputation uncertainty enters the standard errors (Rubin's rules)? |
| `complete_case_fe.csv` | What survives if reconstructed rows are discarded entirely? |
| `leakage_safe_ml.csv` | Do the ML accuracy gains hold when the imputer never sees hold-out years? |

The stage writes `not_applicable` when imputation is disabled, so both research versions expose the same contract.

### Audit

`audit.validate.run` scores Definition-of-Complete criteria across data → framework layers into `outputs/audit/final_audit.json` (+ `.md`). Raises `FrameworkError` only when critical artifacts are missing (raw data, DMI/analysis panel, model comparison).

## Key outputs

- `results/machine_learning/model_comparison.csv`
- `results/econometrics/coefficients.csv`
- `results/hypotheses/hypothesis_evaluation.json`
- `results/comparative/group_comparison.csv`
- `results/imputation/mi_pooled_fe.csv`, `leakage_safe_ml.csv`, `cell_provenance_by_variable.csv`
- `outputs/figures/*.png` (+ Plotly `.html` when available)
- `outputs/reports/results_summary.md`, `discussion_evidence.md`, `tables_index.md`
- `outputs/reports/Results_and_Discussion_v2_MICE.docx` (rebuild: `python scripts/build_results_discussion_docx.py`)
- `outputs/reports/Results_and_Discussion_v1b_OfficialCorrected.docx` (rebuild: `python scripts/build_results_discussion_v1b_docx.py`)
- `outputs/reports/Methodology_Results_Discussion_v2.docx` — dissertation-style chapters 4–6 (Methodology / Results / Discussion) following the proposal structure (rebuild: `python scripts/build_methodology_results_discussion_docx.py`)
- `outputs/reports/Doktorski_proekt_ver02.docx` — mentor revision: processes, ontology, VDA (rebuild: `python scripts/build_doctoral_project_ver02_docx.py`)
- `ontology/dmi-framework.ttl` + `ontology/sigma.md`; populated ABox: `outputs/ontology/`
- `docs/processes/*.md` and `outputs/figures/processes/*.png`
- `outputs/powerbi/` star schema (see `POWERBI_README.md`)
- `outputs/audit/final_audit.json`

## Configuration

All YAML under `config/` (copied to `outputs/audit/config_snapshot/` on pipeline start). Research hypotheses and panel years live in `config/research.yaml`; country groups in `config/countries.yaml`.

## License / research use

Academic research framework for digital maturity monitoring and predictive analytics. Adjust indicators and country groups via config before re-running the pipeline.
