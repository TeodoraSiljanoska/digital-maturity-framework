# Digital Maturity Framework

Intelligent AI framework for **visual analytics, monitoring, and prediction** of digital transformation (Digital Maturity Index — DMI).

The project builds a multi-country panel (2012–2025), constructs a composite DMI, runs descriptive / econometric / ML analyses, evaluates research hypotheses, and exposes results through figures, reports, and a Streamlit dashboard.

## Requirements

- Python **3.9+** (Active/Maintenance LTS recommended: 3.10–3.12)
- Dependencies in `requirements.txt` / `pyproject.toml`
- macOS: OpenMP runtime for XGBoost/LightGBM (`brew install libomp`)

## Setup

```bash
cd ~/Desktop/digital-maturity-framework
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
| **v1b (robustness arm)** | [`versions/v1b_official_corrected/`](versions/v1b_official_corrected/) | Official gaps retained | Corrected official-only track; shows what survives with nothing reconstructed |
| **v2 (inferential arm, active `outputs/`)** | project root + [`versions/v2_mice_imputed/`](versions/v2_mice_imputed/) | **MICE** (IterativeImputer + RandomForest) fills empties; observed official values unchanged | Primary results |

v1b and v2 differ in exactly one respect — whether a missing cell is reconstructed — so
any divergence between them is attributable to the missing-data policy alone. Reproduce
the v1b arm without touching the active outputs:

```bash
python scripts/run_v1b_track.py
python scripts/build_results_discussion_v1b_docx.py
```

See [`versions/README.md`](versions/README.md).

## Run the pipeline

End-to-end orchestration (writes **v2 / MICE** results into `outputs/`):

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
