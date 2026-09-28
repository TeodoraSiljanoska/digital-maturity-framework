# Run this framework on Windows

This zip is the full project **without** a virtualenv. macOS `.venv` will not work on Windows. Frozen v2 results (11 Aug 2026) are included — you do **not** need to re-run the pipeline to open the dashboard or read the doctoral DOCX.

## 1. Install Python

- Python **3.10, 3.11, or 3.12** from https://www.python.org/downloads/windows/
- In the installer, tick **Add python.exe to PATH**
- Confirm in Command Prompt: `python --version`

Do not use an odd-numbered Python (3.13 is usually fine; 3.9 also works).

## 2. Unzip and open a terminal in the project folder

Unzip so you have a folder named `digital-maturity-framework` containing `src\`, `config\`, `outputs\`, `requirements.txt`.

```bat
cd path\to\digital-maturity-framework
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install python-docx openpyxl
```

`python-docx` is required only if you rebuild the Word report. `openpyxl` is used for Excel exports.

If `xgboost` or `lightgbm` fail to import, install the latest matching wheels from PyPI (they ship Windows binaries). You do **not** need Java. HermiT is not used (`reasoner: none`).

## 3. Run the dashboard (main live app)

From the **project root**, with `.venv` active:

```bat
set PYTHONPATH=src
streamlit run src\dashboard\app.py
```

Or double-click `setup_windows.bat` once, then `run_dashboard.bat`.

The browser opens a local Streamlit app. Data is read from `outputs\dashboard\` (already in the zip).

Power BI Desktop (Windows): load `outputs\powerbi\` (star schema CSV + `LoadStarSchema.pq`). A `.pbix` is not generated on macOS; you can build one on this laptop.

## 4. Rebuild the doctoral Word file (optional)

```bat
set PYTHONPATH=src
python scripts\build_doctoral_project_ver02_docx.py
```

Output: `outputs\reports\Doktorski_proekt_ver02.docx`

## 5. Tests (optional)

```bat
set PYTHONPATH=src
python -m pytest tests -q
```

## Do not do this unless you intend a new estimate

```bat
python run_pipeline.py
```

That re-acquires data and re-fits models. It overwrites the frozen v2 arm. The dissertation numbers are already in `results\`, `outputs\`, and `data\processed\`.

## Layout (what you need)

| Path | Role |
|---|---|
| `src\` | Code |
| `config\` | YAML (indicators, countries, index) |
| `data\processed\` | Frozen panel / DMI |
| `results\` | FE, ML, SHAP, hypotheses |
| `outputs\dashboard\` | Streamlit data |
| `outputs\figures\` | Process diagrams + plots |
| `outputs\ontology\` | Populated TTL + SPARQL answers |
| `outputs\powerbi\` | Star schema for Power BI |
| `outputs\reports\` | ver02 DOCX |
| `ontology\` | TBox |
| `versions\` | Frozen v1 / v1b / v2 snapshots |
