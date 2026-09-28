"""Export a Power BI star schema from pipeline artifacts. No fabricated values."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from catalog.data_catalog import DataCatalog
from common.io import ensure_dir, read_df, write_df, write_json
from common.logging_utils import get_logger

logger = get_logger("dmf.powerbi")

ACQUIRED = [f"X{i}" for i in range(1, 11)] + ["C1", "C2", "DMI"]


def _countries(root: Path) -> pd.DataFrame:
    with (root / "config" / "countries.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rows = []
    for gid, group in (cfg.get("groups") or {}).items():
        for c in group.get("countries") or []:
            rows.append(
                {
                    "country_iso3": c.get("iso3"),
                    "country_iso2": c.get("iso2"),
                    "country_name": c.get("name"),
                    "group_id": gid,
                    "group_name": group.get("name"),
                }
            )
    return pd.DataFrame(rows)


def run(project_root: Path | str, config: Any = None) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    out = ensure_dir(root / "outputs" / "powerbi")
    catalog = DataCatalog.from_project(root)
    dim_indicator = catalog.reusable_frame()
    dim_country = _countries(root)
    dim_phase = pd.DataFrame(
        [
            {"phase_id": "acquire", "phase_name": "Acquire", "sort": 1},
            {"phase_id": "validate", "phase_name": "Validate", "sort": 2},
            {"phase_id": "process", "phase_name": "Process", "sort": 3},
            {"phase_id": "index", "phase_name": "Index", "sort": 4},
            {"phase_id": "analyse", "phase_name": "Analyse", "sort": 5},
            {"phase_id": "predict", "phase_name": "Predict", "sort": 6},
            {"phase_id": "explain", "phase_name": "Explain", "sort": 7},
            {"phase_id": "visualize", "phase_name": "Visualize", "sort": 8},
        ]
    )
    dim_source = pd.DataFrame(
        [
            {"source_id": r.get("source"), "collection_method": r.get("collection_method")}
            for r in catalog.list()
        ]
    ).drop_duplicates()

    score_path = root / "outputs" / "ontology" / "score_entries.parquet"
    fact: Optional[pd.DataFrame] = None
    if score_path.exists():
        fact = read_df(score_path)
    else:
        panel_path = root / "data" / "processed" / "analysis_panel.parquet"
        if not panel_path.exists():
            panel_path = root / "data" / "processed" / "dmi_panel.parquet"
        if panel_path.exists():
            panel = read_df(panel_path)
            cols = [c for c in ACQUIRED if c in panel.columns]
            idv = [c for c in ("country_iso3", "year", "group_id") if c in panel.columns]
            fact = panel.melt(id_vars=idv, value_vars=cols, var_name="indicator_id", value_name="value")
            fact["provenance"] = fact["indicator_id"].map(
                lambda i: "constructed" if i == "DMI" else "official"
            )
            fact["phase_id"] = fact["indicator_id"].map(lambda i: "index" if i == "DMI" else "acquire")
            src_map = {r["indicator_id"]: r.get("source") for r in catalog.list()}
            fact["source_id"] = fact["indicator_id"].map(src_map)

    write_df(dim_indicator, out / "dim_indicator.csv")
    write_df(dim_country, out / "dim_country.csv")
    write_df(dim_phase, out / "dim_phase.csv")
    write_df(dim_source, out / "dim_source.csv")
    n_fact = 0
    if fact is not None and not fact.empty:
        write_df(fact, out / "fact_indicator_value.csv")
        n_fact = int(len(fact))

    recipe = """# Power BI Desktop recipe (Windows)

Streamlit (`streamlit run src/dashboard/app.py`) is the VDA that runs on this
machine. Power BI Desktop is a Windows application; this folder does not contain
a `.pbix` because it cannot be generated or opened on macOS.

Related practice in HiTEc CA21163 (repository, Objective B1): Siljanoska Taskovska,
Savoska and Jolevski, *Integrating Python into Power BI for analyzing and predicting
digital development: Case study – Balkan countries*
(https://www.hitecaction.org/repository.php).

## Load the star schema

1. Open Power BI Desktop (Windows).
2. Home → Transform data → New parameter `FolderPath` (text) = this folder.
3. Home → Get data → Blank query → Advanced Editor → paste `LoadStarSchema.pq`.
4. Close and apply.

Relationships:
- fact_indicator_value[indicator_id] → dim_indicator[indicator_id]
- fact_indicator_value[country_iso3] → dim_country[country_iso3]
- fact_indicator_value[phase_id] → dim_phase[phase_id]
- fact_indicator_value[source_id] → dim_source[source_id]

Slicers: year (slider), country_name, group_name, indicator_id, provenance, phase_name.
Visuals: line chart of DMI by year; matrix indicator × country; donut of provenance.
Do not type values by hand. Refresh from this folder after each pipeline run.
"""
    (out / "POWERBI_README.md").write_text(recipe, encoding="utf-8")
    pq = '''let
    FolderPath = "REPLACE_WITH_ABSOLUTE_PATH_TO_outputs/powerbi",
    Fact = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/fact_indicator_value.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimIndicator = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_indicator.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimCountry = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_country.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimPhase = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_phase.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimSource = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_source.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]))
in
    Fact
'''
    (out / "LoadStarSchema.pq").write_text(pq, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_fact_rows": n_fact,
        "files": [
            "dim_indicator.csv",
            "dim_country.csv",
            "dim_phase.csv",
            "dim_source.csv",
            "fact_indicator_value.csv",
            "POWERBI_README.md",
            "LoadStarSchema.pq",
        ],
    }
    write_json(out / "manifest.json", manifest)
    logger.info("Power BI star schema: %s fact rows", n_fact)
    return manifest
