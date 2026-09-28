"""Pipeline entrypoint for raw data acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from catalog.data_catalog import DataCatalog
from ingestion.ingest import DataIngestion
from pipeline.pipeline_config import PipelineConfig


def run(project_root: Path | str, config: PipelineConfig) -> Dict[str, Any]:
    """Acquire multi-source indicators into ``data/raw/integrated_raw.parquet``."""
    project_root = Path(project_root)
    catalog = DataCatalog.from_project(project_root)
    df = DataIngestion(config, catalog).run()
    integrated = project_root / "data" / "raw" / "integrated_raw.parquet"
    return {
        "path": str(integrated),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_indicators": int(df["indicator_id"].nunique()) if "indicator_id" in df.columns else 0,
        "n_countries": int(df["country_iso3"].nunique()) if "country_iso3" in df.columns else 0,
    }
