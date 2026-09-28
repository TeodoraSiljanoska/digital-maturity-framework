"""Pipeline entrypoint for raw panel validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from common.io import ensure_dir, read_df, write_df, write_json
from pipeline.pipeline_config import PipelineConfig
from validation.validator import DataValidator


def run(project_root: Path | str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Validate ``data/raw/integrated_raw.parquet``.

    Writes:
      - ``outputs/audit/validation_report.json``
      - ``data/interim/validated_raw.parquet``
    """
    project_root = Path(project_root)
    raw_path = project_root / "data" / "raw" / "integrated_raw.parquet"
    df = read_df(raw_path)

    start_year, end_year = config.year_range()
    mv = (config.preprocessing or {}).get("missing_values", {}) or {}
    min_ratio = float(mv.get("min_non_null_ratio", 0.0))

    validator = DataValidator(
        allowed_countries=config.iso3_list(),
        start_year=start_year,
        end_year=end_year,
        min_non_null_ratio=min_ratio,
    )
    report = validator.validate(df)

    report_path = project_root / "outputs" / "audit" / "validation_report.json"
    write_json(report_path, report)

    interim_dir = ensure_dir(project_root / "data" / "interim")
    validated_path = write_df(df, interim_dir / "validated_raw.parquet")

    return {
        "report_path": str(report_path),
        "validated_path": str(validated_path),
        "ok": bool(report.get("ok", False)),
        "rows": int(len(df)),
        "warnings": list(report.get("warnings") or []),
    }
