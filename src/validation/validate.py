"""Pipeline entrypoint for raw panel validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from common.io import ensure_dir, read_df, write_df, write_json
from pipeline.pipeline_config import PipelineConfig
from validation.edition_breaks import (
    DEFAULT_EDITION_INDICATORS,
    check_variables,
    detect_edition_breaks,
    write_break_report,
)
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

    # Edition-boundary diagnostic on the values exactly as published (before any
    # rescaling or edition filter); non-blocking, it documents what the
    # preprocessing stage has to reconcile.
    published = (
        df.assign(value=pd.to_numeric(df["value"], errors="coerce"))
        .pivot_table(index=["country_iso3", "year"], columns="indicator_id", values="value", aggfunc="last")
        .reset_index()
    )
    editions_cfg = getattr(config, "editions", None) or {}
    breaks = detect_edition_breaks(
        published,
        check_variables(editions_cfg, DEFAULT_EDITION_INDICATORS),
        editions_cfg=editions_cfg,
    )
    break_summary = write_break_report(project_root, breaks, label="raw")

    return {
        "report_path": str(report_path),
        "validated_path": str(validated_path),
        "ok": bool(report.get("ok", False)),
        "rows": int(len(df)),
        "warnings": list(report.get("warnings") or []),
        "edition_breaks_raw": {
            "n_boundaries": break_summary["n_boundaries"],
            "n_flagged": break_summary["n_flagged"],
        },
    }
