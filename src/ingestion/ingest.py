"""Orchestrate multi-source indicator ingestion into a long-format panel."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from catalog.data_catalog import DataCatalog
from common.errors import DataSourceError, IndicatorUnavailableError
from common.io import ensure_dir, file_checksum, write_df, write_json
from common.logging_utils import get_logger
from data_sources.base import STANDARD_COLUMNS, ensure_standard_columns
from data_sources.registry import get_adapter
from pipeline.pipeline_config import PipelineConfig
from validation.validator import DataValidator


class DataIngestion:
    """Fetch all catalog indicators via adapters and persist raw panels."""

    def __init__(
        self,
        config: PipelineConfig,
        catalog: DataCatalog,
        logger: Optional[Any] = None,
    ):
        self.config = config
        self.catalog = catalog
        self.logger = logger or get_logger("dmf.ingestion")
        self.project_root = Path(config.project_root)
        self.raw_dir = self.project_root / "data" / "raw"

    def run(self, validate: bool = True) -> pd.DataFrame:
        start_year, end_year = self.config.year_range()
        countries = self.config.iso3_list()
        source_cfgs = (self.config.sources or {}).get("sources", {}) or {}

        skip_sources = {"constructed", "derived"}
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ind in self.catalog.list():
            source = str(ind.get("source", "")).strip()
            retrieval = str(ind.get("retrieval_method") or "").strip().lower()
            if source in skip_sources or retrieval == "constructed":
                continue
            if not source:
                raise DataSourceError(
                    f"Indicator missing source: {ind.get('indicator_id')}",
                    details={"indicator": ind},
                )
            by_source[source].append(ind)

        frames: List[pd.DataFrame] = []
        source_meta: Dict[str, Any] = {
            "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "start_year": start_year,
            "end_year": end_year,
            "countries": countries,
            "sources": {},
        }

        for source_name, indicators in sorted(by_source.items()):
            cfg = dict(source_cfgs.get(source_name, {}) or {})
            adapter = get_adapter(source_name, config=cfg, project_root=self.project_root)
            code_to_id = {
                str(i.get("indicator_code")): str(i.get("indicator_id"))
                for i in indicators
                if i.get("indicator_code")
            }
            codes = list(code_to_id.keys())
            self.logger.info(
                "Ingesting source=%s indicators=%s", source_name, ",".join(codes)
            )
            try:
                df = adapter.get_data(
                    indicator_codes=codes,
                    countries=countries,
                    start_year=start_year,
                    end_year=end_year,
                    indicator_id_map=code_to_id,
                )
            except IndicatorUnavailableError as exc:
                # Optional sources may be unavailable; hard sources must not fabricate.
                self.logger.warning(
                    "Source %s unavailable for requested indicators: %s",
                    source_name,
                    exc.message,
                )
                raise DataSourceError(
                    f"Total failure for source '{source_name}': {exc.message}",
                    details=exc.details,
                ) from exc
            except DataSourceError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise DataSourceError(
                    f"Total failure for source '{source_name}'",
                    details={"cause": repr(exc)},
                ) from exc

            if df is None or df.empty:
                raise DataSourceError(
                    f"Total failure for source '{source_name}': empty result",
                    details={"codes": codes},
                )

            df = ensure_standard_columns(df)
            df["source"] = source_name
            # Prefer catalog indicator_id mapping
            if "indicator_code" in df.columns:
                df["indicator_id"] = df["indicator_code"].map(code_to_id).fillna(
                    df.get("indicator_id")
                )

            source_dir = ensure_dir(self.raw_dir / source_name.replace("/", "_"))
            source_path = write_df(df, source_dir / "raw_long.parquet")
            write_df(df, source_dir / "raw_long.csv")
            checksum = file_checksum(source_path)
            try:
                vreport = adapter.validate(df)
            except Exception as exc:  # noqa: BLE001
                vreport = {"ok": False, "error": repr(exc)}

            source_meta["sources"][source_name] = {
                "adapter": adapter.name,
                "rows": int(len(df)),
                "indicator_codes": codes,
                "path": str(source_path.relative_to(self.project_root)),
                "checksum_sha256": checksum,
                "retrieved_at": source_meta["retrieved_at"],
                "adapter_metadata": adapter.get_metadata(),
                "validation": vreport,
            }
            frames.append(df)

        if not frames:
            raise DataSourceError("Ingestion produced no data from any source")

        combined = ensure_standard_columns(pd.concat(frames, ignore_index=True))
        combined["country_iso3"] = combined["country_iso3"].astype(str).str.upper()
        combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
        combined = combined.dropna(subset=["year"]).copy()
        combined["year"] = combined["year"].astype(int)
        combined = combined.sort_values(
            ["country_iso3", "year", "indicator_id"]
        ).reset_index(drop=True)

        if validate:
            validator = DataValidator(
                allowed_countries=countries,
                start_year=start_year,
                end_year=end_year,
            )
            source_meta["combined_validation"] = validator.validate(combined)

        integrated_path = write_df(combined, self.raw_dir / "integrated_raw.parquet")
        write_df(combined, self.raw_dir / "integrated_raw.csv")
        source_meta["integrated"] = {
            "path": str(integrated_path.relative_to(self.project_root)),
            "checksum_sha256": file_checksum(integrated_path),
            "rows": int(len(combined)),
            "columns": list(STANDARD_COLUMNS),
        }

        meta_path = self.raw_dir / "source_metadata.json"
        write_json(meta_path, source_meta)
        self.logger.info(
            "Ingestion complete: rows=%s path=%s",
            len(combined),
            integrated_path,
        )
        return combined
