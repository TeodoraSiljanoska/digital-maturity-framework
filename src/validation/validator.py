"""Panel data validation for ingested long-format frames."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from common.errors import ValidationError
from data_sources.base import STANDARD_COLUMNS


class DataValidator:
    """Validate schema, countries, years, duplicates, and completeness."""

    def __init__(
        self,
        allowed_countries: Optional[Sequence[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        min_non_null_ratio: float = 0.0,
    ):
        self.allowed_countries = {
            str(c).upper() for c in (allowed_countries or []) if str(c).strip()
        }
        self.start_year = start_year
        self.end_year = end_year
        self.min_non_null_ratio = float(min_non_null_ratio)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "ok": True,
            "hard_failures": [],
            "warnings": [],
            "checks": {},
        }

        if df is None or not isinstance(df, pd.DataFrame):
            raise ValidationError("Validation requires a pandas DataFrame")

        if df.empty:
            raise ValidationError(
                "Empty dataset",
                details={"rows": 0},
            )

        missing_cols = [c for c in STANDARD_COLUMNS if c not in df.columns]
        report["checks"]["schema"] = {
            "required": list(STANDARD_COLUMNS),
            "missing": missing_cols,
        }
        if missing_cols:
            report["warnings"].append(f"Missing columns: {missing_cols}")

        work = df.copy()
        if "country_iso3" in work.columns:
            work["country_iso3"] = work["country_iso3"].astype(str).str.upper()

        # Countries allowlist — hard failure for unknown countries
        if self.allowed_countries and "country_iso3" in work.columns:
            present = set(work["country_iso3"].dropna().astype(str).tolist())
            unknown = sorted(present - self.allowed_countries)
            report["checks"]["countries"] = {
                "allowed": sorted(self.allowed_countries),
                "present": sorted(present),
                "unknown": unknown,
            }
            if unknown:
                raise ValidationError(
                    "Unknown countries outside allowlist",
                    details={"unknown": unknown},
                )
        else:
            report["checks"]["countries"] = {"skipped": True}

        # Year range — soft warning if out of range
        if "year" in work.columns:
            years = pd.to_numeric(work["year"], errors="coerce")
            ymin = int(years.min()) if years.notna().any() else None
            ymax = int(years.max()) if years.notna().any() else None
            oob = 0
            if self.start_year is not None and self.end_year is not None:
                oob = int((~years.between(self.start_year, self.end_year)).sum())
                if oob:
                    report["warnings"].append(
                        f"{oob} rows outside year range "
                        f"[{self.start_year}, {self.end_year}]"
                    )
            report["checks"]["years"] = {
                "min": ymin,
                "max": ymax,
                "out_of_range_rows": oob,
                "expected_start": self.start_year,
                "expected_end": self.end_year,
            }
        else:
            report["checks"]["years"] = {"skipped": True}

        # Duplicate keys
        key_cols = [c for c in ("country_iso3", "year", "indicator_code") if c in work.columns]
        dup_count = 0
        if len(key_cols) == 3:
            dup_count = int(work.duplicated(subset=key_cols, keep=False).sum())
            if dup_count:
                report["warnings"].append(f"Duplicate key rows: {dup_count}")
        report["checks"]["duplicates"] = {
            "key_columns": key_cols,
            "duplicate_rows": dup_count,
        }

        # Non-null ratio
        if "value" in work.columns:
            non_null_ratio = float(work["value"].notna().mean())
        else:
            non_null_ratio = 0.0
            report["warnings"].append("Column 'value' missing; non_null_ratio=0")
        report["checks"]["non_null_ratio"] = {
            "ratio": non_null_ratio,
            "min_required": self.min_non_null_ratio,
        }
        if non_null_ratio < self.min_non_null_ratio:
            report["warnings"].append(
                f"Non-null ratio {non_null_ratio:.3f} below minimum "
                f"{self.min_non_null_ratio:.3f}"
            )

        report["rows"] = int(len(work))
        report["ok"] = len(report["hard_failures"]) == 0
        return report


def validate_panel(
    df: pd.DataFrame,
    allowed_countries: Optional[Iterable[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    min_non_null_ratio: float = 0.0,
) -> Dict[str, Any]:
    validator = DataValidator(
        allowed_countries=list(allowed_countries or []),
        start_year=start_year,
        end_year=end_year,
        min_non_null_ratio=min_non_null_ratio,
    )
    return validator.validate(df)
