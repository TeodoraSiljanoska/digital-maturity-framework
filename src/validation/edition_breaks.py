"""Detect level shifts at the boundaries between editions of a published index.

A composite index that is republished under a revised framework or scale moves
every country at once. Within one framework, consecutive editions move
countries by small amounts in both directions. The check contrasts the two: a
boundary is flagged when its median per-year change is several times the
typical change at the indicator's other boundaries and nearly all countries
move the same way. Framework changes declared by the publisher
(``config/editions.yaml``) are reported alongside, because a series with only
two editions offers no second boundary to compare against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.io import ensure_dir, write_df, write_json

DEFAULT_RATIO_THRESHOLD = 3.0
DEFAULT_SIGN_SHARE_THRESHOLD = 0.75
DEFAULT_MIN_COUNTRIES = 6

# Indicators published as editions of composite indices rather than annual statistics.
DEFAULT_EDITION_INDICATORS = ["X3", "X5", "X6", "X7", "X10", "C2"]


def _edition_years(obs: pd.DataFrame, var: str, min_countries: int) -> List[int]:
    """Years in which enough countries carry a published value to count as an edition."""
    counts = obs.groupby("year")[var].size()
    return sorted(int(y) for y, n in counts.items() if n >= min_countries)


def _per_year_changes(obs: pd.DataFrame, var: str, first: int, second: int) -> pd.Series:
    before = obs.loc[obs["year"] == first].set_index("country_iso3")[var]
    after = obs.loc[obs["year"] == second].set_index("country_iso3")[var]
    common = before.index.intersection(after.index)
    return (after[common] - before[common]) / float(second - first)


def _declared_breaks(editions_cfg: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    declared: Dict[str, List[Dict[str, Any]]] = {}
    for var, spec in ((editions_cfg or {}).get("indicators") or {}).items():
        for item in (spec or {}).get("declared_breaks") or []:
            between = item.get("between") or []
            if len(between) == 2:
                declared.setdefault(str(var), []).append(
                    {"between": (int(between[0]), int(between[1])), "note": item.get("note")}
                )
    return declared


def _declared_for(
    declared: Dict[str, List[Dict[str, Any]]], var: str, boundary: Tuple[int, int]
) -> Optional[str]:
    """Return the publisher note when a declared change falls inside the boundary."""
    first, second = boundary
    for item in declared.get(var, []):
        lo, hi = item["between"]
        if first <= lo and hi <= second:
            return str(item.get("note") or "declared by publisher")
    return None


def detect_edition_breaks(
    wide: pd.DataFrame,
    variables: Sequence[str],
    *,
    editions_cfg: Optional[Dict[str, Any]] = None,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    sign_share_threshold: float = DEFAULT_SIGN_SHARE_THRESHOLD,
    min_countries: int = DEFAULT_MIN_COUNTRIES,
) -> pd.DataFrame:
    """
    Return one row per (variable, boundary) between consecutive edition years.

    ``wide`` must hold published values only: carried-forward or reconstructed
    cells would manufacture editions that the publisher never released.
    Changes are expressed per year so that biennial surveys and annual
    editions are compared on the same footing.
    """
    declared = _declared_breaks(editions_cfg)
    rows: List[Dict[str, Any]] = []
    for var in variables:
        if var not in wide.columns:
            continue
        obs = wide.loc[wide[var].notna(), ["country_iso3", "year", var]]
        years = _edition_years(obs, var, min_countries)
        boundaries = list(zip(years[:-1], years[1:]))
        changes = {b: _per_year_changes(obs, var, *b) for b in boundaries}
        for boundary, delta in changes.items():
            others = [c.abs() for key, c in changes.items() if key != boundary and len(c)]
            reference = float(pd.concat(others).median()) if others else np.nan
            median_change = float(delta.median()) if len(delta) else np.nan
            if np.isfinite(reference) and reference > 0 and np.isfinite(median_change):
                ratio = abs(median_change) / reference
            else:
                ratio = np.nan
            if len(delta):
                share_same_sign = float(max((delta > 0).mean(), (delta < 0).mean()))
            else:
                share_same_sign = np.nan
            detected = bool(
                np.isfinite(ratio)
                and ratio >= ratio_threshold
                and share_same_sign >= sign_share_threshold
                and len(delta) >= min_countries
            )
            note = _declared_for(declared, var, boundary)
            if note and detected:
                status = "declared_and_detected"
            elif note:
                status = "declared"
            elif detected:
                status = "detected"
            elif not np.isfinite(ratio):
                status = "untestable"
            else:
                status = "none"
            rows.append(
                {
                    "variable": var,
                    "from_year": int(boundary[0]),
                    "to_year": int(boundary[1]),
                    "n_countries": int(len(delta)),
                    "median_change_per_year": median_change,
                    "reference_abs_change_per_year": reference,
                    "ratio": ratio,
                    "share_same_sign": share_same_sign,
                    "statistical_flag": detected,
                    "declared_note": note,
                    "status": status,
                }
            )
    columns = [
        "variable",
        "from_year",
        "to_year",
        "n_countries",
        "median_change_per_year",
        "reference_abs_change_per_year",
        "ratio",
        "share_same_sign",
        "statistical_flag",
        "declared_note",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)


def check_variables(editions_cfg: Optional[Dict[str, Any]], fallback: Sequence[str]) -> List[str]:
    """Indicators covered by the editions config, or ``fallback`` when none is given."""
    configured = list(((editions_cfg or {}).get("indicators") or {}).keys())
    return [str(v) for v in configured] or [str(v) for v in fallback]


def write_break_report(
    project_root: Path | str,
    breaks: pd.DataFrame,
    *,
    label: str,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist ``outputs/audit/edition_breaks_<label>.{csv,json}`` and return the summary."""
    audit_dir = ensure_dir(Path(project_root) / "outputs" / "audit")
    csv_path = write_df(breaks, audit_dir / f"edition_breaks_{label}.csv")
    flagged = breaks.loc[breaks["status"].isin(["detected", "declared_and_detected", "declared"])]
    summary = {
        "label": label,
        "thresholds": thresholds
        or {
            "ratio": DEFAULT_RATIO_THRESHOLD,
            "share_same_sign": DEFAULT_SIGN_SHARE_THRESHOLD,
            "min_countries": DEFAULT_MIN_COUNTRIES,
        },
        "n_boundaries": int(len(breaks)),
        "n_flagged": int(len(flagged)),
        "flagged": json.loads(flagged.to_json(orient="records")),
        "csv": str(csv_path.relative_to(Path(project_root))),
    }
    write_json(audit_dir / f"edition_breaks_{label}.json", summary)
    return summary
