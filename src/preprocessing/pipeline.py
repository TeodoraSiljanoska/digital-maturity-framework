"""Preprocess long-format raw data into a clean wide country-year panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from common.io import ensure_dir, read_df, write_df
from common.logging_utils import get_logger
from lineage.tracker import LineageTracker
from pipeline.pipeline_config import PipelineConfig

INDICATOR_COLS: List[str] = [
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "X8",
    "X9",
    "X10",
    "C1",
    "C2",
]

logger = get_logger("dmf.preprocessing")


def _pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["country_iso3"] = work["country_iso3"].astype(str).str.upper()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year", "indicator_id"]).copy()
    work["year"] = work["year"].astype(int)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")

    # Collapse duplicate long keys before pivot
    dup_cfg_subset = ["country_iso3", "year", "indicator_id"]
    work = work.sort_values(dup_cfg_subset)
    work = work.drop_duplicates(subset=dup_cfg_subset, keep="last")

    wide = work.pivot_table(
        index=["country_iso3", "year"],
        columns="indicator_id",
        values="value",
        aggfunc="last",
    )
    wide = wide.reset_index()
    wide.columns.name = None

    for col in INDICATOR_COLS:
        if col not in wide.columns:
            wide[col] = np.nan

    return wide


def _complete_panel(
    wide: pd.DataFrame,
    countries: Sequence[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if not countries:
        return wide.sort_values(["country_iso3", "year"]).reset_index(drop=True)
    idx = pd.MultiIndex.from_product(
        [list(countries), list(range(start_year, end_year + 1))],
        names=["country_iso3", "year"],
    )
    out = (
        wide.set_index(["country_iso3", "year"])
        .reindex(idx)
        .reset_index()
    )
    return out


def _merge_country_group(wide: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    table = config.country_table(as_dataframe=True)
    if table is None or table.empty:
        wide = wide.copy()
        wide["group_id"] = pd.NA
        wide["group_name"] = pd.NA
        return wide

    meta = table.rename(columns={"iso3": "country_iso3"})[
        ["country_iso3", "group_id", "group_name"]
    ].drop_duplicates(subset=["country_iso3"])
    meta["country_iso3"] = meta["country_iso3"].astype(str).str.upper()
    out = wide.merge(meta, on="country_iso3", how="left")
    return out


def _harmonize_scales(
    df: pd.DataFrame, rules: Sequence[Dict[str, Any]]
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Rescale editions of an index that were published on a different scale.

    Composite indices occasionally change their reporting scale between
    editions (e.g. Government AI Readiness moved from 0–10 to 0–100). Splicing
    such editions without rescaling injects a step change that fixed-effects
    and tree models would read as genuine within-country movement.
    """
    out = df.copy()
    applied: List[Dict[str, Any]] = []
    for rule in rules or []:
        var = rule.get("variable")
        if not var or var not in out.columns:
            continue
        factor = float(rule.get("multiply_by", 1.0))
        before_year = rule.get("apply_to_years_before")
        selector = out[var].notna()
        if before_year is not None:
            selector &= out["year"].astype(int) < int(before_year)
        n = int(selector.sum())
        if n == 0:
            continue
        out.loc[selector, var] = out.loc[selector, var] * factor
        applied.append(
            {
                "variable": var,
                "multiply_by": factor,
                "apply_to_years_before": before_year,
                "n_values_rescaled": n,
                "reason": rule.get("reason"),
            }
        )
    return out, applied


def _gap_limited_fill_series(s: pd.Series, max_gap: int) -> pd.Series:
    """Forward/backward fill only across gaps of at most ``max_gap`` years."""
    if max_gap <= 0:
        return s
    return s.ffill(limit=max_gap).bfill(limit=max_gap)


def _impute_missing(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    max_gap: int,
) -> pd.DataFrame:
    """
    Carry each country's own observations across gaps of at most ``max_gap`` years.

    Applied once per country series: an earlier implementation ran the same
    fill twice (grouped by ``group_id`` then by country), which silently
    doubled the effective reach of ``max_gap_years`` and let a single biennial
    observation cover up to five years.

    Values are never invented across gaps larger than ``max_gap`` years and
    never borrowed from another country.
    """
    sort_keys = [k for k in ("country_iso3", "year") if k in df.columns]
    out = df.sort_values(sort_keys).reset_index(drop=True).copy()
    cols = [c for c in value_cols if c in out.columns]
    if not cols:
        return out

    for col in cols:
        out[col] = out.groupby("country_iso3", group_keys=False)[col].transform(
            lambda s: _gap_limited_fill_series(s, max_gap)
        )

    return out



def _apply_log_transforms(df: pd.DataFrame, log_controls: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in log_controls:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        # Only log strictly positive values; leave others as NaN
        logged = np.where(values > 0, np.log(values), np.nan)
        out[col] = logged
    return out


def run(project_root: Path | str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Build a clean wide panel from integrated raw long data.

    Saves:
      - ``data/interim/panel_clean.parquet``
      - ``data/processed/panel_wide.parquet``
    """
    project_root = Path(project_root)
    raw_path = project_root / "data" / "raw" / "integrated_raw.parquet"
    logger.info("Loading raw panel from %s", raw_path)
    raw = read_df(raw_path)

    pre_cfg = config.preprocessing or {}
    mv_cfg = pre_cfg.get("missing_values", {}) or {}
    max_gap = int(mv_cfg.get("max_gap_years", 2))
    transforms = pre_cfg.get("transforms", {}) or {}
    log_controls = list(transforms.get("log_controls") or [])

    wide = _pivot_long_to_wide(raw)
    start_year, end_year = config.year_range()
    countries = config.iso3_list()
    wide = _complete_panel(wide, countries, start_year, end_year)
    wide = _merge_country_group(wide, config)

    harmonization_cfg = pre_cfg.get("scale_harmonization") or {}
    harmonized: List[Dict[str, Any]] = []
    if bool(harmonization_cfg.get("enabled", False)):
        wide, harmonized = _harmonize_scales(wide, harmonization_cfg.get("rules") or [])
        for entry in harmonized:
            logger.info(
                "Scale harmonisation: %s x%s for years before %s (%s values)",
                entry["variable"],
                entry["multiply_by"],
                entry["apply_to_years_before"],
                entry["n_values_rescaled"],
            )

    # Cell provenance: distinguish published observations from carried-forward
    # values and (later) reconstructed values, so authenticity claims can be
    # quantified rather than asserted.
    official_mask = wide[INDICATOR_COLS].notna()
    wide = _impute_missing(wide, INDICATOR_COLS, max_gap=max_gap)
    carried_mask = wide[INDICATOR_COLS].notna() & ~official_mask

    # Multivariate MICE (v2): fill remaining empties; preserve observed official cells
    mice_report: Dict[str, Any] = {"enabled": False}
    imp_cfg = getattr(config, "imputation", None) or {}
    mv_multi = mv_cfg.get("multivariate") or {}
    mice_enabled = bool(imp_cfg.get("enabled", False) or mv_multi.get("enabled", False))
    out_cfg = imp_cfg.get("output") or {}

    # The pre-imputation panel is the authenticity baseline: every robustness
    # check (complete-case FE, masked-cell benchmark, multiple imputation)
    # re-derives its own completion from this frame rather than from a panel
    # that already contains reconstructed cells.
    preimp_path = write_df(
        wide,
        project_root
        / out_cfg.get(
            "preimputation_path", "data/processed/panel_wide_preimputation.parquet"
        ),
    )

    if mice_enabled:
        from imputation.mice import apply_mice_imputation, imputation_kwargs_from_config

        wide, mask_df, mice_report = apply_mice_imputation(
            wide, INDICATOR_COLS, **imputation_kwargs_from_config(config)
        )
        mice_report["enabled"] = True
        mice_report["preimputation_panel"] = str(preimp_path.relative_to(project_root))
        write_df(
            mask_df,
            project_root
            / out_cfg.get("mask_path", "data/processed/imputation_mask.parquet"),
        )
        write_df(
            wide,
            project_root
            / out_cfg.get(
                "panel_path", "data/processed/panel_wide_imputed.parquet"
            ),
        )
        from common.io import write_json

        write_json(
            project_root
            / out_cfg.get("report_path", "outputs/audit/imputation_report.json"),
            mice_report,
        )
        logger.info(
            "MICE imputation applied: cells_filled=%s",
            mice_report.get("total_cells_imputed"),
        )

    provenance = pd.DataFrame(
        np.where(
            official_mask.to_numpy(),
            "official",
            np.where(carried_mask.to_numpy(), "carried_forward", "missing"),
        ),
        columns=INDICATOR_COLS,
        index=wide.index,
    )
    if mice_enabled:
        reconstructed = wide[INDICATOR_COLS].notna().to_numpy() & (
            provenance.to_numpy() == "missing"
        )
        provenance = provenance.mask(pd.DataFrame(reconstructed, columns=INDICATOR_COLS, index=wide.index), "mice_imputed")
    provenance.insert(0, "country_iso3", wide["country_iso3"].values)
    provenance.insert(1, "year", wide["year"].values)
    provenance_path = write_df(
        provenance,
        project_root
        / (pre_cfg.get("provenance") or {}).get(
            "path", "data/processed/cell_provenance.parquet"
        ),
    )
    provenance_shares = {
        state: float(
            (provenance[INDICATOR_COLS].to_numpy() == state).sum()
            / provenance[INDICATOR_COLS].size
        )
        for state in ("official", "carried_forward", "mice_imputed", "missing")
    }
    logger.info("Cell provenance shares: %s", provenance_shares)

    if log_controls:
        wide = _apply_log_transforms(wide, log_controls)

    # Drop duplicate country-year keys
    before = len(wide)
    wide = wide.drop_duplicates(subset=["country_iso3", "year"], keep="last")
    dropped_dupes = before - len(wide)

    keep_cols = (
        ["country_iso3", "year", "group_id", "group_name"]
        + [c for c in INDICATOR_COLS if c in wide.columns]
    )
    wide = wide[keep_cols].sort_values(["country_iso3", "year"]).reset_index(drop=True)

    interim_path = write_df(
        wide, ensure_dir(project_root / "data" / "interim") / "panel_clean.parquet"
    )
    processed_path = write_df(
        wide, ensure_dir(project_root / "data" / "processed") / "panel_wide.parquet"
    )

    tracker = LineageTracker(project_root)
    tracker.record(
        node_id="raw.integrated",
        node_type="dataset",
        parent_ids=[],
        transform="ingestion",
        path=raw_path.relative_to(project_root),
    )
    tracker.record(
        node_id="interim.panel_clean",
        node_type="dataset",
        parent_ids=["raw.integrated"],
        transform="preprocess_pivot_impute",
        path=interim_path.relative_to(project_root),
    )
    tracker.record(
        node_id="processed.panel_wide",
        node_type="dataset",
        parent_ids=["interim.panel_clean"],
        transform="persist_wide_panel",
        path=processed_path.relative_to(project_root),
    )

    non_null = {
        c: float(wide[c].notna().mean()) for c in INDICATOR_COLS if c in wide.columns
    }
    result = {
        "interim_path": str(interim_path),
        "processed_path": str(processed_path),
        "rows": int(len(wide)),
        "columns": list(wide.columns),
        "n_countries": int(wide["country_iso3"].nunique()),
        "year_min": int(wide["year"].min()) if len(wide) else None,
        "year_max": int(wide["year"].max()) if len(wide) else None,
        "dropped_duplicates": int(dropped_dupes),
        "max_gap_years": max_gap,
        "log_controls": log_controls,
        "non_null_ratio": non_null,
        "shape": [int(wide.shape[0]), int(wide.shape[1])],
        "mice": mice_report,
        "scale_harmonization": harmonized,
        "provenance_path": str(provenance_path.relative_to(project_root)),
        "provenance_shares": provenance_shares,
    }
    logger.info(
        "Preprocessing complete: rows=%s countries=%s path=%s",
        result["rows"],
        result["n_countries"],
        processed_path,
    )
    return result
