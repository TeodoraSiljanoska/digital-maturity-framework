#!/usr/bin/env python3
"""
Reproduce the v1b research track: the corrected pipeline with imputation off.

v1b answers the question "which conclusions survive if no cell is reconstructed?".
It must differ from v2 in exactly one respect, so rather than editing the active
configuration it builds a throwaway workspace that copies config/ and the source
data, disables the multivariate stage there, and runs the standard orchestrator
against that root. The active outputs/ and results/ trees are never touched.

Usage:
    python scripts/run_v1b_track.py            # run and snapshot
    python scripts/run_v1b_track.py --keep     # leave the workspace for inspection
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / ".v1b_run"
SNAPSHOT = ROOT / "versions" / "v1b_official_corrected"

CONFIGS_TO_PRESERVE = [
    "preprocessing",
    "imputation",
    "research",
    "machine_learning",
    "econometrics",
]


def build_workspace() -> Path:
    """Create a clean project root that shares the inputs but not the outputs."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "data").mkdir(parents=True)
    shutil.copytree(ROOT / "config", WORKSPACE / "config")
    for name in ("raw", "interim"):
        source = ROOT / "data" / name
        if source.is_dir():
            shutil.copytree(source, WORKSPACE / "data" / name)

    pre = WORKSPACE / "config" / "preprocessing.yaml"
    text = pre.read_text()
    text = text.replace(
        "  strategy: country_ffill_bfill_then_mice",
        "  strategy: country_ffill_bfill_official_only",
    ).replace(
        "  multivariate:\n    enabled: true",
        "  multivariate:\n    enabled: false",
    )
    pre.write_text(text)

    imp = WORKSPACE / "config" / "imputation.yaml"
    imp.write_text(imp.read_text().replace("enabled: true", "enabled: false", 1))
    return WORKSPACE


def snapshot() -> Path:
    """Mirror the v1/v2 snapshot layout so the three tracks are browsed alike."""
    if SNAPSHOT.exists():
        report = SNAPSHOT / "outputs" / "reports"
        keep = {p.name: p.read_bytes() for p in report.glob("*.docx")} if report.is_dir() else {}
        shutil.rmtree(SNAPSHOT)
    else:
        keep = {}
    SNAPSHOT.mkdir(parents=True)
    shutil.copytree(WORKSPACE / "outputs", SNAPSHOT / "outputs")
    shutil.copytree(WORKSPACE / "results", SNAPSHOT / "results")
    shutil.copytree(WORKSPACE / "data" / "processed", SNAPSHOT / "data_processed")
    if (WORKSPACE / "experiments").is_dir():
        shutil.copytree(WORKSPACE / "experiments", SNAPSHOT / "experiments")
    for name in CONFIGS_TO_PRESERVE:
        shutil.copy2(WORKSPACE / "config" / f"{name}.yaml", SNAPSHOT / f"{name}.yaml")
    for filename, payload in keep.items():
        target = SNAPSHOT / "outputs" / "reports" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return SNAPSHOT


def write_version_file() -> Path:
    """Describe the snapshot from its own artifacts rather than from memory."""
    import json

    import pandas as pd

    audit = json.loads((SNAPSHOT / "outputs/audit/final_audit.json").read_text())
    audit = audit.get("data", audit)["definition_of_complete"]

    prov = pd.read_parquet(SNAPSHOT / "data_processed/cell_provenance.parquet")
    cells = prov[[c for c in prov.columns if c.startswith(("X", "C"))]].to_numpy()
    shares = {
        state: (cells == state).sum() / cells.size
        for state in ("official", "carried_forward", "mice_imputed", "missing")
    }

    wide = pd.read_parquet(SNAPSHOT / "data_processed/panel_wide.parquet")
    indicators = [f"X{i}" for i in range(1, 11)]
    complete = int((wide[indicators].notna().sum(axis=1) == len(indicators)).sum())

    dmi = pd.read_parquet(SNAPSHOT / "data_processed/dmi_panel.parquet")
    metrics = pd.read_csv(SNAPSHOT / "results/econometrics/econometrics_metrics.csv")
    fe_nobs = int(metrics.query("model == 'FE'").iloc[0]["nobs"])
    ml = pd.read_csv(SNAPSHOT / "outputs/tables/ml_model_comparison.csv")

    text = f"""version: v1b_official_corrected
period: {int(dmi["year"].min())}-{int(dmi["year"].max())}
generated_by: scripts/run_v1b_track.py

purpose: >-
  Corrected counterpart of v1. Reproduces the "official data only" research track
  with the v2 revision-2 data corrections applied, so that v1b and v2 differ in
  exactly one respect - whether missing cells are reconstructed - and any
  divergence between them is attributable to the missing-data policy alone.

missing_strategy: official gaps retained
imputation: none (single-pass country ffill/bfill, max_gap_years=2, then cells stay empty)
corrections_applied:
  - X7 (Government AI Readiness) pre-2020 editions rescaled x10 onto the 0-100 scale
  - carry-forward runs once per country series (v1 ran it twice, doubling its reach)
  - VIF computed with an intercept
  - negative Hausman statistic reported as inconclusive; Mundlak regression decides FE vs RE
  - cell provenance recorded for every indicator cell

cell_provenance:
  official: {shares["official"]:.3f}
  carried_forward: {shares["carried_forward"]:.3f}
  missing: {shares["missing"]:.3f}
  mice_imputed: {shares["mice_imputed"]:.3f}

coverage:
  dmi_panel_rows: {len(dmi)}
  countries: {dmi["country_iso3"].nunique()}
  complete_10_indicator_baskets: {complete} of {len(wide)} ({complete / len(wide) * 100:.1f}%)
  fixed_effects_estimation_sample: {fe_nobs} of {len(wide)} rows (listwise deletion over 12 lagged regressors)
  ml_sample: {int(ml["n_train"].iloc[0])} train / {int(ml["n_test"].iloc[0])} test (median feature imputation fitted on training rows only)

audit: {audit["n_pass"]} of {audit["n_criteria"]} criteria pass

note: >-
  v1b does not replace v1. The original v1 snapshot is preserved unchanged in
  versions/v1_official_unbalanced/ and still contains the pre-correction data.
  Findings are compared against v2 in
  outputs/reports/Results_and_Discussion_v1b_OfficialCorrected.docx.
"""
    path = SNAPSHOT / "VERSION.txt"
    path.write_text(text)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the .v1b_run workspace after snapshotting",
    )
    parser.add_argument("--from", dest="from_stage", default="DATA_VALIDATED")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.orchestrator import PipelineOrchestrator  # noqa: PLC0415

    workspace = build_workspace()
    print(f"Workspace: {workspace}")
    PipelineOrchestrator(workspace).run(from_stage=args.from_stage)

    destination = snapshot()
    print(f"Snapshot: {destination}")
    print(f"Wrote {write_version_file()}")
    if not args.keep:
        shutil.rmtree(workspace)
        print("Workspace removed (pass --keep to retain it)")
    print(
        "Rebuild the chapter with: "
        "python scripts/build_results_discussion_v1b_docx.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
