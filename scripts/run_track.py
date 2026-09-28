#!/usr/bin/env python3
"""
Run one research track in a throwaway workspace and snapshot it under versions/.

A track is a configuration set, a set of source data and a missing-data arm.
Configuration and raw data come either from the working tree or from a git
reference (for example the ``v2-frozen`` tag), so an earlier track can be
re-run with the current code while the active outputs/ and results/ trees stay
untouched. Generalises scripts/run_v1b_track.py.

Usage:
    python scripts/run_track.py --name v2r_reproduction --config-ref v2-frozen \
        --raw-ref v2-frozen --from DATA_VALIDATED
    python scripts/run_track.py --name v3_edition_harmonised
    python scripts/run_track.py --name v3b_official_only --arm official_only
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "versions"
INDICATORS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "C1", "C2"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)


def _extract_from_ref(ref: str, path: str, destination: Path) -> None:
    """Materialise ``path`` as it exists at git ``ref`` inside ``destination``."""
    archive = _git("archive", "--format=tar", ref, path).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination)


def _code_commit() -> str:
    sha = _git("rev-parse", "--short", "HEAD").stdout.decode().strip()
    dirty = _git("status", "--porcelain", "--", "src", "scripts").stdout.decode().strip()
    return f"{sha}{'+uncommitted-changes' if dirty else ''}"


def _set_official_only(config_dir: Path) -> None:
    """Switch reconstruction off: gaps beyond the carry-forward reach stay empty."""
    pre_path = config_dir / "preprocessing.yaml"
    pre = yaml.safe_load(pre_path.read_text(encoding="utf-8"))
    missing = pre.setdefault("missing_values", {})
    missing["strategy"] = "country_ffill_bfill_official_only"
    missing.setdefault("multivariate", {})["enabled"] = False
    pre_path.write_text(yaml.safe_dump(pre, sort_keys=False, allow_unicode=True), encoding="utf-8")

    imp_path = config_dir / "imputation.yaml"
    imp = yaml.safe_load(imp_path.read_text(encoding="utf-8"))
    imp["enabled"] = False
    imp_path.write_text(yaml.safe_dump(imp, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_workspace(
    name: str, *, config_ref: Optional[str], raw_ref: Optional[str], arm: str
) -> Path:
    workspace = ROOT / f".track_{name}"
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "data").mkdir(parents=True)

    if config_ref:
        _extract_from_ref(config_ref, "config", workspace)
    else:
        shutil.copytree(ROOT / "config", workspace / "config")

    if raw_ref:
        _extract_from_ref(raw_ref, "data/raw", workspace)
    else:
        shutil.copytree(ROOT / "data" / "raw", workspace / "data" / "raw")

    if arm == "official_only":
        _set_official_only(workspace / "config")
    return workspace


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def snapshot(workspace: Path, name: str) -> Path:
    destination = VERSIONS / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copytree(workspace / "outputs", destination / "outputs")
    shutil.copytree(workspace / "results", destination / "results")
    shutil.copytree(workspace / "data" / "processed", destination / "data_processed")
    shutil.copytree(workspace / "data" / "raw", destination / "data_raw")
    shutil.copytree(workspace / "config", destination / "config")
    return destination


def write_version_file(
    destination: Path,
    *,
    name: str,
    config_ref: Optional[str],
    raw_ref: Optional[str],
    arm: str,
    start_stage: str,
) -> Path:
    """Describe the snapshot from its own artifacts rather than from memory."""
    import pandas as pd

    prov = pd.read_parquet(destination / "data_processed" / "cell_provenance.parquet")
    cells = prov[[c for c in INDICATORS if c in prov.columns]].to_numpy()
    states = ("official", "carried_forward", "mice_imputed", "missing")
    counts = {s: int((cells == s).sum()) for s in states}

    scope_path = destination / "data_processed" / "cell_reconstruction_scope.parquet"
    scope_counts: Dict[str, int] = {}
    if scope_path.exists():
        scope = pd.read_parquet(scope_path)[[c for c in INDICATORS if c in prov.columns]].to_numpy()
        for s in ("interior", "backcast", "forecast", "unobserved_series"):
            scope_counts[s] = int((scope == s).sum())

    wide = pd.read_parquet(destination / "data_processed" / "panel_wide.parquet")
    x_cols = [f"X{i}" for i in range(1, 11)]
    complete = int((wide[x_cols].notna().sum(axis=1) == len(x_cols)).sum())

    dmi = pd.read_parquet(destination / "data_processed" / "dmi_panel.parquet")
    metrics = pd.read_csv(destination / "results" / "econometrics" / "econometrics_metrics.csv")
    fe_rows = metrics.query("model == 'FE'")
    fe_nobs = int(fe_rows.iloc[0]["nobs"]) if not fe_rows.empty else None
    ml = pd.read_csv(destination / "outputs" / "tables" / "ml_model_comparison.csv")

    audit = _read_json(destination / "outputs" / "audit" / "final_audit.json")
    audit = audit.get("data", audit).get("definition_of_complete", {})
    reproducibility = _read_json(destination / "outputs" / "audit" / "reproducibility.json")
    breaks_raw = _read_json(destination / "outputs" / "audit" / "edition_breaks_raw.json")
    breaks_filtered = _read_json(destination / "outputs" / "audit" / "edition_breaks_filtered.json")

    def _flag_list(summary: Dict[str, Any]) -> str:
        items = summary.get("flagged") or []
        if not items:
            return "[]"
        return "[" + ", ".join(
            f"{i['variable']} {i['from_year']}->{i['to_year']} ({i['status']})" for i in items
        ) + "]"

    total = sum(counts.values())
    packages = reproducibility.get("package_versions") or {}
    text = f"""version: {name}
generated_by: scripts/run_track.py
created_at: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
code_commit: {_code_commit()}
config_source: {'git:' + config_ref if config_ref else 'working tree'}
raw_data_source: {'git:' + raw_ref if raw_ref else 'working tree'}
missing_data_arm: {arm}
start_stage: {start_stage}
period: {int(dmi['year'].min())}-{int(dmi['year'].max())}

cell_provenance:
  official: {counts['official']} ({counts['official'] / total:.3f})
  carried_forward: {counts['carried_forward']} ({counts['carried_forward'] / total:.3f})
  mice_imputed: {counts['mice_imputed']} ({counts['mice_imputed'] / total:.3f})
  missing: {counts['missing']} ({counts['missing'] / total:.3f})
reconstruction_scope: {scope_counts or 'n/a'}

edition_breaks:
  raw_flagged: {_flag_list(breaks_raw)}
  filtered_flagged: {_flag_list(breaks_filtered)}

coverage:
  dmi_panel_rows: {len(dmi)}
  countries: {dmi['country_iso3'].nunique()}
  complete_10_indicator_baskets: {complete} of {len(wide)}
  fixed_effects_estimation_rows: {fe_nobs}
  ml_sample: {int(ml['n_train'].iloc[0])} train / {int(ml['n_test'].iloc[0])} test

dmi: mean {dmi['DMI'].mean():.3f}, sd {dmi['DMI'].std(ddof=1):.3f}, min {dmi['DMI'].min():.3f}, max {dmi['DMI'].max():.3f}
audit: {audit.get('n_pass')} of {audit.get('n_criteria')} criteria pass
environment: python {'.'.join(str(v) for v in reproducibility.get('python_version_info', []))}; scikit-learn {packages.get('scikit-learn')}; xgboost {packages.get('xgboost')}; lightgbm {packages.get('lightgbm')}; catboost {packages.get('catboost')}
"""
    path = destination / "VERSION.txt"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="snapshot directory under versions/")
    parser.add_argument("--config-ref", default=None, help="git ref to take config/ from")
    parser.add_argument("--raw-ref", default=None, help="git ref to take data/raw/ from")
    parser.add_argument("--arm", choices=["mice", "official_only"], default="mice")
    parser.add_argument("--from", dest="from_stage", default="RAW_DATA_ACQUIRED")
    parser.add_argument("--keep", action="store_true", help="keep the workspace afterwards")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.orchestrator import PipelineOrchestrator  # noqa: PLC0415

    workspace = build_workspace(
        args.name, config_ref=args.config_ref, raw_ref=args.raw_ref, arm=args.arm
    )
    print(f"Workspace: {workspace}")
    PipelineOrchestrator(workspace).run(from_stage=args.from_stage)

    destination = snapshot(workspace, args.name)
    version = write_version_file(
        destination,
        name=args.name,
        config_ref=args.config_ref,
        raw_ref=args.raw_ref,
        arm=args.arm,
        start_stage=args.from_stage,
    )
    print(f"Snapshot: {destination}")
    print(version.read_text(encoding="utf-8"))
    if not args.keep:
        shutil.rmtree(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
