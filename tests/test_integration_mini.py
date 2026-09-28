"""Mini integration: DMI + stats path on a tiny synthetic panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.io import ensure_dir, write_df
from index.build import build_dmi_panel
from statistics.descriptive import run as stats_run
from pipeline.pipeline_config import PipelineConfig


def test_mini_index_and_stats(tmp_path: Path):
    root = tmp_path
    # Build synthetic feature panel matching research schema
    rng = np.random.default_rng(0)
    rows = []
    for iso, gid in [("AAA", "g1"), ("BBB", "g1"), ("CCC", "g2"), ("DDD", "g2")]:
        for year in range(2012, 2024):
            rows.append(
                {
                    "country_iso3": iso,
                    "year": year,
                    "group_id": gid,
                    "group_name": gid,
                    **{f"X{i}": float(rng.uniform(10, 100)) for i in range(1, 11)},
                    "C1": float(rng.uniform(5000, 50000)),
                    "C2": float(rng.uniform(0.4, 0.95)),
                }
            )
    feats = pd.DataFrame(rows)
    for col in [f"X{i}" for i in range(1, 11)] + ["C1", "C2"]:
        feats[f"{col}_lag1"] = feats.groupby("country_iso3")[col].shift(1)

    processed = ensure_dir(root / "data" / "processed")
    write_df(feats, processed / "panel_features.parquet")

    cfg = PipelineConfig.load(Path(__file__).resolve().parents[1])
    # Point index build at tmp by copying index config usage directly
    out, meta = build_dmi_panel(feats, cfg.index)
    assert out["DMI"].notna().sum() >= 30
    write_df(out, processed / "analysis_panel.parquet")
    write_df(out, ensure_dir(root / "outputs" / "data") / "dmi_panel.parquet")

    result = stats_run(root, cfg)
    assert Path(root / "results" / "descriptive").exists() or result is not None
    assert meta["n_dmi"] >= 30