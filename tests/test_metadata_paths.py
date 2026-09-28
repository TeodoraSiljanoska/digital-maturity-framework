"""Generated metadata must not carry absolute local paths (the repository is published)."""

from __future__ import annotations

import pandas as pd

from data_sources.gii import GIIAdapter
from framework.outputs import build_framework_status


def test_snapshot_metadata_records_the_configured_relative_path(tmp_path):
    target = tmp_path / "data" / "raw" / "gii"
    target.mkdir(parents=True)
    (target / "gii_panel.csv").write_text("country_iso3,year,GII_SCORE\nMKD,2020,35.0\n", encoding="utf-8")

    meta = GIIAdapter(project_root=tmp_path).get_metadata()

    assert meta["snapshot_file"] == "data/raw/gii/gii_panel.csv"
    assert meta["snapshot_exists"] is True
    assert str(tmp_path) not in str(meta)


def test_framework_status_records_only_the_project_directory_name(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {"country_iso3": ["MKD", "SVN"], "group_id": ["balkan", "balkan"], "year": [2025, 2025], "DMI": [55.0, 70.0]}
    ).to_parquet(processed / "dmi_panel.parquet")

    status = build_framework_status(tmp_path)

    assert status["project_root"] == tmp_path.name
