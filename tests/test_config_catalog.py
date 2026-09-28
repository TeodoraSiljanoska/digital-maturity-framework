"""Config and catalog smoke tests."""

from __future__ import annotations

from pathlib import Path

from catalog.data_catalog import DataCatalog
from pipeline.pipeline_config import PipelineConfig

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_config_loads():
    cfg = PipelineConfig.load(ROOT)
    countries = cfg.iso3_list()
    assert len(countries) == 12
    assert "MKD" in countries
    start, end = cfg.year_range()
    assert start == 2012
    assert end == int(cfg.research["panel"]["end_year"])
    assert end >= 2023


def test_data_catalog_indicators():
    catalog = DataCatalog.from_project(ROOT)
    ids = set(catalog.ids())
    for vid in ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "C1", "C2"]:
        assert vid in ids
    wb = catalog.by_source("world_bank")
    assert len(wb) >= 5