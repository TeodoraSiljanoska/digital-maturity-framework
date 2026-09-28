"""Cell provenance must follow each (country, year), whatever order the config lists countries in."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.pipeline_config import PipelineConfig
from preprocessing.pipeline import _apply_edition_filter, _reconstruction_scope, run


def _config(root, countries):
    groups = {
        f"g{i}": {"name": f"Group {i}", "countries": [{"iso3": iso}]}
        for i, iso in enumerate(countries)
    }
    return PipelineConfig(
        project_root=root,
        research={"panel": {"start_year": 2012, "end_year": 2017}},
        countries={"groups": groups},
        preprocessing={
            "missing_values": {"max_gap_years": 2},
            "transforms": {"log_controls": []},
        },
        imputation={"enabled": False},
    )


def _raw(rows):
    frame = pd.DataFrame(rows, columns=["country_iso3", "year", "indicator_id", "value"])
    frame["indicator_code"] = frame["indicator_id"]
    frame["source"] = "test"
    frame["retrieved_at"] = "2026-01-01T00:00:00+00:00"
    return frame


def test_labels_follow_country_year_when_config_order_is_not_alphabetical(tmp_path):
    # ZZA is listed first in the config but sorts last; it publishes X1 once, in 2012.
    rows = [("ZZA", 2012, "X1", 5.0)]
    rows += [("AAB", year, "X1", 10.0 + year - 2012) for year in range(2012, 2018)]
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _raw(rows).to_parquet(raw_dir / "integrated_raw.parquet", index=False)

    run(tmp_path, _config(tmp_path, ["ZZA", "AAB"]))

    prov = pd.read_parquet(tmp_path / "data" / "processed" / "cell_provenance.parquet")
    label = prov.set_index(["country_iso3", "year"])["X1"]
    assert label[("ZZA", 2012)] == "official"
    assert label[("ZZA", 2013)] == "carried_forward"
    assert label[("ZZA", 2014)] == "carried_forward"
    assert label[("ZZA", 2015)] == "missing"
    assert (label.loc["AAB"] == "official").all()


def test_reconstruction_scope_separates_interpolation_from_extrapolation():
    countries = pd.Series(["AAA"] * 5 + ["BBB"] * 5)
    official = pd.DataFrame({"X1": [False, True, False, True, False, False, False, False, False, False]})
    provenance = pd.DataFrame(
        {
            "X1": [
                "mice_imputed", "official", "mice_imputed", "official", "mice_imputed",
                "mice_imputed", "mice_imputed", "mice_imputed", "mice_imputed", "mice_imputed",
            ]
        }
    )
    for col in ["X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "C1", "C2"]:
        official[col] = True
        provenance[col] = "official"

    scope = _reconstruction_scope(provenance, official, countries)

    assert scope["X1"].tolist()[:5] == ["backcast", "", "interior", "", "forecast"]
    assert set(scope["X1"].tolist()[5:]) == {"unobserved_series"}


def test_edition_filter_blanks_only_excluded_editions():
    df = pd.DataFrame(
        {"country_iso3": ["AAA"] * 4, "year": [2019, 2020, 2024, 2025], "X7": [6.0, 60.0, 70.0, 55.0]}
    )
    cfg = {"indicators": {"X7": {"exclude_reference_years": [2019, 2025], "family": "3-pillar"}}}

    out, applied = _apply_edition_filter(df, cfg)

    assert out["X7"].isna().tolist() == [True, False, False, True]
    assert applied[0]["n_values_excluded"] == 2


def test_edition_filter_is_a_no_op_without_config():
    df = pd.DataFrame({"country_iso3": ["AAA"], "year": [2019], "X7": [6.0]})
    out, applied = _apply_edition_filter(df, {})
    assert out.equals(df)
    assert applied == []
