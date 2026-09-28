"""Official-only masking pool and the cached World Bank vintage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_sources.world_bank import WorldBankAdapter
from imputation.mice import validate_imputation


def _panel(n_countries=6, n_years=12):
    rng = np.random.default_rng(11)
    rows = []
    for c in range(n_countries):
        for t in range(n_years):
            rows.append(
                {
                    "country_iso3": f"C{c:02d}",
                    "group_id": "g1" if c < n_countries // 2 else "g2",
                    "year": 2012 + t,
                    "X1": 20.0 + 10.0 * c + t + rng.normal(0, 0.5),
                    "X2": 16.0 + 8.0 * c + 1.5 * t + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_official_pool_limits_which_cells_can_be_hidden():
    df = _panel()
    eligible = pd.DataFrame({"X1": df["year"] <= 2018, "X2": np.ones(len(df), dtype=bool)})

    result = validate_imputation(
        df, ["X1", "X2"], mask_fraction=0.15, n_repeats=1, seed=5, eligible=eligible,
        base_kwargs={"n_estimators": 10, "max_iter": 3},
    )

    masked = result.query("configuration == 'mice_country_aware'").set_index("variable")["n_masked"]
    assert masked["X1"] == int(np.floor(eligible["X1"].sum() * 0.15))
    assert masked["X2"] == int(np.floor(len(df) * 0.15))


def test_cached_vintage_serves_stored_values_without_network(tmp_path):
    vintage = pd.DataFrame(
        {
            "country_iso3": ["MKD", "MKD", "SRB"],
            "year": [2023, 2024, 2023],
            "indicator_id": ["X4", "X4", "X4"],
            "indicator_code": ["IT.NET.USER.ZS"] * 3,
            "value": [83.0, 84.5, 85.4],
            "source": ["world_bank"] * 3,
            "retrieved_at": ["2026-08-09T20:18:24+00:00"] * 3,
        }
    )
    path = tmp_path / "wdi_vintage.parquet"
    vintage.to_parquet(path, index=False)
    adapter = WorldBankAdapter(
        config={"use_cached_raw": True, "cached_raw_path": str(path), "base_url": "http://invalid.invalid"},
        project_root=tmp_path,
    )

    out = adapter.get_data(["IT.NET.USER.ZS"], ["MKD"], 2012, 2025, indicator_id_map={"IT.NET.USER.ZS": "X4"})

    assert out["value"].tolist() == [83.0, 84.5]
    assert set(out["retrieved_at"]) == {"2026-08-09T20:18:24+00:00"}
    assert adapter.get_metadata()["mode"] == "cached_vintage"
