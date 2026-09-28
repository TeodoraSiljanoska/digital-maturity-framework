"""Unit tests for Digital Maturity Index construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from index.build import build_dmi_panel


def test_dmi_bounds_and_pillars():
    rows = []
    for i, iso in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        for year in range(2012, 2024):
            rows.append(
                {
                    "country_iso3": iso,
                    "year": year,
                    "group_id": "g1",
                    "X1": 50 + i * 10 + year % 3,
                    "X2": 40 + i * 5,
                    "X3": 0.5 + i * 0.1,
                    "X4": 60 + i,
                    "X5": 70 + i,
                    "X6": 45 + i,
                    "X7": 50 + i,
                    "X8": 1.0 + i * 0.2,
                    "X9": 10 + i,
                    "X10": 0.4 + i * 0.1,
                }
            )
    df = pd.DataFrame(rows)
    index_cfg = {
        "scale": [0, 100],
        "default_weights": {
            "infrastructure": 0.2,
            "e_government": 0.2,
            "skills": 0.2,
            "trust_innovation": 0.2,
            "digital_economy": 0.2,
        },
        "pillars": {
            "infrastructure": {"indicators": ["X1", "X2"]},
            "e_government": {"indicators": ["X3"]},
            "skills": {"indicators": ["X4"]},
            "trust_innovation": {"indicators": ["X5", "X6", "X7"]},
            "digital_economy": {"indicators": ["X8", "X9", "X10"]},
        },
        "sensitivity": {"pca": True, "weight_shock_pct": 0.2, "n_shocks": 5},
        "reliability": {"cronbach_alpha": True, "item_correlations": True},
    }
    out, meta = build_dmi_panel(df, index_cfg)
    assert "DMI" in out.columns
    assert out["DMI"].notna().all()
    assert float(out["DMI"].min()) >= -1e-6
    assert float(out["DMI"].max()) <= 100 + 1e-6
    assert meta.get("n_dmi", len(out)) >= 30
    assert np.isfinite(out["DMI"]).all()