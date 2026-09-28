"""Edition-boundary diagnostic: level shifts are flagged, ordinary movement is not."""

from __future__ import annotations

import numpy as np
import pandas as pd

from validation.edition_breaks import detect_edition_breaks


def _panel(shift_at=None, shift=-10.0, years=(2019, 2020, 2021, 2022, 2023), n=12, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n):
        level = 40.0 + 3.0 * c
        for year in years:
            value = level + 0.5 * (year - years[0]) + rng.normal(0, 0.6)
            if shift_at is not None and year >= shift_at:
                value += shift
            rows.append({"country_iso3": f"C{c:02d}", "year": year, "X7": value})
    return pd.DataFrame(rows)


def test_uniform_shift_at_one_boundary_is_detected():
    breaks = detect_edition_breaks(_panel(shift_at=2020), ["X7"]).set_index(["from_year", "to_year"])

    assert breaks.loc[(2019, 2020), "status"] == "detected"
    assert breaks.loc[(2019, 2020), "share_same_sign"] == 1.0
    others = breaks.drop(index=(2019, 2020))
    assert not others["statistical_flag"].any()


def test_series_without_shift_has_no_flags():
    breaks = detect_edition_breaks(_panel(), ["X7"])
    assert not breaks["statistical_flag"].any()


def test_declared_break_is_reported_even_when_untestable():
    two_editions = _panel(years=(2022, 2023), shift_at=2023, shift=12.0)
    cfg = {"indicators": {"X5": {}, "X7": {"declared_breaks": [{"between": [2022, 2023], "note": "new framework"}]}}}

    breaks = detect_edition_breaks(two_editions, ["X7"], editions_cfg=cfg)

    assert breaks.loc[0, "status"] == "declared"
    assert breaks.loc[0, "declared_note"] == "new framework"


def test_sparse_years_do_not_create_editions():
    panel = _panel()
    # A single stray value in 2018 must not open a 2018->2019 boundary.
    panel = pd.concat([panel, pd.DataFrame([{"country_iso3": "C00", "year": 2018, "X7": 41.0}])])
    breaks = detect_edition_breaks(panel, ["X7"])
    assert breaks["from_year"].min() == 2019
