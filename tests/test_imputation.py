"""Tests for the v2 imputation layer: carry-forward reach, PMM draws, Rubin pooling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation.mice import apply_mice_imputation, multiple_imputation, validate_imputation
from imputation.robustness import _rubin_pool
from preprocessing.pipeline import _harmonize_scales, _impute_missing


def _panel(n_countries: int = 6, n_years: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for c in range(n_countries):
        level = 20.0 + 10.0 * c
        for t in range(n_years):
            rows.append(
                {
                    "country_iso3": f"C{c:02d}",
                    "group_id": "g1" if c < n_countries // 2 else "g2",
                    "year": 2012 + t,
                    "X1": level + t + rng.normal(0, 0.5),
                    "X2": 0.8 * level + 1.5 * t + rng.normal(0, 0.5),
                    "X5": 0.5 * level + 0.8 * t + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_carry_forward_respects_configured_gap():
    """A single observation may not spread beyond max_gap years in either direction."""
    df = _panel(n_countries=2, n_years=12)
    df["X5"] = np.nan
    df.loc[df["year"] == 2018, "X5"] = 50.0

    filled = _impute_missing(df, ["X5"], max_gap=2)
    covered = sorted(filled.loc[filled["X5"].notna(), "year"].unique())

    assert covered == [2016, 2017, 2018, 2019, 2020]


def test_scale_harmonization_rescales_only_earlier_editions():
    df = pd.DataFrame(
        {
            "country_iso3": ["AAA"] * 4,
            "year": [2018, 2019, 2020, 2021],
            "X7": [5.0, 6.0, 60.0, 62.0],
        }
    )
    out, applied = _harmonize_scales(
        df, [{"variable": "X7", "apply_to_years_before": 2020, "multiply_by": 10.0}]
    )

    assert out["X7"].tolist() == [50.0, 60.0, 60.0, 62.0]
    assert applied[0]["n_values_rescaled"] == 2


def test_pmm_draws_stay_within_observed_support():
    """Predictive mean matching must reuse observed values, never invent new ones."""
    df = _panel()
    observed = set(df["X5"].round(6).tolist())
    df.loc[df["year"].isin([2015, 2016]), "X5"] = np.nan

    filled, mask, report = apply_mice_imputation(
        df, ["X1", "X2", "X5"], seed=1, max_iter=3, n_estimators=8, pmm=True, donors=3
    )

    imputed_values = filled.loc[mask["X5"].to_numpy(), "X5"].round(6)
    assert len(imputed_values) > 0
    assert set(imputed_values).issubset(observed)
    assert report["draw_type"] == "predictive_mean_matching"


def test_multiple_imputation_produces_varying_draws():
    df = _panel()
    df.loc[df["year"].isin([2015, 2016]), "X5"] = np.nan

    draws, meta = multiple_imputation(
        df, ["X1", "X2", "X5"], n_imputations=3, seed=3, max_iter=3, n_estimators=8
    )

    assert meta["n_imputations"] == 3
    means = {round(float(d["X5"].mean()), 6) for d in draws}
    assert len(means) > 1, "draws must differ, otherwise Rubin pooling adds no variance"


def test_rubin_pooling_adds_between_draw_variance():
    per_draw = [
        pd.DataFrame({"term": ["b"], "coefficient": [value], "std_error": [0.10]})
        for value in (0.40, 0.55, 0.30, 0.60, 0.45)
    ]

    pooled = _rubin_pool(per_draw).iloc[0]

    assert pooled["coefficient"] == np.mean([0.40, 0.55, 0.30, 0.60, 0.45])
    assert pooled["std_error"] > 0.10
    assert 0.0 < pooled["fraction_missing_information"] < 1.0


def test_masked_cell_benchmark_beats_global_median():
    df = _panel()
    scores = validate_imputation(
        df,
        ["X1", "X2", "X5"],
        mask_fraction=0.2,
        n_repeats=1,
        seed=5,
        configurations={"mice_country_aware": {"include_country_dummies": True}},
        base_kwargs={"max_iter": 3, "n_estimators": 8},
    )

    mice_error = scores.loc[scores["configuration"] == "mice_country_aware", "nrmse_sd"].mean()
    median_error = scores.loc[
        scores["configuration"] == "baseline_global_median", "nrmse_sd"
    ].mean()
    assert mice_error < median_error
