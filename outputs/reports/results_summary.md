# Results summary

_Generated: 2026-09-28T16:41:38.997252+00:00_

This report is generated from pipeline artifacts under `results/` and `outputs/`.

## Analysis panel

- Rows: **168**
- Countries: **12**
- Years: **2012–2025**
- Mean DMI: **60.30**

## DMI by group

| n | mean | std | variance | skewness | kurtosis | min | max | shapiro_stat | shapiro_pvalue | shapiro_applied | shapiro_note | group_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 51.3549 | 11.9891 | 143.7380 | -0.0303 | -1.0480 | 30.5228 | 71.1423 | 0.9611 | 0.1611 | True | — | balkan |
| 42 | 75.6338 | 4.2674 | 18.2108 | -0.0404 | -0.7828 | 67.4949 | 84.2780 | 0.9766 | 0.5346 | True | — | developed_eu |
| 42 | 72.4833 | 4.5690 | 20.8761 | 0.5695 | -0.3544 | 64.6265 | 81.9277 | 0.9511 | 0.0710 | True | — | developed_non_eu |
| 42 | 41.7403 | 12.4619 | 155.2999 | -0.7438 | -0.0655 | 15.1221 | 57.9381 | 0.9107 | 0.0031 | True | — | developing |

## Econometrics

- Formula: `DMI ~ X1_lag1 + X2_lag1 + X3_lag1 + X4_lag1 + X5_lag1 + X6_lag1 + X7_lag1 + X8_lag1 + X9_lag1 + X10_lag1 + C1_lag1 + C2_lag1`
- N observations: **156**
- Preferred model (Hausman): **FE**

### FE coefficients (excerpt)

| model | term | coefficient | std_error | pvalue | group_id |
| --- | --- | --- | --- | --- | --- |
| FE | const | -115.2030 | 23.5067 | 0.0000 | — |
| FE | X1_lag1 | 0.0204 | 0.0288 | 0.4797 | — |
| FE | X2_lag1 | 0.2867 | 0.0838 | 0.0008 | — |
| FE | X3_lag1 | 20.5443 | 8.6187 | 0.0186 | — |
| FE | X4_lag1 | 0.2851 | 0.0429 | 0.0000 | — |
| FE | X5_lag1 | 0.0768 | 0.0603 | 0.2056 | — |
| FE | X6_lag1 | 0.1777 | 0.0691 | 0.0112 | — |
| FE | X7_lag1 | 0.0024 | 0.0440 | 0.9558 | — |
| FE | X8_lag1 | 0.9253 | 0.7176 | 0.1995 | — |
| FE | X9_lag1 | 0.0355 | 0.0950 | 0.7094 | — |
| FE | X10_lag1 | 4.9963 | 3.3643 | 0.1399 | — |
| FE | C1_lag1 | 9.6881 | 2.7233 | 0.0005 | — |
| FE | C2_lag1 | 7.2392 | 12.1802 | 0.5533 | — |

## Machine learning

- Best model: **ridge**
- Test RMSE: **1.5248**
- Model id: `ridge_20260928T163149Z`

### Model comparison

| model_type | status | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| random_forest | ok | 2.6568 | 2.2276 | 0.9555 |
| xgboost | ok | 2.2748 | 1.6583 | 0.9674 |
| lightgbm | ok | 2.0497 | 1.6275 | 0.9735 |
| catboost | ok | 2.3195 | 1.8771 | 0.9661 |
| svr | ok | 1.6416 | 1.2882 | 0.9830 |
| mlp | ok | 12.9223 | 10.7394 | -0.0516 |
| linear_regression | ok | 1.5468 | 1.2569 | 0.9849 |
| ridge | ok | 1.5248 | 1.2145 | 0.9854 |

## Comparative group metrics

| group_id | n | mean_dmi | std_dmi | min_dmi | max_dmi | start_year | end_year | mean_dmi_start | mean_dmi_end | abs_growth | pct_growth | cagr | n_obs | n_countries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balkan | 42 | 51.3549 | 11.9891 | 30.5228 | 71.1423 | 2012 | 2025 | 42.8496 | 61.6913 | 18.8418 | 0.4397 | 0.0284 | 42 | 3 |
| developed_eu | 42 | 75.6338 | 4.2674 | 67.4949 | 84.2780 | 2012 | 2025 | 69.9121 | 80.4646 | 10.5525 | 0.1509 | 0.0109 | 42 | 3 |
| developed_non_eu | 42 | 72.4833 | 4.5690 | 64.6265 | 81.9277 | 2012 | 2025 | 66.1754 | 77.3359 | 11.1606 | 0.1687 | 0.0121 | 42 | 3 |
| developing | 42 | 41.7403 | 12.4619 | 15.1221 | 57.9381 | 2012 | 2025 | 30.6611 | 53.8148 | 23.1537 | 0.7551 | 0.0442 | 42 | 3 |

## Convergence

- Source: `results\convergence\sigma_convergence.csv`
- Sigma start → end: **18.8454 → 6.9681**
- Converging: **True**

## Result registry

- Registered results: **45**
