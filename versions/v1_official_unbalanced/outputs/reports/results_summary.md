# Results summary

_Generated: 2026-08-09T20:19:03.525553+00:00_

This report is generated from pipeline artifacts under `results/` and `outputs/`.

## Analysis panel

- Rows: **168**
- Countries: **12**
- Years: **2012–2025**
- Mean DMI: **59.61**

## DMI by group

| n | mean | std | variance | skewness | kurtosis | min | max | shapiro_stat | shapiro_pvalue | shapiro_applied | shapiro_note | group_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 51.8338 | 12.3048 | 151.4079 | -0.0454 | -0.9801 | 28.8358 | 72.8638 | 0.9605 | 0.1536 | True | — | balkan |
| 42 | 74.7411 | 6.2597 | 39.1835 | -0.1683 | -0.8516 | 62.9151 | 85.2877 | 0.9687 | 0.2977 | True | — | developed_eu |
| 42 | 69.9259 | 5.9355 | 35.2299 | 0.5299 | -0.6499 | 61.8900 | 82.5060 | 0.9393 | 0.0270 | True | — | developed_non_eu |
| 42 | 41.9451 | 13.8484 | 191.7790 | -0.4972 | -0.3547 | 14.2392 | 62.4654 | 0.9348 | 0.0188 | True | — | developing |

## Econometrics

- Formula: `DMI ~ X1_lag1 + X2_lag1 + X3_lag1 + X4_lag1 + X5_lag1 + X6_lag1 + X7_lag1 + X8_lag1 + X9_lag1 + X10_lag1 + C1_lag1 + C2_lag1`
- N observations: **84**
- Preferred model (Hausman): **RE**

### FE coefficients (excerpt)

| model | term | coefficient | std_error | pvalue | group_id |
| --- | --- | --- | --- | --- | --- |
| FE | const | -98.5623 | 89.5692 | 0.2756 | — |
| FE | X1_lag1 | -0.0331 | 0.1308 | 0.8010 | — |
| FE | X2_lag1 | 0.7310 | 0.2639 | 0.0075 | — |
| FE | X3_lag1 | -14.1564 | 22.2639 | 0.5273 | — |
| FE | X4_lag1 | 0.3511 | 0.0823 | 0.0001 | — |
| FE | X5_lag1 | 0.0787 | 0.0328 | 0.0197 | — |
| FE | X6_lag1 | -0.2122 | 0.2336 | 0.3672 | — |
| FE | X7_lag1 | 0.0435 | 0.0130 | 0.0014 | — |
| FE | X8_lag1 | 4.0997 | 3.3296 | 0.2230 | — |
| FE | X9_lag1 | -0.0073 | 0.1262 | 0.9541 | — |
| FE | X10_lag1 | 15.4419 | 7.5518 | 0.0453 | — |
| FE | C1_lag1 | 8.6403 | 6.9507 | 0.2187 | — |
| FE | C2_lag1 | 22.0478 | 79.5028 | 0.7825 | — |

## Machine learning

- Best model: **catboost**
- Test RMSE: **2.0801**
- Model id: `catboost_20260809T201832Z`

### Model comparison

| model_type | status | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| random_forest | ok | 3.3552 | 2.3923 | 0.9070 |
| xgboost | ok | 3.0144 | 2.1698 | 0.9249 |
| lightgbm | ok | 4.0876 | 2.9055 | 0.8619 |
| catboost | ok | 2.0801 | 1.4086 | 0.9642 |
| svr | ok | 2.5957 | 1.8061 | 0.9443 |
| mlp | ok | 4.2886 | 3.4814 | 0.8480 |
| linear_regression | ok | 2.2548 | 1.7491 | 0.9580 |
| ridge | ok | 2.0859 | 1.7061 | 0.9640 |

## Comparative group metrics

| group_id | n | mean_dmi | std_dmi | min_dmi | max_dmi | start_year | end_year | mean_dmi_start | mean_dmi_end | abs_growth | pct_growth | cagr | n_obs | n_countries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balkan | 42 | 51.8338 | 12.3048 | 28.8358 | 72.8638 | 2012 | 2025 | 42.2605 | 65.6088 | 23.3482 | 0.5525 | 0.0344 | 42 | 3 |
| developed_eu | 42 | 74.7411 | 6.2597 | 62.9151 | 85.2877 | 2012 | 2025 | 71.7824 | 81.8519 | 10.0695 | 0.1403 | 0.0101 | 42 | 3 |
| developed_non_eu | 42 | 69.9259 | 5.9355 | 61.8900 | 82.5060 | 2012 | 2025 | 67.7481 | 77.8410 | 10.0929 | 0.1490 | 0.0107 | 42 | 3 |
| developing | 42 | 41.9451 | 13.8484 | 14.2392 | 62.4654 | 2012 | 2025 | 30.0131 | 58.3714 | 28.3583 | 0.9449 | 0.0525 | 42 | 3 |

## Convergence

- Source: `results/convergence/sigma_convergence.csv`
- Sigma start → end: **19.9525 → 4.9098**
- Converging: **True**

## Result registry

- Registered results: **38**
