# Results summary

_Generated: 2026-09-28T16:07:56.041448+00:00_

This report is generated from pipeline artifacts under `results/` and `outputs/`.

## Analysis panel

- Rows: **168**
- Countries: **12**
- Years: **2012–2025**
- Mean DMI: **60.59**

## DMI by group

| n | mean | std | variance | skewness | kurtosis | min | max | shapiro_stat | shapiro_pvalue | shapiro_applied | shapiro_note | group_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 51.1753 | 11.3776 | 129.4491 | 0.0421 | -1.0416 | 31.7685 | 71.1423 | 0.9619 | 0.1723 | True | — | balkan |
| 42 | 76.6179 | 4.3482 | 18.9065 | -0.2294 | -0.9537 | 67.9047 | 84.2780 | 0.9636 | 0.1983 | True | — | developed_eu |
| 42 | 72.8765 | 4.5423 | 20.6329 | 0.6783 | -0.1605 | 64.3359 | 82.5287 | 0.9342 | 0.0180 | True | — | developed_non_eu |
| 42 | 41.6769 | 12.8079 | 164.0424 | -0.7563 | -0.1434 | 14.6186 | 57.9381 | 0.9134 | 0.0037 | True | — | developing |

## Econometrics

- Formula: `DMI ~ X1_lag1 + X2_lag1 + X3_lag1 + X4_lag1 + X5_lag1 + X6_lag1 + X7_lag1 + X8_lag1 + X9_lag1 + X10_lag1 + C1_lag1 + C2_lag1`
- N observations: **30**
- Preferred model (Hausman): **FE**

### FE coefficients (excerpt)

| model | term | coefficient | std_error | pvalue |
| --- | --- | --- | --- | --- |
| FE | const | 208.6320 | 165.3131 | 0.2425 |
| FE | X1_lag1 | -0.0373 | 0.1109 | 0.7451 |
| FE | X2_lag1 | 0.7723 | 0.5172 | 0.1738 |
| FE | X3_lag1 | -4.9025 | 19.8856 | 0.8115 |
| FE | X4_lag1 | 0.2177 | 0.0799 | 0.0260 |
| FE | X6_lag1 | 0.0301 | 0.1702 | 0.8641 |
| FE | X7_lag1 | -0.0959 | 0.1385 | 0.5080 |
| FE | X8_lag1 | -12.4925 | 3.9306 | 0.0130 |
| FE | X9_lag1 | -0.0244 | 0.4469 | 0.9577 |
| FE | X10_lag1 | 1.6205 | 10.0106 | 0.8754 |
| FE | C1_lag1 | -16.8701 | 16.0615 | 0.3242 |
| FE | C2_lag1 | 31.5154 | 33.2761 | 0.3713 |

## Machine learning

- Best model: **svr**
- Test RMSE: **2.3488**
- Model id: `svr_20260928T160335Z`

### Model comparison

| model_type | status | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| random_forest | ok | 2.5632 | 1.9459 | 0.9529 |
| xgboost | ok | 2.9058 | 2.1832 | 0.9395 |
| lightgbm | ok | 3.9922 | 2.4163 | 0.8858 |
| catboost | ok | 2.9635 | 2.0528 | 0.9371 |
| svr | ok | 2.3488 | 1.8175 | 0.9605 |
| mlp | ok | 9.3269 | 7.8308 | 0.3768 |
| linear_regression | ok | 2.6135 | 2.3594 | 0.9511 |
| ridge | ok | 2.8009 | 2.5591 | 0.9438 |

## Comparative group metrics

| group_id | n | mean_dmi | std_dmi | min_dmi | max_dmi | start_year | end_year | mean_dmi_start | mean_dmi_end | abs_growth | pct_growth | cagr | n_obs | n_countries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balkan | 42 | 51.1753 | 11.3776 | 31.7685 | 71.1423 | 2012 | 2025 | 43.2807 | 61.6913 | 18.4106 | 0.4254 | 0.0276 | 42 | 3 |
| developed_eu | 42 | 76.6179 | 4.3482 | 67.9047 | 84.2780 | 2012 | 2025 | 71.8068 | 80.4646 | 8.6579 | 0.1206 | 0.0088 | 42 | 3 |
| developed_non_eu | 42 | 72.8765 | 4.5423 | 64.3359 | 82.5287 | 2012 | 2025 | 67.0864 | 76.9448 | 9.8583 | 0.1469 | 0.0106 | 42 | 3 |
| developing | 42 | 41.6769 | 12.8079 | 14.6186 | 57.9381 | 2012 | 2025 | 30.0282 | 55.5418 | 25.5136 | 0.8497 | 0.0484 | 42 | 3 |

## Convergence

- Source: `results\convergence\sigma_convergence.csv`
- Sigma start → end: **19.7096 → 3.9774**
- Converging: **True**

## Result registry

- Registered results: **31**
