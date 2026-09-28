# Results summary

_Generated: 2026-08-11T13:23:46.867447+00:00_

This report is generated from pipeline artifacts under `results/` and `outputs/`.

## Analysis panel

- Rows: **168**
- Countries: **12**
- Years: **2012–2025**
- Mean DMI: **60.47**

## DMI by group

| n | mean | std | variance | skewness | kurtosis | min | max | shapiro_stat | shapiro_pvalue | shapiro_applied | shapiro_note | group_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 51.9586 | 10.9839 | 120.6460 | -0.1269 | -0.9003 | 31.6870 | 70.8554 | 0.9682 | 0.2870 | True | — | balkan |
| 42 | 76.3142 | 4.3792 | 19.1770 | -0.0609 | -0.9878 | 67.9863 | 83.9979 | 0.9673 | 0.2669 | True | — | developed_eu |
| 42 | 71.0965 | 4.5109 | 20.3478 | 0.8035 | 0.1411 | 63.4256 | 81.7590 | 0.9337 | 0.0173 | True | — | developed_non_eu |
| 42 | 42.4928 | 11.8132 | 139.5528 | -0.5694 | -0.1853 | 17.2520 | 60.1777 | 0.9445 | 0.0411 | True | — | developing |

## Econometrics

- Formula: `DMI ~ X1_lag1 + X2_lag1 + X3_lag1 + X4_lag1 + X5_lag1 + X6_lag1 + X7_lag1 + X8_lag1 + X9_lag1 + X10_lag1 + C1_lag1 + C2_lag1`
- N observations: **156**
- Preferred model (Hausman): **FE**

### FE coefficients (excerpt)

| model | term | coefficient | std_error | pvalue | group_id |
| --- | --- | --- | --- | --- | --- |
| FE | const | -161.9799 | 33.5864 | 0.0000 | — |
| FE | X1_lag1 | 0.0522 | 0.0340 | 0.1275 | — |
| FE | X2_lag1 | 0.3196 | 0.0914 | 0.0006 | — |
| FE | X3_lag1 | 16.9798 | 7.7220 | 0.0296 | — |
| FE | X4_lag1 | 0.2193 | 0.0493 | 0.0000 | — |
| FE | X5_lag1 | 0.0548 | 0.0196 | 0.0060 | — |
| FE | X6_lag1 | -0.0606 | 0.1537 | 0.6939 | — |
| FE | X7_lag1 | 0.1281 | 0.0265 | 0.0000 | — |
| FE | X8_lag1 | 0.4555 | 1.2384 | 0.7136 | — |
| FE | X9_lag1 | 0.1167 | 0.0648 | 0.0740 | — |
| FE | X10_lag1 | 1.9764 | 3.1284 | 0.5286 | — |
| FE | C1_lag1 | 10.8440 | 3.3838 | 0.0017 | — |
| FE | C2_lag1 | 61.6201 | 21.1389 | 0.0042 | — |

## Machine learning

- Best model: **catboost**
- Test RMSE: **1.9438**
- Model id: `catboost_20260811T132151Z`

### Model comparison

| model_type | status | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| random_forest | ok | 3.3920 | 2.9165 | 0.9192 |
| xgboost | ok | 2.3540 | 1.8675 | 0.9611 |
| lightgbm | ok | 3.6569 | 2.5845 | 0.9061 |
| catboost | ok | 1.9438 | 1.4558 | 0.9735 |
| svr | ok | 2.5125 | 1.8698 | 0.9557 |
| mlp | ok | 18.2150 | 13.8340 | -1.3297 |
| linear_regression | ok | 3.0793 | 2.6302 | 0.9334 |
| ridge | ok | 2.6984 | 2.2182 | 0.9489 |

## Comparative group metrics

| group_id | n | mean_dmi | std_dmi | min_dmi | max_dmi | start_year | end_year | mean_dmi_start | mean_dmi_end | abs_growth | pct_growth | cagr | n_obs | n_countries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balkan | 42 | 51.9586 | 10.9839 | 31.6870 | 70.8554 | 2012 | 2025 | 42.7744 | 63.1397 | 20.3653 | 0.4761 | 0.0304 | 42 | 3 |
| developed_eu | 42 | 76.3142 | 4.3792 | 67.9863 | 83.9979 | 2012 | 2025 | 70.5308 | 80.6402 | 10.1094 | 0.1433 | 0.0104 | 42 | 3 |
| developed_non_eu | 42 | 71.0965 | 4.5109 | 63.4256 | 81.7590 | 2012 | 2025 | 65.5635 | 77.0500 | 11.4866 | 0.1752 | 0.0125 | 42 | 3 |
| developing | 42 | 42.4928 | 11.8132 | 17.2520 | 60.1777 | 2012 | 2025 | 31.5996 | 54.5514 | 22.9518 | 0.7263 | 0.0429 | 42 | 3 |

## Convergence

- Source: `results/convergence/sigma_convergence.csv`
- Sigma start → end: **18.3605 → 7.7581**
- Converging: **True**

## Result registry

- Registered results: **44**
