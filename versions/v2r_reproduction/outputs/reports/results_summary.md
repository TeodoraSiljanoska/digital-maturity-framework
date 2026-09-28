# Results summary

_Generated: 2026-09-28T16:02:34.636559+00:00_

This report is generated from pipeline artifacts under `results/` and `outputs/`.

## Analysis panel

- Rows: **168**
- Countries: **12**
- Years: **2012–2025**
- Mean DMI: **60.45**

## DMI by group

| n | mean | std | variance | skewness | kurtosis | min | max | shapiro_stat | shapiro_pvalue | shapiro_applied | shapiro_note | group_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 51.9485 | 10.9883 | 120.7435 | -0.1267 | -0.8981 | 31.6978 | 70.8554 | 0.9682 | 0.2876 | True | — | balkan |
| 42 | 76.2874 | 4.3944 | 19.3103 | -0.0625 | -0.9836 | 67.9272 | 83.9979 | 0.9674 | 0.2698 | True | — | developed_eu |
| 42 | 71.0960 | 4.5102 | 20.3422 | 0.8045 | 0.1422 | 63.4192 | 81.7590 | 0.9336 | 0.0171 | True | — | developed_non_eu |
| 42 | 42.4796 | 11.8159 | 139.6165 | -0.5663 | -0.1894 | 17.2434 | 60.1777 | 0.9447 | 0.0419 | True | — | developing |

## Econometrics

- Formula: `DMI ~ X1_lag1 + X2_lag1 + X3_lag1 + X4_lag1 + X5_lag1 + X6_lag1 + X7_lag1 + X8_lag1 + X9_lag1 + X10_lag1 + C1_lag1 + C2_lag1`
- N observations: **156**
- Preferred model (Hausman): **FE**

### FE coefficients (excerpt)

| model | term | coefficient | std_error | pvalue | group_id |
| --- | --- | --- | --- | --- | --- |
| FE | const | -162.1792 | 33.6702 | 0.0000 | — |
| FE | X1_lag1 | 0.0519 | 0.0342 | 0.1315 | — |
| FE | X2_lag1 | 0.3225 | 0.0922 | 0.0006 | — |
| FE | X3_lag1 | 16.9128 | 7.7185 | 0.0302 | — |
| FE | X4_lag1 | 0.2192 | 0.0493 | 0.0000 | — |
| FE | X5_lag1 | 0.0546 | 0.0198 | 0.0066 | — |
| FE | X6_lag1 | -0.0584 | 0.1549 | 0.7068 | — |
| FE | X7_lag1 | 0.1284 | 0.0263 | 0.0000 | — |
| FE | X8_lag1 | 0.4663 | 1.2360 | 0.7066 | — |
| FE | X9_lag1 | 0.1165 | 0.0651 | 0.0757 | — |
| FE | X10_lag1 | 1.9765 | 3.1323 | 0.5291 | — |
| FE | C1_lag1 | 10.8046 | 3.4003 | 0.0018 | — |
| FE | C2_lag1 | 62.2187 | 21.0773 | 0.0037 | — |

## Machine learning

- Best model: **xgboost**
- Test RMSE: **2.1213**
- Model id: `xgboost_20260928T155109Z`

### Model comparison

| model_type | status | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| random_forest | ok | 3.3716 | 2.9151 | 0.9202 |
| xgboost | ok | 2.1213 | 1.7310 | 0.9684 |
| lightgbm | ok | 3.4646 | 2.4872 | 0.9157 |
| catboost | ok | 2.2128 | 1.6808 | 0.9656 |
| svr | ok | 2.5192 | 1.8714 | 0.9554 |
| mlp | ok | 18.0954 | 13.7687 | -1.2991 |
| linear_regression | ok | 3.0578 | 2.6107 | 0.9343 |
| ridge | ok | 2.6585 | 2.1800 | 0.9504 |

## Comparative group metrics

| group_id | n | mean_dmi | std_dmi | min_dmi | max_dmi | start_year | end_year | mean_dmi_start | mean_dmi_end | abs_growth | pct_growth | cagr | n_obs | n_countries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balkan | 42 | 51.9485 | 10.9883 | 31.6978 | 70.8554 | 2012 | 2025 | 42.7277 | 63.1397 | 20.4120 | 0.4777 | 0.0305 | 42 | 3 |
| developed_eu | 42 | 76.2874 | 4.3944 | 67.9272 | 83.9979 | 2012 | 2025 | 70.4825 | 80.6402 | 10.1577 | 0.1441 | 0.0104 | 42 | 3 |
| developed_non_eu | 42 | 71.0960 | 4.5102 | 63.4192 | 81.7590 | 2012 | 2025 | 65.5584 | 77.0500 | 11.4916 | 0.1753 | 0.0125 | 42 | 3 |
| developing | 42 | 42.4796 | 11.8159 | 17.2434 | 60.1777 | 2012 | 2025 | 31.5624 | 54.5497 | 22.9873 | 0.7283 | 0.0430 | 42 | 3 |

## Convergence

- Source: `results\convergence\sigma_convergence.csv`
- Sigma start → end: **18.3609 → 7.7609**
- Converging: **True**

## Result registry

- Registered results: **38**
