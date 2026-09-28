# Results summary

_Generated: 2026-09-28T16:11:11.600311+00:00_

This report is generated from pipeline artifacts under `results/` and `outputs/`.

## Analysis panel

- Rows: **168**
- Countries: **12**
- Years: **2012–2025**
- Mean DMI: **60.88**

## DMI by group

| n | mean | std | variance | skewness | kurtosis | min | max | shapiro_stat | shapiro_pvalue | shapiro_applied | shapiro_note | group_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 51.8651 | 11.2497 | 126.5559 | -0.1365 | -0.9966 | 31.7034 | 70.8554 | 0.9642 | 0.2091 | True | — | balkan |
| 42 | 76.8074 | 4.6176 | 21.3221 | -0.2588 | -1.1119 | 68.2154 | 83.9979 | 0.9492 | 0.0604 | True | — | developed_eu |
| 42 | 72.2502 | 4.1468 | 17.1957 | 0.6504 | 0.0469 | 64.7279 | 81.7590 | 0.9560 | 0.1063 | True | — | developed_non_eu |
| 42 | 42.5889 | 12.7328 | 162.1237 | -0.7895 | 0.0466 | 14.2392 | 60.1777 | 0.9244 | 0.0084 | True | — | developing |

## Econometrics

- Formula: `DMI ~ X1_lag1 + X2_lag1 + X3_lag1 + X4_lag1 + X5_lag1 + X6_lag1 + X7_lag1 + X8_lag1 + X9_lag1 + X10_lag1 + C1_lag1 + C2_lag1`
- N observations: **57**
- Preferred model (Hausman): **FE**

### FE coefficients (excerpt)

| model | term | coefficient | std_error | pvalue |
| --- | --- | --- | --- | --- |
| FE | const | -67.4249 | 127.1773 | 0.5995 |
| FE | X1_lag1 | 0.1153 | 0.1044 | 0.2775 |
| FE | X2_lag1 | 0.3551 | 0.4406 | 0.4260 |
| FE | X3_lag1 | -32.4667 | 22.6548 | 0.1612 |
| FE | X4_lag1 | 0.5315 | 0.1350 | 0.0004 |
| FE | X5_lag1 | 0.0225 | 0.0345 | 0.5192 |
| FE | X6_lag1 | -0.1730 | 0.2480 | 0.4903 |
| FE | X7_lag1 | 0.2864 | 0.0988 | 0.0066 |
| FE | X8_lag1 | -3.4713 | 5.6193 | 0.5410 |
| FE | X9_lag1 | 0.0091 | 0.1204 | 0.9404 |
| FE | X10_lag1 | 35.8963 | 9.0534 | 0.0004 |
| FE | C1_lag1 | 0.2007 | 12.0075 | 0.9868 |
| FE | C2_lag1 | 60.9724 | 86.4096 | 0.4854 |

## Machine learning

- Best model: **ridge**
- Test RMSE: **2.0235**
- Model id: `ridge_20260928T161026Z`

### Model comparison

| model_type | status | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| random_forest | ok | 2.7543 | 2.3450 | 0.9383 |
| xgboost | ok | 3.1321 | 2.4910 | 0.9202 |
| lightgbm | ok | 3.3464 | 2.5156 | 0.9089 |
| catboost | ok | 2.8459 | 2.1077 | 0.9341 |
| svr | ok | 3.1327 | 2.5293 | 0.9202 |
| mlp | ok | 16.2059 | 12.6379 | -1.1362 |
| linear_regression | ok | 2.2277 | 1.8712 | 0.9596 |
| ridge | ok | 2.0235 | 1.7433 | 0.9667 |

## Comparative group metrics

| group_id | n | mean_dmi | std_dmi | min_dmi | max_dmi | start_year | end_year | mean_dmi_start | mean_dmi_end | abs_growth | pct_growth | cagr | n_obs | n_countries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balkan | 42 | 51.8651 | 11.2497 | 31.7034 | 70.8554 | 2012 | 2025 | 42.2605 | 63.1397 | 20.8792 | 0.4941 | 0.0314 | 42 | 3 |
| developed_eu | 42 | 76.8074 | 4.6176 | 68.2154 | 83.9979 | 2012 | 2025 | 71.7824 | 80.6402 | 8.8578 | 0.1234 | 0.0090 | 42 | 3 |
| developed_non_eu | 42 | 72.2502 | 4.1468 | 64.7279 | 81.7590 | 2012 | 2025 | 67.7481 | 76.6958 | 8.9477 | 0.1321 | 0.0096 | 42 | 3 |
| developing | 42 | 42.5889 | 12.7328 | 14.2392 | 60.1777 | 2012 | 2025 | 30.0131 | 56.2892 | 26.2761 | 0.8755 | 0.0496 | 42 | 3 |

## Convergence

- Source: `results\convergence\sigma_convergence.csv`
- Sigma start → end: **19.9525 → 4.8059**
- Converging: **True**

## Result registry

- Registered results: **31**
