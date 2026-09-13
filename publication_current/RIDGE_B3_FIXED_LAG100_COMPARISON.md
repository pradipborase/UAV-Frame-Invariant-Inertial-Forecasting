# Fixed 100-lag Ridge B3 sensitivity

This is a **supplementary equal-history benchmark**. It is not a primary result.

## Protocol

- Model: Ridge B3 with inputs `[q_f, q_w]`.
- History length **fixed at lag = 100 samples (1.0 s)**, the same lookback used by TCN / GRU / Transformer.
- Only `alpha` is selected, from the original predeclared grid `[1e-6, 1e-4, 1e-2, 1, 100]`.
- Within-domain LORO: lag stays 100; alpha is chosen with source-only group-aware validation on the training recordings of that fold. The held-out sequence is excluded.
- Zero-shot transfer: alpha is chosen on the source dataset only, the Ridge and scaler are frozen, and the target is evaluated unchanged.
- Scaler, equal-recording weights, prediction origins, 20-step horizon, sequence-level RMSE, and 10,000 sequence-bootstrap CIs (seed 20260912) match the primary study.
- **No TCN, GRU, or Transformer was retrained.**
- **Target-domain data were not used to choose alpha or any other fitted object.**
- **Primary manuscript result files were not changed.** Comparison uses frozen Stage-2 `B3_RIDGE_QF_QW`.

Delta = RMSE_fixed100 − RMSE_primary_B3. Positive delta means the forced 100-lag history is worse.

## Source-only selected alpha (lag locked at 100)

| Context | Phase | Fold | Selected alpha | Source val RMSE |
|---|---|---|---|---|
| SOURCE_SELECTION_EUROC | source freeze | EUROC_SOURCE_ALL | 0.0001 | 0.747376 |
| SOURCE_SELECTION_UZH | source freeze | UZH_SOURCE_ALL | 1.0 | 1.999626 |

LORO fold alphas are in `RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv`.

## Equal-sequence mean RMSE (m/s^2)

| Context | Primary B3 RMSE [95% CI] | Fixed lag-100 B3 RMSE [95% CI] | Delta (fixed100 − primary) |
|---|---|---|---|
| Within EuRoC | 0.728256 [0.582642, 0.850571] | 0.748209 [0.606630, 0.863671] | +0.019953 (worse) |
| Within UZH-FPV | 1.943141 [1.377703, 2.526565] | 1.956741 [1.385041, 2.559978] | +0.013601 (worse) |
| EuRoC -> UZH-FPV | 2.065150 [1.483872, 2.698028] | 2.125632 [1.558536, 2.748705] | +0.060482 (worse) |
| UZH-FPV -> EuRoC | 0.830804 [0.702719, 0.936093] | 0.842334 [0.701287, 0.952415] | +0.011530 (worse) |

50 / 100 / 200 ms equal-sequence means (m/s^2):

| Context | Primary 50/100/200 | Fixed100 50/100/200 |
|---|---|---|
| Within EuRoC | 0.642868 / 0.753203 / 0.816588 | 0.668633 / 0.765633 / 0.823503 |
| Within UZH-FPV | 1.833155 / 2.094722 / 2.148914 | 1.887984 / 2.066537 / 2.121225 |
| EuRoC -> UZH-FPV | 1.844445 / 2.205469 / 2.369796 | 1.881461 / 2.266810 / 2.445912 |
| UZH-FPV -> EuRoC | 0.697231 / 0.834738 / 0.920151 | 0.755811 / 0.854300 / 0.906304 |

## Explicit statements

1. No neural model was retrained.
2. Target data were not used for tuning.
3. Primary manuscript results were not changed.
