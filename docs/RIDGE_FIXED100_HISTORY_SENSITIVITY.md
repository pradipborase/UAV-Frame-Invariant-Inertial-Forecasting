# Fixed 100-lag Ridge B3 equal-history sensitivity

This is a **post hoc supplementary equal-history sensitivity**. It does **not** replace the primary source-selected-lag Ridge comparator (`B3_RIDGE_QF_QW` with lag 25 on EuRoC and lag 10 on UZH-FPV).

## Why it exists

Neural models (TCN, GRU, Transformer) use the full 100-sample / 1.0 s candidate history. Primary Ridge is allowed to use that same window but selects a shorter lag from `[10, 25, 50, 100]` on source validation only. Reviewers asked what happens if Ridge is forced to the same 100-sample history.

## Protocol (frozen; no neural retrain)

- Model: `B3_RIDGE_QF_QW` (`[q_f, q_w]`)
- Lag **fixed at 100** samples
- Tune **only alpha** from the original grid `[1e-6, 1e-4, 1e-2, 1, 100]`
- Within-domain LORO: alpha chosen using only the training/source recordings of that fold
- Zero-shot transfer: alpha chosen on the source dataset only; scaler and Ridge then frozen before target evaluation
- Same scaler family, equal-recording weights, origins, 20-step horizon, sequence-level RMSE, and 10,000 sequence-level bootstrap (seed 20260912)
- **No TCN, GRU, or Transformer was retrained**
- **No target-domain tuning**
- **Primary manuscript Stage-2/3 result files were not changed**

## Source-only selected alpha with lag = 100

- EuRoC source: alpha = 1e-4, source validation RMSE = 0.747376
- UZH-FPV source: alpha = 1, source validation RMSE = 1.999626

## Aggregate equal-sequence mean RMSE (m/s²)

| Context | Primary B3 RMSE | Fixed100 B3 RMSE | Delta (fixed100 − primary) |
| --- | ---: | ---: | ---: |
| Within EuRoC | 0.728256 | 0.748209 | +0.019953 |
| Within UZH-FPV | 1.943141 | 1.956741 | +0.013601 |
| EuRoC → UZH-FPV | 2.065150 | 2.125632 | +0.060482 |
| UZH-FPV → EuRoC | 0.830804 | 0.842334 | +0.011530 |

Forcing lag = 100 **increases** aggregate Ridge RMSE in **all four** contexts.

### 50 / 100 / 200 ms

- Within EuRoC — primary 0.642868 / 0.753203 / 0.816588; fixed100 0.668633 / 0.765633 / 0.823503
- Within UZH-FPV — primary 1.833155 / 2.094722 / 2.148914; fixed100 1.887984 / 2.066537 / 2.121225
- EuRoC → UZH-FPV — primary 1.844445 / 2.205469 / 2.369796; fixed100 1.881461 / 2.266810 / 2.445912
- UZH-FPV → EuRoC — primary 0.697231 / 0.834738 / 0.920151; fixed100 0.755811 / 0.854300 / 0.906304

## Interpretation that must not be overstated

- Within EuRoC, the fixed100 Ridge result modestly favors the nonlinear models relative to the primary tuned Ridge (because forced 100-lag Ridge is worse than source-selected-lag Ridge).
- That does **not** change the central conclusion: **no nonlinear architecture dominates all domains**.
- Machine-readable files: `results/sensitivity/RIDGE_B3_FIXED_LAG100_*.csv` (also copied into `publication_current/` and `supplementary/`).

To recompute this sensitivity from processed streams (not shipped): `python scripts/run_ridge_fixed_lag100_sensitivity.py`. That script does not train neural models and does not write into `results/stage2/` or `results/stage3/`.
