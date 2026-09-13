# Input-history contract

Every forecast origin in this study has the same **candidate** window, horizon, and stride. That is not the same as saying every **model** uses every sample in that window.

## Shared window (all models)

At 100 Hz:

| Quantity | Samples | Physical |
| --- | ---: | --- |
| Candidate history | 100 | 1.0 s |
| Forecast horizon | 20 | 0.20 s |
| Origin stride | 5 | 0.05 s |

The candidate history is the lookback stored for each origin (`lookback_samples: 100` in `configs/stage2_baselines.yaml` and `configs/stage3_advanced.yaml`).

## Neural models (effective history = 100)

TCN, GRU, and Transformer each receive the full 100-sample `[q_f, q_w]` history at every origin. Their effective input history is therefore **100 samples / 1.0 s**.

## Ridge (effective history = source-selected lag)

Ridge is given access to the same 100-sample candidate history, but it uses only the **most recent `lag` samples**.

The original predeclared Ridge lag grid is:

```text
[10, 25, 50, 100]
```

Lag (and alpha) are chosen by **source-domain validation only**. Target-domain RMSE is never used to pick lag.

Primary source-selected B3 (`B3_RIDGE_QF_QW`, inputs `[q_f, q_w]`) therefore has:

- EuRoC source / within EuRoC / EuRoC→UZH: **lag = 25** (0.25 s), alpha = 1e-4
- UZH-FPV source / within UZH / UZH→EuRoC: **lag = 10** (0.10 s), alpha = 1e-2

A post hoc **fixed lag = 100** B3 sensitivity (same 1.0 s history as the neural models, alpha re-selected on source data only) is reported separately. It does **not** replace the primary source-selected-lag Ridge comparator. See `docs/RIDGE_FIXED100_HISTORY_SENSITIVITY.md`.

Do not write that “every model uses 100 samples” without this qualification.
