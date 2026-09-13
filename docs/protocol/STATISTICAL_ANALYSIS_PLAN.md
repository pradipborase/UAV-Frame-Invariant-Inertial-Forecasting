# STATISTICAL_ANALYSIS_PLAN.md

Locked at Stage 1. No hypothesis tests are executed in Stage 1.

## Inferential unit

**PRIMARY INFERENTIAL UNIT = FLIGHT / RECORDING / SEQUENCE**

Windows are not independent experimental units. Thousands of overlapping 1.0 s histories do not create thousands of independent samples.

## Primary outcome

Sequence-level multi-horizon RMSE of accelerometer specific-force magnitude q_f over horizons h = 1,...,20 at 100 Hz, in m/s^2.

RMSE_s = sqrt( mean_{origins, h=1..20} (qhat_f - q_f)^2 )

## Dataset summary (primary estimand)

Equal-sequence mean RMSE: each recording has weight 1/n_sequences.

Not pooled-window RMSE.

## Uncertainty

Sequence-level bootstrap, 10,000 replicates, seed 20260912, percentile confidence intervals.

Because n = 6 (EuRoC) or n = 8 (UZH) is small, p-values will not be overstated.

## Future paired comparison versus persistence

Delta_s = RMSE_candidate,s - RMSE_persistence,s

Report mean paired difference, median paired difference, bootstrap CI, and win/loss/tie counts.

An exact paired sign-flip/permutation test may be used for a single pre-specified primary model. Not computed in Stage 1.

## Two transfer directions

EUROC -> UZH and UZH -> EUROC are separate estimands. Do not pool them. If both are tested formally, use Holm on those two primary comparisons. Not computed in Stage 1.

## Secondary metrics

MAE (m/s^2); RMSE at 50, 100, 200 ms; full RMSE-versus-horizon; persistence skill = 1 - RMSE_model / RMSE_persistence.

Window-weighted descriptive RMSE may be reported only as secondary and must be labelled as such.

## Future model selection

Source-domain validation only. The target dataset must not influence family or hyperparameter choice. After source-only selection, refit on authorized source training recordings and evaluate once, frozen, on the target. No post-test retuning.

## Future persistence (definition only)

qhat_f(t+h) = q_f(t) for h = 1,...,20. Not executed in Stage 1.

## Primary scientific question (frozen)

To what extent can short-horizon prediction of accelerometer specific-force magnitude generalize across UAV platforms, sensor hardware, flight regimes, and datasets when the representation is rotation invariant and all fitted transformations are restricted to the source domain?

Secondary questions only:

- RQ2: within-dataset vs cross-dataset zero-shot performance
- RQ3: error vs horizon 10–200 ms
- RQ4: consistency across recordings and flight-condition groups
