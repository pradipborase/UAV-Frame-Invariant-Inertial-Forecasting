# Novelty risk assessment

## What is methodological novelty?
Locked invariant scalars plus a two-phase freeze that forbids target-fitted normalization/alignment/selection.

## What is protocol novelty?
Equal-sequence evaluation, nested LORO/LGO, and an explicit zero-shot definition for inertial telemetry forecasting across official UAV datasets.

## What is empirical novelty?
Quantified complexity-generalization on EuRoC MAV vs UZH-FPV Snapdragon: compact Ridge transfers comparably to compact TCN/GRU; a compact Transformer helps within UZH and degrades EuRoC→UZH.

## What is merely good practice?
Reproducible hashes, causal resampling, not using windows as n, not claiming p<0.05 on n=6.

## Likely reviewer attacks
| concern | severity | existing defense | remaining vulnerability | manuscript action |
|---|---|---|---|---|
| only two datasets | HIGH | Official heterogeneous pair; honest scope | Cannot claim all UAVs | State two-dataset scope in title/abstract |
| small sequence counts | HIGH | Sequence CIs; avoid p-hacking | Wide intervals | Lead with effect sizes and wins/losses |
| scalar loses direction | HIGH | Frame audit MAJOR rows | Not a 3-axis estimator | Methods: why magnitude |
| sensor observation not independent truth | HIGH | Estimand is future q_f | Not "true acceleration" | Define telemetry forecasting |
| limited advanced-model gain | MEDIUM | Report Ridge competitive as a finding | "Why DL?" | Complexity-generalization framing |
| different platforms | MEDIUM | That is the transfer question | Confounded factors | Do not claim isolated sensor-shift |
| reference-trajectory differences | MEDIUM | Pose unused as labels | Residual confusion | Terminology: reference trajectory |
| no onboard experiment | HIGH | Footprint only | No deployment claim | Forbidden wording list |
| gravity magnitude | MEDIUM | Gravity characterization + Persistence | Near-g sequences exist | Report fractions and SD |
| no probabilistic forecast | MEDIUM | Out of scope | Not GUM | Explicit limitation |
