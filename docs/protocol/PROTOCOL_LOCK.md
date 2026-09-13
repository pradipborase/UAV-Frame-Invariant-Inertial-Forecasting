# PROTOCOL_LOCK.md — Stage 1 locked protocol

Created utc: 2026-09-12T17:35:07Z

This is a **PRE-SPECIFIED LOCKED PROTOCOL**.

It is not an external OSF/Zenodo registration. Do not describe this file as pre-registered.

Git commit hash, if any, is recorded in `results/stage1/PROTOCOL_LOCK_HASHES.txt` after these bytes are hashed, to avoid a self-referential hash.

## Datasets

- D1 = EuRoC MAV (6 Vicon primary recordings)
- D2 = UZH-FPV Snapdragon (8 primary recordings)

## Primary sequences

EuRoC: V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult

UZH-FPV: indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon

## Primary physical quantity

Accelerometer specific-force magnitude

q_f(t) = sqrt(f_x(t)^2 + f_y(t)^2 + f_z(t)^2)

unit: m/s^2

## Auxiliary / input channel

Angular-speed magnitude

q_w(t) = sqrt(w_x(t)^2 + w_y(t)^2 + w_z(t)^2)

unit: rad/s

## Primary future forecast target

q_f(t)

Primary input channels: q_f(t), q_w(t)

Secondary candidate target q_w(t) is NOT in the confirmatory experiment.

## Common sampling rate

100 Hz after causal anti-alias filtering and integer decimation.

## Filter requirements

Causal elliptic IIR SOS. Passband 30 Hz, stopband 45 Hz, passband ripple <= 1 dB, stopband >= 60 dB. Same physical edges for both native rates. Warmup 1.0 s. FIT_SCOPE = NONE_DETERMINISTIC_PHYSICAL.

## Windowing

- lookback 1.0 s = 100 samples
- horizon 0.20 s = 20 samples
- stride 0.05 s = 5 samples
- report 50 / 100 / 200 ms (h = 5, 10, 20) plus h = 1..20 curve

## Inferential unit

SEQUENCE / flight / recording. Not window.

## Folds

- Within-EuRoC: 6 leave-one-recording-out folds
- Within-UZH: 8 leave-one-recording-out folds
- Source selection: EuRoC leave-room-out; UZH leave-trajectory-group-out (outdoor_forward is a single-recording group)
- Transfer: EUROC -> UZH and UZH -> EUROC, reported separately

## Future normalization

Any statistical scaler: SOURCE TRAIN ONLY. Never target mean/std/min/max.

## Primary metric

Equal-sequence mean of sequence-level multi-horizon RMSE of q_f over h=1..20, reported in m/s^2 after any inverse scaling.

## Secondary metrics

MAE (m/s^2); RMSE at 50/100/200 ms; RMSE-versus-horizon; persistence-relative skill. Window-weighted RMSE is descriptive only.

## Future persistence definition (not executed in Stage 1)

qhat_f(t+h) = q_f(t) for h=1..20.

## Bootstrap

10,000 sequence-level bootstrap replicates, seed 20260912, percentile CIs.

## Zero-shot rules

Target data are not used to fit model parameters, scalers, coordinate alignment, or hyperparameters.

Manuscript wording: "zero-shot with respect to model fitting and data-driven adaptation" — not "the target dataset was completely unseen during study design".

## Forbidden target adaptations

target-test mean/std or min/max; target-fitted PCA; target-fitted rotation; target-fitted bias; target-fitted time warp; post-test retuning.

## Primary scientific question

To what extent can short-horizon prediction of accelerometer specific-force magnitude generalize across UAV platforms, sensor hardware, flight regimes, and datasets when the representation is rotation invariant and all fitted transformations are restricted to the source domain?
