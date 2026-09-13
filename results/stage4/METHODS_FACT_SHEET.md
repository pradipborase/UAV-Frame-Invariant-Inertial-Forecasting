# Methods fact sheet

## Datasets and sequences
EuRoC: V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult
UZH-FPV: indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon

## Physical quantities
f: 3-axis accelerometer specific force, native IMU frame, m/s^2.
ω: 3-axis angular velocity, native IMU frame, rad/s.
q_f = ||f||_2; q_w = ||ω||_2.

## Sampling and filtering
Native rates ~200 Hz (EuRoC) and ~500 Hz (UZH). Causal elliptic SOS anti-alias, pass 30 Hz / stop 45 Hz, ≤1 dB / ≥60 dB, then integer decimation to 100 Hz. Causal forward SOS only; zero-phase bidirectional filtering is forbidden. Warmup 1.0 s discarded.

## Windows
Lookback 100 samples (1.0 s); horizon 20 samples (0.20 s); stride 5 samples (0.05 s). Direct 20-step head. Origins shared across models.

## Splits
Within-domain leave-one-recording-out. Nested group-aware source validation (EuRoC rooms; UZH trajectory groups). Transfer: all source sequences, frozen, evaluate all target sequences.

## Normalization
Equal-recording-weighted standard scaler on source training recordings only. Stage-2 Ridge: features only. Stage-3 networks: features and targets in source-normalized space; metrics inverse-transformed to m/s^2.

## Models
M0 Persistence: last q_f held for 20 steps.
M1 Best frozen Ridge: see RIDGE_COMPARATOR_MAP.csv (B3 except UZH→EuRoC B2).
M2–M4 compact TCN/GRU/Transformer, ≤12 source-only configs, seeds 20260912/13/14, AdamW, batch 64, max 100 epochs, patience 10 on source validation.

## Metrics and statistics
Primary: equal-sequence mean multi-horizon RMSE (m/s^2). Also MAE; RMSE at h=5,10,20 (50/100/200 ms). Sequence bootstrap 10,000, seed 20260912. Paired Δ = RMSE_advanced − RMSE_Ridge. Sign-flip p supplementary only.

## Zero-shot rule
No target data in parameters, hyperparameters, scaler, alignment, early stopping, or architecture choice.

## Reproducibility hashes
See LOCK_VERIFICATION.csv and REPRODUCIBILITY_MANIFEST.csv.
