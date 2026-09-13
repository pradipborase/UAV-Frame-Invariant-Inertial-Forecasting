# Manuscript Evidence Package

## A. Study design in one paragraph
Two official UAV IMU datasets (EuRoC MAV, n=6; UZH-FPV Snapdragon, n=8) are compared on short-horizon forecasting of accelerometer specific-force magnitude after causal 100 Hz resampling. All fitted objects are source-only. Evaluation is leave-one-recording-out within domain and frozen zero-shot across datasets. The inferential unit is the flight recording.

## B. Measurement quantities
q_f = ||f||_2 (m/s^2); q_w = ||ω||_2 (rad/s). f is measured specific force, not linear acceleration of the origin.

## C. Dataset facts
EuRoC Firefly/ADIS16448 ~200 Hz; UZH Snapdragon ~500 Hz. Reference trajectories exist but are not q_f labels.

## D. Locked protocol
Lookback 1.0 s, horizon 0.20 s, stride 0.05 s, equal-sequence RMSE, 10,000 sequence bootstrap, seed 20260912.

## E. Primary results
See TABLE_PRIMARY_RESULTS.md. Ridge 0.728 m/s^2 within EuRoC; Transformer 1.895 vs Ridge 1.943 within UZH.

## F. Within-domain findings
EuRoC: Ridge matches or beats compact networks. UZH: Transformer shows a clear practical gain over Ridge (Δ=-0.048 m/s^2, 7–1, CI excludes 0).

## G. Zero-shot findings
Learned models beat Persistence without target adaptation. EuRoC→UZH Transformer degrades vs Ridge (Δ=0.132). TCN remains near Ridge. Transfer gaps are positive for all learned models.

## H. Complexity-generalization findings
Larger architectural family (Transformer) is not uniformly better under transfer. Compact Ridge remains competitive.

## I. High-specific-force findings
Difficulty ratios exceed 1 in all contexts and are largest on UZH (secondary).

## J. Auxiliary q_w finding
Modest source-validation RMSE change; do not claim major sensor fusion gains.

## K. Computational footprint
≤26516 parameters; sub-millisecond median inference per window on the reported CPU.

## L. Statistical limitations
n=6/8; CIs are sequence-level sampling uncertainty.

## M. Dataset/measurement limitations
Two platforms; unresolved UZH body frame; magnitude loses direction; sensor-observation target.

## N. Claims that ARE supported
C1–C10 safe wordings in CLAIM_EVIDENCE_MATRIX.csv.

## O. Claims that are NOT supported
Transformer/TCN universally best; DL significantly outperforms Ridge; domain shift eliminated; true physical acceleration; onboard real-time; all UAV platforms; major q_w fusion; preregistration; CIs as measurement uncertainty.

## P. Recommended manuscript emphasis
Frame-invariant cross-platform UAV inertial forecasting with emphasis on the complexity-generalization trade-off under zero-shot dataset transfer.
