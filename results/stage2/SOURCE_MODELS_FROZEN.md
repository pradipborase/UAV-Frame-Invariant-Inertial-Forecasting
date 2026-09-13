# Source models frozen (Phase 2A)

utc: 2026-09-12T17:58:16Z

configs/stage2_baselines.yaml sha256: 0073b683fa0420810c8225be00fb649f11c3616949f07b4b97bae710700a4561

Do not modify this file after Phase 2B begins.
Hyperparameters below were selected with SOURCE-DOMAIN validation only.

## EuRoC selected B2 (Ridge q_f)

- lag: 25
- alpha: 0.0001
- equal-validation-sequence multi-horizon RMSE: 0.7467378132
- n_tied within 1% rule: 5

## EuRoC selected B3 (Ridge q_f+q_w)

- lag: 25
- alpha: 0.0001
- equal-validation-sequence multi-horizon RMSE: 0.7416754064
- n_tied within 1% rule: 4

## UZH selected B2 (Ridge q_f)

- lag: 25
- alpha: 1.0
- equal-validation-sequence multi-horizon RMSE: 2.015532414
- n_tied within 1% rule: 4

## UZH selected B3 (Ridge q_f+q_w)

- lag: 10
- alpha: 0.01
- equal-validation-sequence multi-horizon RMSE: 1.981207773
- n_tied within 1% rule: 3

## Scaler rules

- Feature scaler: equal-recording-weighted standard scaler
- Fitted on source training windows only
- Target scaler: NONE (errors remain in m/s^2 without inverse transform)
- FIT_SCOPE: SOURCE_TRAIN_ONLY
- Never fit on validation, within-domain test, or cross-domain target sequences

## Training weights

Each training window weight is 1/N_windows of its recording, then normalized so each recording has equal total weight.
