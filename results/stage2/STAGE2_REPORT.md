# STAGE 2 REPORT

Protocol lock verified:
PASS (`PROTOCOL_LOCK.md`=b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed; yaml=8fd75b59e35aac291f597fb856ff1debfc9609cf7a52adf6fc0e3502bf3c0d7d)

Extreme-value audit:
all 14 sequences KEEP; no clipping; see results/stage2/EXTREME_VALUE_AUDIT.csv
native Stage-1 maximum 166.609 m/s^2 remains a retained observation (indoor_forward_6_snapdragon)

Sequences used:
EuRoC = 6
UZH = 8

Models:
B0 Persistence
B1 Linear extrapolation
B2 Ridge q_f
B3 Ridge q_f+q_w

Selected EuRoC source hyperparameters:
B2 lag=25 alpha=0.0001
B3 lag=25 alpha=0.0001

Selected UZH source hyperparameters:
B2 lag=25 alpha=1.0
B3 lag=10 alpha=0.01

Within EuRoC:
- B0_PERSISTENCE: mean seq RMSE=0.9170 m/s^2 (95% CI 0.7430–1.0694); 50/100/200 ms=0.7515/0.9114/1.0782; skill=n/a
- B1_LINEAR: mean seq RMSE=1.6187 m/s^2 (95% CI 1.2177–2.0414); 50/100/200 ms=0.9842/1.4821/2.4158; skill=-0.7301
- B2_RIDGE_QF: mean seq RMSE=0.7315 m/s^2 (95% CI 0.5867–0.8528); 50/100/200 ms=0.6456/0.7578/0.8205; skill=0.2039
- B3_RIDGE_QF_QW: mean seq RMSE=0.7283 m/s^2 (95% CI 0.5826–0.8506); 50/100/200 ms=0.6429/0.7532/0.8166; skill=0.2078

Within UZH:
- B0_PERSISTENCE: mean seq RMSE=2.3493 m/s^2 (95% CI 1.6376–3.0531); 50/100/200 ms=2.0383/2.5136/2.6545; skill=n/a
- B1_LINEAR: mean seq RMSE=5.1674 m/s^2 (95% CI 3.3528–6.8994); 50/100/200 ms=3.1512/4.7669/7.5539; skill=-1.0933
- B2_RIDGE_QF: mean seq RMSE=1.9808 m/s^2 (95% CI 1.4285–2.5688); 50/100/200 ms=1.8737/2.1295/2.1956; skill=0.1215
- B3_RIDGE_QF_QW: mean seq RMSE=1.9431 m/s^2 (95% CI 1.3777–2.5266); 50/100/200 ms=1.8332/2.0947/2.1489; skill=0.1512

EuRoC -> UZH:
- B0_PERSISTENCE: mean seq RMSE=2.3493 m/s^2 (95% CI 1.6376–3.0531); 50/100/200 ms=2.0383/2.5136/2.6545; skill=n/a
- B1_LINEAR: mean seq RMSE=5.1674 m/s^2 (95% CI 3.3528–6.8994); 50/100/200 ms=3.1512/4.7669/7.5539; skill=-1.0933
- B2_RIDGE_QF: mean seq RMSE=2.1061 m/s^2 (95% CI 1.5238–2.7460); 50/100/200 ms=1.8664/2.2409/2.4150; skill=0.0587
- B3_RIDGE_QF_QW: mean seq RMSE=2.0651 m/s^2 (95% CI 1.4839–2.6980); 50/100/200 ms=1.8444/2.2055/2.3698; skill=0.0830

UZH -> EuRoC:
- B0_PERSISTENCE: mean seq RMSE=0.9170 m/s^2 (95% CI 0.7430–1.0694); 50/100/200 ms=0.7515/0.9114/1.0782; skill=n/a
- B1_LINEAR: mean seq RMSE=1.6187 m/s^2 (95% CI 1.2177–2.0414); 50/100/200 ms=0.9842/1.4821/2.4158; skill=-0.7301
- B2_RIDGE_QF: mean seq RMSE=0.8153 m/s^2 (95% CI 0.6622–0.9324); 50/100/200 ms=0.7177/0.8159/0.8910; skill=0.1079
- B3_RIDGE_QF_QW: mean seq RMSE=0.8308 m/s^2 (95% CI 0.7027–0.9361); 50/100/200 ms=0.6972/0.8347/0.9202; skill=0.0841

Best baseline by context:
- WITHIN_EUROC: B3_RIDGE_QF_QW (mean seq RMSE 0.7283 m/s^2)
- WITHIN_UZH: B3_RIDGE_QF_QW (mean seq RMSE 1.9431 m/s^2)
- EUROC_TO_UZH: B3_RIDGE_QF_QW (mean seq RMSE 2.0651 m/s^2)
- UZH_TO_EUROC: B2_RIDGE_QF (mean seq RMSE 0.8153 m/s^2)

Persistence difficulty:
median Persistence_RMSE / SD(q_f) = 1.0664; median ACF at 200 ms = 0.2408

Effect of q_w input:
WITHIN_EUROC: B3−B2 = -0.0033 m/s^2; WITHIN_UZH: B3−B2 = -0.0376 m/s^2; EUROC_TO_UZH: B3−B2 = -0.0410 m/s^2; UZH_TO_EUROC: B3−B2 = +0.0155 m/s^2

50 ms results:
WITHIN_EUROC Persistence/Linear/Ridge_qf/Ridge_both = 0.7515 / 0.9842 / 0.6456 / 0.6429
WITHIN_UZH = 2.0383 / 3.1512 / 1.8737 / 1.8332

100 ms results:
WITHIN_EUROC = 0.9114 / 1.4821 / 0.7578 / 0.7532
WITHIN_UZH = 2.5136 / 4.7669 / 2.1295 / 2.0947

200 ms results:
WITHIN_EUROC = 1.0782 / 2.4158 / 0.8205 / 0.8166
WITHIN_UZH = 2.6545 / 7.5539 / 2.1956 / 2.1489

High-specific-force secondary analysis:
secondary RMSE on targets above sequence p95 is tabulated in HIGH_QF_SECONDARY.csv; training was not altered.

Sequence-level uncertainty:
10,000 sequence-level bootstrap percentile CIs in BASELINE_SUMMARY.csv and BOOTSTRAP_SUMMARY.csv. n=6/8; p-values are exploratory only.

Transfer degradation:
B0_PERSISTENCE: (EUROC→UZH minus within-UZH)=+0.0000; (UZH→EUROC minus within-EuRoC)=+0.0000 B2_RIDGE_QF: (EUROC→UZH minus within-UZH)=+0.1254; (UZH→EUROC minus within-EuRoC)=+0.0838 B3_RIDGE_QF_QW: (EUROC→UZH minus within-UZH)=+0.1220; (UZH→EUROC minus within-EuRoC)=+0.1025

Target-domain fitted objects:
0

Source-model freeze hash:
0aba0b20ddc30136de783b9f98a1a475915ee3ef20577e03efec8be1c15bb892

Transfer freeze verification:
PASS

Critical findings:
0
none

Major findings:
0

pytest:
75 passed
0 failed

ADVANCED-MODEL GATE:
ADVANCED_MODELS_JUSTIFIED

At least one learned baseline shows nontrivial within-domain skill, remaining error is non-negligible, and zero-shot transfer is materially harder.

OVERALL STAGE-2 DECISION:
PASS
