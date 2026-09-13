# STAGE 1 REPORT

Primary representation:
frame-invariant accelerometer specific-force magnitude q_f = ||f||_2 and angular-speed magnitude q_w = ||omega||_2

Primary target:
q_f (m/s^2)

Auxiliary input:
q_w (rad/s)

Reason for invariant representation:
EuRoC and UZH share the physical inertial quantity and SI units, but native sensor axes are not demonstrated to be the same vehicle directions. UZH body CAD is unresolved. A fitted cross-dataset rotation is prohibited.

EuRoC sequences retained:
6 (V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult)

UZH sequences retained:
8 (indoor_forward_6/9/10_snapdragon, indoor_45_2/4/13/14_snapdragon, outdoor_forward_1_snapdragon)

Total recordings:
14

Native rates:
EuRoC = 200 Hz documented (measured median 200.00256 Hz)
UZH = 500 Hz documented (measured ~500 Hz)

Common rate:
100 Hz

Filter:
causal elliptic IIR SOS, order 6; passband 30 Hz; stopband 45 Hz; ripple <= 1 dB; stopband >= 60 dB; designed at 200 Hz and 500 Hz

Causal:
YES

Future leakage test:
see pytest test_preprocessing_causality.py

Warmup removed:
1.0 s physical time (same for both datasets)

q_f descriptive range:
0.169159 to 166.609 m/s^2 (native, all 14 recordings)

q_w descriptive range:
0 to 20.2092 rad/s (native, all 14 recordings)

Near-constant sequences:
none

Processed-integrity:
14/14 PASS

Lookback:
1.0 s / 100 samples

Forecast horizon:
0.20 s / 20 samples

Evaluation stride:
0.05 s / 5 samples

Report horizons:
50 / 100 / 200 ms

Within-EuRoC folds:
6 leave-one-recording-out

Within-UZH folds:
8 leave-one-recording-out

Source-selection groups:
EuRoC Vicon Room 1 vs Room 2; UZH indoor_forward, indoor_45, outdoor_forward (outdoor flagged as n=1)

Cross-dataset directions:
EUROC -> UZH
UZH -> EUROC

Inferential unit:
SEQUENCE

Primary future metric:
equal-sequence multi-horizon RMSE in m/s^2

Future zero-shot normalization:
SOURCE-TRAIN-ONLY

Protocol lock SHA-256:
b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed

Compatibility clarification:
STAGE0C_REPORT integrity Major=0 is distinct from 4 MAJOR compatibility-matrix rows; see results/stage1/STAGE0C_COMPATIBILITY_CLARIFICATION.md

Critical findings:
0

Major findings:
0

pytest:
58 passed
0 failed

OVERALL STAGE-1 DECISION:
PASS

NO FORECASTING MODEL WAS TRAINED.
NO PREDICTIONS WERE GENERATED.
NO MODEL PERFORMANCE WAS INSPECTED.
