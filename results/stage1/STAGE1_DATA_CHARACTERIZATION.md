# Stage 1 data characterization (not model performance)

No forecast curves and no forecast errors are reported.

q_f is accelerometer specific-force magnitude (m/s^2), not linear acceleration.
q_w is angular-speed magnitude (rad/s).

## Native q_f by sequence

| dataset | sequence | median | mean | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| EUROC | V1_01_easy | 9.8081 | 9.80122 | 1.27729 | 4.01519 | 17.0061 |
| EUROC | V1_02_medium | 9.85668 | 9.88216 | 1.50507 | 4.33729 | 19.929 |
| EUROC | V1_03_difficult | 9.8502 | 9.94973 | 2.02475 | 2.30743 | 17.578 |
| EUROC | V2_01_easy | 9.79002 | 9.81285 | 1.45491 | 1.48215 | 35.2594 |
| EUROC | V2_02_medium | 9.86527 | 9.90559 | 1.497 | 3.03317 | 22.7818 |
| EUROC | V2_03_difficult | 9.86526 | 9.90624 | 1.53779 | 2.2693 | 32.8438 |
| UZH_FPV | indoor_forward_6_snapdragon | 10.3325 | 12.2668 | 5.70366 | 0.390304 | 166.609 |
| UZH_FPV | indoor_forward_9_snapdragon | 10.1156 | 11.2415 | 3.51177 | 0.528265 | 28.4465 |
| UZH_FPV | indoor_forward_10_snapdragon | 10.0637 | 11.2319 | 3.43711 | 0.292248 | 34.5478 |
| UZH_FPV | indoor_45_2_snapdragon | 10.0908 | 10.7056 | 3.64672 | 0.848841 | 104.934 |
| UZH_FPV | indoor_45_4_snapdragon | 9.91784 | 10.5561 | 3.93725 | 0.169159 | 140.893 |
| UZH_FPV | indoor_45_13_snapdragon | 9.97907 | 10.6412 | 3.33902 | 0.791836 | 130.507 |
| UZH_FPV | indoor_45_14_snapdragon | 10.3557 | 11.4193 | 3.97458 | 0.889273 | 100.966 |
| UZH_FPV | outdoor_forward_1_snapdragon | 9.93588 | 10.7367 | 3.62807 | 0.564562 | 102.953 |

## Native q_w by sequence

| dataset | sequence | median | mean | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| EUROC | V1_01_easy | 0.230523 | 0.282828 | 0.18327 | 0.00636028 | 0.938193 |
| EUROC | V1_02_medium | 0.511542 | 0.557611 | 0.345957 | 0.0120718 | 2.47344 |
| EUROC | V1_03_difficult | 0.575926 | 0.622771 | 0.416353 | 0.00508248 | 2.21076 |
| EUROC | V2_01_easy | 0.227558 | 0.282345 | 0.188313 | 0.00888577 | 1.88384 |
| EUROC | V2_02_medium | 0.541955 | 0.58557 | 0.377862 | 0.0120718 | 2.23644 |
| EUROC | V2_03_difficult | 0.628711 | 0.657136 | 0.398702 | 0.0020944 | 2.23327 |
| UZH_FPV | indoor_forward_6_snapdragon | 0.673973 | 0.983948 | 1.02521 | 0.00106526 | 11.0858 |
| UZH_FPV | indoor_forward_9_snapdragon | 0.53798 | 0.670432 | 0.580931 | 0.00106526 | 3.36143 |
| UZH_FPV | indoor_forward_10_snapdragon | 0.542815 | 0.665508 | 0.612131 | 0.00150651 | 3.33078 |
| UZH_FPV | indoor_45_2_snapdragon | 0.61386 | 0.659117 | 0.510892 | 0.00336866 | 5.20362 |
| UZH_FPV | indoor_45_4_snapdragon | 0.54092 | 0.59121 | 0.501655 | 0.002382 | 20.2092 |
| UZH_FPV | indoor_45_13_snapdragon | 0.576498 | 0.606171 | 0.49847 | 0.00106526 | 8.32054 |
| UZH_FPV | indoor_45_14_snapdragon | 0.737053 | 0.865557 | 0.730435 | 0 | 4.73721 |
| UZH_FPV | outdoor_forward_1_snapdragon | 0.345849 | 0.388722 | 0.34023 | 0.00150651 | 4.83647 |

## Processed duration and rate

| dataset | sequence | processed_n | duration_s | effective_hz | valid_windows |
|---|---|---:|---:|---:|---:|
| EUROC | V1_01_easy | 14460 | 144.59 | 99.99872001665304 | 2868 |
| EUROC | V1_02_medium | 8450 | 84.49 | 100.00128001629685 | 1666 |
| EUROC | V1_03_difficult | 10650 | 106.49 | 100.00128001629685 | 2106 |
| EUROC | V2_01_easy | 11300 | 112.99 | 100.00128001629685 | 2236 |
| EUROC | V2_02_medium | 11645 | 116.44 | 100.0000000000199 | 2305 |
| EUROC | V2_03_difficult | 11585 | 115.84 | 100.0000000000199 | 2293 |
| UZH_FPV | indoor_forward_6_snapdragon | 6863 | 68.62 | 100.00009536752259 | 1349 |
| UZH_FPV | indoor_forward_9_snapdragon | 7583 | 75.82 | 100.00009536752259 | 1493 |
| UZH_FPV | indoor_forward_10_snapdragon | 7527 | 75.26 | 100.00009536752259 | 1482 |
| UZH_FPV | indoor_45_2_snapdragon | 7368 | 73.67 | 100.00000000009095 | 1450 |
| UZH_FPV | indoor_45_4_snapdragon | 6523 | 65.22 | 100.00000000009095 | 1281 |
| UZH_FPV | indoor_45_13_snapdragon | 6094 | 60.93 | 100.00000000009095 | 1195 |
| UZH_FPV | indoor_45_14_snapdragon | 6354 | 63.53 | 100.00000000009095 | 1247 |
| UZH_FPV | outdoor_forward_1_snapdragon | 8476 | 84.75 | 100.00000000009095 | 1672 |

## Trajectory groups

- EuRoC V1_* : EUROC_FIREFLY_VICON_ROOM1 (easy/medium/difficult)
- EuRoC V2_* : EUROC_FIREFLY_VICON_ROOM2 (easy/medium/difficult)
- UZH indoor_forward, indoor_45, outdoor_forward (one retained outdoor recording)
