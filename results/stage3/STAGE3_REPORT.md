# STAGE 3 REPORT

Protocol verified: PASS
PROTOCOL_LOCK.md sha256: b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed
configs/protocol_lock.yaml sha256: 8fd75b59e35aac291f597fb856ff1debfc9609cf7a52adf6fc0e3502bf3c0d7d
Stage-2 SOURCE_MODELS_FROZEN.md sha256: 0aba0b20ddc30136de783b9f98a1a475915ee3ef20577e03efec8be1c15bb892
HYPERPARAMETER_GRID.csv sha256 (pre-training): c5105e181542ad6afa8457acb598709ce82603c15aae75800756098da2eb123f
configs/stage3_advanced.yaml sha256: a8453474ea26a552f8414408e91456da78ebf63b4b88b90dbd950540f72e9700

Models:
TCN
GRU
Transformer

Input:
Primary `[q_f, q_w]` history of 100 samples at 100 Hz. Direct 20-step `q_f` head. No recursive rollout.
No derivatives, moving averages, spectral features, GT, dataset, or environment labels.

Target:
`q_f(t) = ||f(t)||_2` (m/s^2), accelerometer specific-force magnitude.

Parameter counts:
- TCN TCN01 n_in=2: 4308 params
- TCN TCN02 n_in=2: 4308 params
- TCN TCN03 n_in=2: 6868 params
- TCN TCN04 n_in=2: 6868 params
- TCN TCN05 n_in=2: 16276 params
- TCN TCN06 n_in=2: 16276 params
- TCN TCN07 n_in=2: 26516 params
- TCN TCN08 n_in=2: 26516 params
- TCN TCN09 n_in=2: 3524 params
- TCN TCN10 n_in=2: 13172 params
- TCN TCN11 n_in=2: 5572 params
- TCN TCN12 n_in=2: 16212 params
- GRU GRU01 n_in=2: 1300 params
- GRU GRU02 n_in=2: 1300 params
- GRU GRU03 n_in=2: 2932 params
- GRU GRU04 n_in=2: 4116 params
- GRU GRU05 n_in=2: 4116 params
- GRU GRU06 n_in=2: 10452 params
- GRU GRU07 n_in=2: 14356 params
- GRU GRU08 n_in=2: 14356 params
- GRU GRU09 n_in=2: 39316 params
- GRU GRU10 n_in=2: 4116 params
- GRU GRU11 n_in=2: 2932 params
- GRU GRU12 n_in=2: 10452 params
- TRANSFORMER TR01 n_in=2: 2612 params
- TRANSFORMER TR02 n_in=2: 3668 params
- TRANSFORMER TR03 n_in=2: 4836 params
- TRANSFORMER TR04 n_in=2: 9300 params
- TRANSFORMER TR05 n_in=2: 13460 params
- TRANSFORMER TR06 n_in=2: 17844 params
- TRANSFORMER TR07 n_in=2: 3668 params
- TRANSFORMER TR08 n_in=2: 17844 params
- TRANSFORMER TR09 n_in=2: 2612 params
- TRANSFORMER TR10 n_in=2: 9300 params
- TRANSFORMER TR11 n_in=2: 6948 params
- TRANSFORMER TR12 n_in=2: 13460 params

Selected EuRoC configurations:
- TCN: TCN12 (epochs=23, val RMSE=0.7443, params=16212)
- GRU: GRU07 (epochs=49, val RMSE=0.7300, params=14356)
- TRANSFORMER: TR12 (epochs=66, val RMSE=0.7423, params=13460)

Selected UZH configurations:
- TCN: TCN07 (epochs=28, val RMSE=1.9179, params=26516)
- GRU: GRU07 (epochs=27, val RMSE=1.9323, params=14356)
- TRANSFORMER: TR05 (epochs=33, val RMSE=1.9079, params=13460)

WITHIN EUROC:
- Persistence: mean RMSE 0.9170 (CI 0.7430–1.0694); 50/100/200 ms = 0.7515/0.9114/1.0782
- Best Ridge: mean RMSE 0.7283 (CI 0.5826–0.8506); 50/100/200 ms = 0.6429/0.7532/0.8166
- TCN: mean RMSE 0.7351 (CI 0.5937–0.8541); 50/100/200 ms = 0.6409/0.7348/0.8149
- GRU: mean RMSE 0.7316 (CI 0.5913–0.8472); 50/100/200 ms = 0.6254/0.7211/0.8120
- Transformer: mean RMSE 0.7439 (CI 0.6032–0.8553); 50/100/200 ms = 0.6456/0.7305/0.8182

WITHIN UZH:
- Persistence: mean RMSE 2.3493 (CI 1.6376–3.0531); 50/100/200 ms = 2.0383/2.5136/2.6545
- Best Ridge: mean RMSE 1.9431 (CI 1.3777–2.5266); 50/100/200 ms = 1.8332/2.0947/2.1489
- TCN: mean RMSE 1.9201 (CI 1.3622–2.5059); 50/100/200 ms = 1.8234/2.0269/2.1105
- GRU: mean RMSE 1.9458 (CI 1.3510–2.5724); 50/100/200 ms = 1.8376/2.0109/2.1247
- Transformer: mean RMSE 1.8952 (CI 1.3448–2.4792); 50/100/200 ms = 1.7844/2.0154/2.1080

EUROC -> UZH:
- Persistence: mean RMSE 2.3493 (CI 1.6376–3.0531); 50/100/200 ms = 2.0383/2.5136/2.6545
- Best Ridge: mean RMSE 2.0651 (CI 1.4839–2.6980); 50/100/200 ms = 1.8444/2.2055/2.3698
- TCN: mean RMSE 2.0499 (CI 1.4706–2.6901); 50/100/200 ms = 1.9451/2.1281/2.2912
- GRU: mean RMSE 2.0800 (CI 1.4705–2.7580); 50/100/200 ms = 1.9603/2.0932/2.2683
- Transformer: mean RMSE 2.1976 (CI 1.5899–2.9056); 50/100/200 ms = 2.0543/2.1730/2.3902

UZH -> EUROC:
- Persistence: mean RMSE 0.9170 (CI 0.7430–1.0694); 50/100/200 ms = 0.7515/0.9114/1.0782
- Best Ridge: mean RMSE 0.8153 (CI 0.6622–0.9324); 50/100/200 ms = 0.7177/0.8159/0.8910
- TCN: mean RMSE 0.8254 (CI 0.6938–0.9316); 50/100/200 ms = 0.6951/0.8117/0.8827
- GRU: mean RMSE 0.8047 (CI 0.6654–0.9172); 50/100/200 ms = 0.6934/0.7981/0.8968
- Transformer: mean RMSE 0.8169 (CI 0.6760–0.9350); 50/100/200 ms = 0.7068/0.8113/0.9025

Best advanced model: TCN

Improvement over Ridge:
- WITHIN_EUROC TCN: mean Δ=0.0069 (median 0.0091), CI [-0.0051, 0.0157], wins/losses/ties=1/5/0
- WITHIN_EUROC GRU: mean Δ=0.0033 (median 0.0096), CI [-0.0098, 0.0138], wins/losses/ties=2/4/0
- WITHIN_EUROC TRANSFORMER: mean Δ=0.0157 (median 0.0237), CI [-0.0053, 0.0321], wins/losses/ties=1/5/0
- WITHIN_UZH TCN: mean Δ=-0.0230 (median -0.0198), CI [-0.0469, 0.0027], wins/losses/ties=6/2/0
- WITHIN_UZH GRU: mean Δ=0.0027 (median -0.0118), CI [-0.0386, 0.0494], wins/losses/ties=5/3/0
- WITHIN_UZH TRANSFORMER: mean Δ=-0.0479 (median -0.0398), CI [-0.0726, -0.0223], wins/losses/ties=7/1/0
- EUROC_TO_UZH TCN: mean Δ=-0.0152 (median -0.0066), CI [-0.0440, 0.0110], wins/losses/ties=6/2/0
- EUROC_TO_UZH GRU: mean Δ=0.0149 (median -0.0106), CI [-0.0451, 0.0818], wins/losses/ties=4/4/0
- EUROC_TO_UZH TRANSFORMER: mean Δ=0.1324 (median 0.0963), CI [0.0232, 0.2540], wins/losses/ties=1/7/0
- UZH_TO_EUROC TCN: mean Δ=0.0101 (median -0.0037), CI [-0.0099, 0.0366], wins/losses/ties=3/3/0
- UZH_TO_EUROC GRU: mean Δ=-0.0106 (median -0.0197), CI [-0.0327, 0.0123], wins/losses/ties=4/2/0
- UZH_TO_EUROC TRANSFORMER: mean Δ=0.0016 (median 0.0044), CI [-0.0195, 0.0195], wins/losses/ties=2/4/0

50 ms:
- WITHIN_EUROC Persistence: 0.7515
- WITHIN_EUROC Best Ridge: 0.6429
- WITHIN_EUROC TCN: 0.6409
- WITHIN_EUROC GRU: 0.6254
- WITHIN_EUROC Transformer: 0.6456
- WITHIN_UZH Persistence: 2.0383
- WITHIN_UZH Best Ridge: 1.8332
- WITHIN_UZH TCN: 1.8234
- WITHIN_UZH GRU: 1.8376
- WITHIN_UZH Transformer: 1.7844
- EUROC_TO_UZH Persistence: 2.0383
- EUROC_TO_UZH Best Ridge: 1.8444
- EUROC_TO_UZH TCN: 1.9451
- EUROC_TO_UZH GRU: 1.9603
- EUROC_TO_UZH Transformer: 2.0543
- UZH_TO_EUROC Persistence: 0.7515
- UZH_TO_EUROC Best Ridge: 0.7177
- UZH_TO_EUROC TCN: 0.6951
- UZH_TO_EUROC GRU: 0.6934
- UZH_TO_EUROC Transformer: 0.7068

100 ms:
- WITHIN_EUROC Persistence: 0.9114
- WITHIN_EUROC Best Ridge: 0.7532
- WITHIN_EUROC TCN: 0.7348
- WITHIN_EUROC GRU: 0.7211
- WITHIN_EUROC Transformer: 0.7305
- WITHIN_UZH Persistence: 2.5136
- WITHIN_UZH Best Ridge: 2.0947
- WITHIN_UZH TCN: 2.0269
- WITHIN_UZH GRU: 2.0109
- WITHIN_UZH Transformer: 2.0154
- EUROC_TO_UZH Persistence: 2.5136
- EUROC_TO_UZH Best Ridge: 2.2055
- EUROC_TO_UZH TCN: 2.1281
- EUROC_TO_UZH GRU: 2.0932
- EUROC_TO_UZH Transformer: 2.1730
- UZH_TO_EUROC Persistence: 0.9114
- UZH_TO_EUROC Best Ridge: 0.8159
- UZH_TO_EUROC TCN: 0.8117
- UZH_TO_EUROC GRU: 0.7981
- UZH_TO_EUROC Transformer: 0.8113

200 ms:
- WITHIN_EUROC Persistence: 1.0782
- WITHIN_EUROC Best Ridge: 0.8166
- WITHIN_EUROC TCN: 0.8149
- WITHIN_EUROC GRU: 0.8120
- WITHIN_EUROC Transformer: 0.8182
- WITHIN_UZH Persistence: 2.6545
- WITHIN_UZH Best Ridge: 2.1489
- WITHIN_UZH TCN: 2.1105
- WITHIN_UZH GRU: 2.1247
- WITHIN_UZH Transformer: 2.1080
- EUROC_TO_UZH Persistence: 2.6545
- EUROC_TO_UZH Best Ridge: 2.3698
- EUROC_TO_UZH TCN: 2.2912
- EUROC_TO_UZH GRU: 2.2683
- EUROC_TO_UZH Transformer: 2.3902
- UZH_TO_EUROC Persistence: 1.0782
- UZH_TO_EUROC Best Ridge: 0.8910
- UZH_TO_EUROC TCN: 0.8827
- UZH_TO_EUROC GRU: 0.8968
- UZH_TO_EUROC Transformer: 0.9025

q_w ablation:
- EUROC GRU q_f: source-val RMSE 0.7313
- EUROC GRU q_f+q_w: source-val RMSE 0.7300
- UZH_FPV TRANSFORMER q_f: source-val RMSE 1.9204
- UZH_FPV TRANSFORMER q_f+q_w: source-val RMSE 1.9079

Seed variability:
- WITHIN_EUROC TCN: mean=0.7351 sd=0.0028 min=0.7324 max=0.7381
- WITHIN_EUROC GRU: mean=0.7316 sd=0.0006 min=0.7309 max=0.7320
- WITHIN_EUROC TRANSFORMER: mean=0.7439 sd=0.0055 min=0.7396 max=0.7501
- WITHIN_UZH TCN: mean=1.9201 sd=0.0222 min=1.9015 max=1.9447
- WITHIN_UZH GRU: mean=1.9458 sd=0.0030 min=1.9428 max=1.9488
- WITHIN_UZH TRANSFORMER: mean=1.8952 sd=0.0041 min=1.8911 max=1.8994
- EUROC_TO_UZH TCN: mean=2.0499 sd=0.0112 min=2.0404 max=2.0623
- EUROC_TO_UZH GRU: mean=2.0800 sd=0.0045 min=2.0751 max=2.0838
- EUROC_TO_UZH TRANSFORMER: mean=2.1976 sd=0.0181 min=2.1844 max=2.2183
- UZH_TO_EUROC TCN: mean=0.8254 sd=0.0086 min=0.8172 max=0.8343
- UZH_TO_EUROC GRU: mean=0.8047 sd=0.0075 min=0.7971 max=0.8122
- UZH_TO_EUROC TRANSFORMER: mean=0.8169 sd=0.0103 min=0.8059 max=0.8262

High-specific-force results:
- WITHIN_EUROC Persistence: mean high-q_f RMSE 1.8691
- WITHIN_EUROC Best Ridge: mean high-q_f RMSE 1.8722
- WITHIN_EUROC TCN: mean high-q_f RMSE 1.8715
- WITHIN_EUROC GRU: mean high-q_f RMSE 1.8716
- WITHIN_EUROC Transformer: mean high-q_f RMSE 1.8665
- WITHIN_UZH Persistence: mean high-q_f RMSE 7.9670
- WITHIN_UZH Best Ridge: mean high-q_f RMSE 7.0885
- WITHIN_UZH TCN: mean high-q_f RMSE 6.9263
- WITHIN_UZH GRU: mean high-q_f RMSE 7.3423
- WITHIN_UZH Transformer: mean high-q_f RMSE 7.0530
- EUROC_TO_UZH Persistence: mean high-q_f RMSE 7.9670
- EUROC_TO_UZH Best Ridge: mean high-q_f RMSE 7.7001
- EUROC_TO_UZH TCN: mean high-q_f RMSE 7.6700
- EUROC_TO_UZH GRU: mean high-q_f RMSE 8.0521
- EUROC_TO_UZH Transformer: mean high-q_f RMSE 8.2399
- UZH_TO_EUROC Persistence: mean high-q_f RMSE 1.8691
- UZH_TO_EUROC Best Ridge: mean high-q_f RMSE 1.9897
- UZH_TO_EUROC TCN: mean high-q_f RMSE 1.9837
- UZH_TO_EUROC GRU: mean high-q_f RMSE 1.9591
- UZH_TO_EUROC Transformer: mean high-q_f RMSE 1.8726

Transfer gaps:
- EUROC_TO_UZH BEST_RIDGE: gap=0.1220 (6.3% of within-target)
- UZH_TO_EUROC BEST_RIDGE: gap=0.0871 (12.0% of within-target)
- EUROC_TO_UZH TCN: gap=0.1298 (6.8% of within-target)
- UZH_TO_EUROC TCN: gap=0.0903 (12.3% of within-target)
- EUROC_TO_UZH GRU: gap=0.1342 (6.9% of within-target)
- UZH_TO_EUROC GRU: gap=0.0731 (10.0% of within-target)
- EUROC_TO_UZH TRANSFORMER: gap=0.3024 (16.0% of within-target)
- UZH_TO_EUROC TRANSFORMER: gap=0.0729 (9.8% of within-target)

Computational cost:
- EUROC TCN TCN12: 16212 params, train 3.1s, median inference 0.0549 ms/window, artifact 70005 bytes (computational footprint)
- EUROC GRU GRU07: 14356 params, train 17.2s, median inference 0.1468 ms/window, artifact 61321 bytes (computational footprint)
- EUROC TRANSFORMER TR12: 13460 params, train 23.4s, median inference 0.1359 ms/window, artifact 60973 bytes (computational footprint)
- UZH_FPV TCN TCN07: 26516 params, train 9.6s, median inference 0.0824 ms/window, artifact 112589 bytes (computational footprint)
- UZH_FPV GRU GRU07: 14356 params, train 11.4s, median inference 0.1484 ms/window, artifact 61537 bytes (computational footprint)
- UZH_FPV TRANSFORMER TR05: 13460 params, train 27.5s, median inference 0.2140 ms/window, artifact 61209 bytes (computational footprint)

Target-fitted objects: 0

Source freeze hash: f4be16be714f2e6ad6d9ec4628fe7346dbf2294713604b4c158f4771ac072bb0

Freeze preserved: PASS

Critical findings:
- None. Target-fitted objects = 0.

Major findings:
- Scientific classification: ADVANCED_MODEL_DOMAIN_SPECIFIC_GAIN
- Best advanced model by mean Δ vs Ridge across four contexts: TCN
- Sequence remains the inferential unit; three seeds are not extra experimental samples.
- Principal Ridge comparator is the frozen Stage-2 winner in each context (B3 except UZH→EuRoC, which uses B2).

pytest:
90 passed, 0 failed

STAGE-3 SCIENTIFIC CLASSIFICATION: ADVANCED_MODEL_DOMAIN_SPECIFIC_GAIN

RECOMMENDED PAPER DIRECTION: OPTION 3: Complexity-generalization trade-off in cross-UAV telemetry forecasting

OVERALL STAGE-3 DECISION: PASS

Environment:
- Python 3.12.6
- numpy 2.5.3, torch 2.14.0+cpu
- platform Windows-11-10.0.26200-SP0
- processor AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
- CUDA: not used (CPU torch 2.14.0+cpu)
- Determinism: Python/NumPy/PyTorch RNGs seeded. CPU kernels are not claimed bitwise deterministic across hardware.

This stage does not write a manuscript.
