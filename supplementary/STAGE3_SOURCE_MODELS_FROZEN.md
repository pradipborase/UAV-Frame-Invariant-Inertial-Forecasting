# Source models frozen (Phase 3A)

utc: 2026-09-13T03:02:05Z
configs/stage3_advanced.yaml sha256: a8453474ea26a552f8414408e91456da78ebf63b4b88b90dbd950540f72e9700
results/stage3/HYPERPARAMETER_GRID.csv sha256: c5105e181542ad6afa8457acb598709ce82603c15aae75800756098da2eb123f

Do not modify this file after Phase 3B begins.
Hyperparameters below were selected with SOURCE-DOMAIN validation only.
Target datasets were not accessed during architecture choice, scaling, or early stopping.

Training seed policy: evaluate each selected configuration at seeds 20260912, 20260913, 20260914.
Hyperparameter selection used the mean source-validation equal-sequence RMSE over the three seeds.
Final source fits train on all source sequences for the median best source-validation epoch.
Minibatches sample a recording uniformly, then a valid window from that recording.
Early-stopping source-validation MSE uses a deterministic strided subset of up to 96 windows per validation sequence;
configuration ranking uses full-window equal-sequence RMSE in m/s^2.
Scaler: equal-recording-weighted standard scaler on features and targets, source-train windows only.
Reported metrics are inverse-transformed to m/s^2.

## EUROC

- source sequences: V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult
- validation methodology: leave-trajectory-group-out on source recordings
- scaler source: all source sequences of this dataset (final freeze fit)

### TCN  `TCN12`

- architecture: TCN
- input channels: q_f, q_w (n_in=2)
- channels: [32, 32]
- kernel_size: 5
- dilations: [1, 2, 4]
- activation: gelu
- dropout: 0.0
- lr: 0.001
- weight_decay: 0.0001
- selected_epochs: 23
- source-validation mean RMSE (m/s^2): 0.7442967233
- parameter count: 16212
- checkpoint pattern: artifacts/stage3/EUROC_TCN_seed{seed}.pt

### GRU  `GRU07`

- architecture: GRU
- input channels: q_f, q_w (n_in=2)
- hidden_size: 64
- n_layers: 1
- dropout: 0.0
- lr: 0.001
- weight_decay: 0.0001
- selected_epochs: 49
- source-validation mean RMSE (m/s^2): 0.7300144951
- parameter count: 14356
- checkpoint pattern: artifacts/stage3/EUROC_GRU_seed{seed}.pt

### TRANSFORMER  `TR12`

- architecture: TRANSFORMER
- input channels: q_f, q_w (n_in=2)
- n_layers: 1
- d_model: 32
- n_heads: 2
- ff_multiplier: 4
- pooling: mean
- dropout: 0.0
- lr: 0.001
- weight_decay: 0.0
- selected_epochs: 66
- source-validation mean RMSE (m/s^2): 0.7423329298
- parameter count: 13460
- checkpoint pattern: artifacts/stage3/EUROC_TRANSFORMER_seed{seed}.pt

## UZH_FPV

- source sequences: indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon
- validation methodology: leave-trajectory-group-out on source recordings
- scaler source: all source sequences of this dataset (final freeze fit)

### TCN  `TCN07`

- architecture: TCN
- input channels: q_f, q_w (n_in=2)
- channels: [32, 32]
- kernel_size: 5
- dilations: [1, 2, 4, 8, 16]
- activation: relu
- dropout: 0.0
- lr: 0.001
- weight_decay: 0.0
- selected_epochs: 28
- source-validation mean RMSE (m/s^2): 1.917872324
- parameter count: 26516
- checkpoint pattern: artifacts/stage3/UZH_FPV_TCN_seed{seed}.pt

### GRU  `GRU07`

- architecture: GRU
- input channels: q_f, q_w (n_in=2)
- hidden_size: 64
- n_layers: 1
- dropout: 0.0
- lr: 0.001
- weight_decay: 0.0001
- selected_epochs: 27
- source-validation mean RMSE (m/s^2): 1.932265437
- parameter count: 14356
- checkpoint pattern: artifacts/stage3/UZH_FPV_GRU_seed{seed}.pt

### TRANSFORMER  `TR05`

- architecture: TRANSFORMER
- input channels: q_f, q_w (n_in=2)
- n_layers: 1
- d_model: 32
- n_heads: 4
- ff_multiplier: 4
- pooling: last
- dropout: 0.1
- lr: 0.001
- weight_decay: 0.0001
- selected_epochs: 33
- source-validation mean RMSE (m/s^2): 1.907869478
- parameter count: 13460
- checkpoint pattern: artifacts/stage3/UZH_FPV_TRANSFORMER_seed{seed}.pt
