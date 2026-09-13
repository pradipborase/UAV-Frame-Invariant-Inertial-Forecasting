# Table 2. Locked forecasting protocol

| Item | Setting |
|---|---|
| Representation | q_f = ||f||_2 (specific-force magnitude); q_w = ||ω||_2 |
| Primary target | future q_f (m/s^2) |
| Primary inputs | history of q_f and q_w |
| Sample rate | 100 Hz after causal elliptic anti-alias (30/45 Hz) and integer decimation |
| Lookback | 1.0 s (100 samples) |
| Horizon | 0.20 s (20 samples), direct multi-output |
| Stride | 0.05 s (5 samples) |
| Inferential unit | flight recording / sequence |
| Dataset estimand | equal-sequence mean multi-horizon RMSE |
| Transfer rule | freeze source objects; no target-fitted scaler, alignment, or selection |
| Normalization | equal-recording-weighted standard scaler, source-train recordings only |
