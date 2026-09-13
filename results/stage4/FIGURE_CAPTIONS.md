# Figure captions

## Figure 1. Study/protocol schematic
Locked evaluation path from two official UAV IMU datasets through rotation-invariant specific-force and angular-speed magnitudes, causal 100 Hz resampling, source-only fitting, within-domain leave-one-recording-out evaluation, and frozen zero-shot transfer. No numeric estimates are encoded.

## Figure 2. Primary RMSE comparison
Equal-sequence mean multi-horizon RMSE of q_f (m/s^2) for Persistence, context-specific best frozen Ridge, TCN, GRU, and Transformer. Error bars are 95% percentile intervals from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8). Values are not window-pooled.

## Figure 3. Forecast-horizon RMSE
Equal-sequence mean RMSE versus forecast horizon (10 ms steps to 200 ms) for the same models and four contexts. Curves are descriptive; horizons are not tested as 20 separate hypotheses.

## Figure 4. Advanced-model paired difference versus Ridge
Per-sequence Δ = RMSE_advanced − RMSE_Ridge (m/s^2). Negative values mean the advanced model has lower sequence RMSE than the context-specific frozen Ridge comparator. The horizontal line marks zero.

## Figure 5. Transfer gap
Descriptive transfer gap (m/s^2) = zero-shot target RMSE − corresponding within-target RMSE for Ridge, TCN, GRU, and Transformer. Persistence is omitted because it has no trained source-domain model. Positive values indicate higher error after transfer.

## Figure 6. High-specific-force difficulty
Equal-sequence RMSE on all forecast targets versus targets with future q_f above the sequence-specific 95th percentile of processed q_f. These are high-specific-force target intervals, not labelled extreme maneuvers.
