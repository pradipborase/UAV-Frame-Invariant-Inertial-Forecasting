# Figure captions (manuscript-facing)

## Figure 1. Protocol schematic
Locked measurement and evaluation protocol: invariant scalars, causal 100 Hz streams, source-only fit, within-domain LORO, and zero-shot transfer.

## Figure 2. Primary RMSE
Equal-sequence mean multi-horizon RMSE of $q_f$ (m/s^2) for Persistence, the source-selected B3 Ridge comparator, TCN, GRU, and Transformer. Error bars are 95% percentile intervals from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8).

## Figure 3. Transfer gap
Transfer gap = zero-shot target RMSE − corresponding within-target RMSE for the source-selected B3 Ridge comparator and the three advanced models. Persistence is omitted because it has no trained source model.

## Figure 4. Horizon RMSE
Equal-sequence RMSE versus forecast horizon for all 20 steps from 10 ms to 200 ms.

## Figure 5. Advanced versus Ridge
Per-flight paired $\Delta$ = RMSE_advanced − RMSE_B3 Ridge (m/s^2), with the equal-sequence mean and 95% sequence-bootstrap CI shown to the right of each panel. Negative values mean the advanced model has lower sequence RMSE than source-selected B3 Ridge.
