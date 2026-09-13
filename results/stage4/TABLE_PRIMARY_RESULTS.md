# Table 3. Primary forecasting results

Equal-sequence mean multi-horizon RMSE of specific-force magnitude $q_f$ (m/s^2).
Intervals are 95% percentile CIs from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8).
Best frozen Ridge is context-specific (see Ridge comparator map): B3 $q_f{+}q_w$ except UZH$\rightarrow$EuRoC, which uses source-frozen B2 $q_f$.
$\Delta$ is not shown here; negative $\Delta$ in Table 4 means the advanced model has lower RMSE than Ridge.

| Model | Within EuRoC | Within UZH | EuRoC → UZH | UZH → EuRoC |
|---|---|---|---|---|
| Persistence | 0.917 [0.743, 1.069] | 2.349 [1.638, 3.053] | 2.349 [1.638, 3.053] | 0.917 [0.743, 1.069] |
| Best frozen Ridge | 0.728 [0.583, 0.851] | 1.943 [1.378, 2.527] | 2.065 [1.484, 2.698] | 0.815 [0.662, 0.932] |
| TCN | 0.735 [0.594, 0.854] | 1.920 [1.362, 2.506] | 2.050 [1.471, 2.690] | 0.825 [0.694, 0.932] |
| GRU | 0.732 [0.591, 0.847] | 1.946 [1.351, 2.572] | 2.080 [1.470, 2.758] | 0.805 [0.665, 0.917] |
| Transformer | 0.744 [0.603, 0.855] | 1.895 [1.345, 2.479] | 2.198 [1.590, 2.906] | 0.817 [0.676, 0.935] |

RMSE at 50 / 100 / 200 ms (equal-sequence means, m/s^2):

| Model | Context | 50 ms | 100 ms | 200 ms |
|---|---|---|---|---|
| Persistence | WITHIN_EUROC | 0.751 | 0.911 | 1.078 |
| Best frozen Ridge | WITHIN_EUROC | 0.643 | 0.753 | 0.817 |
| TCN | WITHIN_EUROC | 0.641 | 0.735 | 0.815 |
| GRU | WITHIN_EUROC | 0.625 | 0.721 | 0.812 |
| Transformer | WITHIN_EUROC | 0.646 | 0.731 | 0.818 |
| Persistence | WITHIN_UZH | 2.038 | 2.514 | 2.655 |
| Best frozen Ridge | WITHIN_UZH | 1.833 | 2.095 | 2.149 |
| TCN | WITHIN_UZH | 1.823 | 2.027 | 2.111 |
| GRU | WITHIN_UZH | 1.838 | 2.011 | 2.125 |
| Transformer | WITHIN_UZH | 1.784 | 2.015 | 2.108 |
| Persistence | EUROC_TO_UZH | 2.038 | 2.514 | 2.655 |
| Best frozen Ridge | EUROC_TO_UZH | 1.844 | 2.205 | 2.370 |
| TCN | EUROC_TO_UZH | 1.945 | 2.128 | 2.291 |
| GRU | EUROC_TO_UZH | 1.960 | 2.093 | 2.268 |
| Transformer | EUROC_TO_UZH | 2.054 | 2.173 | 2.390 |
| Persistence | UZH_TO_EUROC | 0.751 | 0.911 | 1.078 |
| Best frozen Ridge | UZH_TO_EUROC | 0.718 | 0.816 | 0.891 |
| TCN | UZH_TO_EUROC | 0.695 | 0.812 | 0.883 |
| GRU | UZH_TO_EUROC | 0.693 | 0.798 | 0.897 |
| Transformer | UZH_TO_EUROC | 0.707 | 0.811 | 0.902 |
