# Results fact sheet

## WITHIN_EUROC
- Persistence: 0.917 [0.743, 1.069] m/s^2; 50/100/200 ms = 0.751/0.911/1.078
- Best frozen Ridge: 0.728 [0.583, 0.851] m/s^2; 50/100/200 ms = 0.643/0.753/0.817
- TCN: 0.735 [0.594, 0.854] m/s^2; 50/100/200 ms = 0.641/0.735/0.815
- GRU: 0.732 [0.591, 0.847] m/s^2; 50/100/200 ms = 0.625/0.721/0.812
- Transformer: 0.744 [0.603, 0.855] m/s^2; 50/100/200 ms = 0.646/0.731/0.818

## WITHIN_UZH
- Persistence: 2.349 [1.638, 3.053] m/s^2; 50/100/200 ms = 2.038/2.514/2.655
- Best frozen Ridge: 1.943 [1.378, 2.527] m/s^2; 50/100/200 ms = 1.833/2.095/2.149
- TCN: 1.920 [1.362, 2.506] m/s^2; 50/100/200 ms = 1.823/2.027/2.111
- GRU: 1.946 [1.351, 2.572] m/s^2; 50/100/200 ms = 1.838/2.011/2.125
- Transformer: 1.895 [1.345, 2.479] m/s^2; 50/100/200 ms = 1.784/2.015/2.108

## EUROC_TO_UZH
- Persistence: 2.349 [1.638, 3.053] m/s^2; 50/100/200 ms = 2.038/2.514/2.655
- Best frozen Ridge: 2.065 [1.484, 2.698] m/s^2; 50/100/200 ms = 1.844/2.205/2.370
- TCN: 2.050 [1.471, 2.690] m/s^2; 50/100/200 ms = 1.945/2.128/2.291
- GRU: 2.080 [1.470, 2.758] m/s^2; 50/100/200 ms = 1.960/2.093/2.268
- Transformer: 2.198 [1.590, 2.906] m/s^2; 50/100/200 ms = 2.054/2.173/2.390

## UZH_TO_EUROC
- Persistence: 0.917 [0.743, 1.069] m/s^2; 50/100/200 ms = 0.751/0.911/1.078
- Best frozen Ridge: 0.815 [0.662, 0.932] m/s^2; 50/100/200 ms = 0.718/0.816/0.891
- TCN: 0.825 [0.694, 0.932] m/s^2; 50/100/200 ms = 0.695/0.812/0.883
- GRU: 0.805 [0.665, 0.917] m/s^2; 50/100/200 ms = 0.693/0.798/0.897
- Transformer: 0.817 [0.676, 0.935] m/s^2; 50/100/200 ms = 0.707/0.811/0.902

## High-specific-force (equal-sequence mean RMSE, secondary)
- WITHIN_EUROC Persistence: all 0.917; high 1.869; ratio 2.04
- WITHIN_EUROC Best frozen Ridge: all 0.728; high 1.872; ratio 2.57
- WITHIN_EUROC TCN: all 0.735; high 1.872; ratio 2.55
- WITHIN_EUROC GRU: all 0.732; high 1.872; ratio 2.56
- WITHIN_EUROC Transformer: all 0.744; high 1.866; ratio 2.51
- WITHIN_UZH Persistence: all 2.349; high 7.967; ratio 3.39
- WITHIN_UZH Best frozen Ridge: all 1.943; high 7.088; ratio 3.65
- WITHIN_UZH TCN: all 1.920; high 6.926; ratio 3.61
- WITHIN_UZH GRU: all 1.946; high 7.342; ratio 3.77
- WITHIN_UZH Transformer: all 1.895; high 7.053; ratio 3.72
- EUROC_TO_UZH Persistence: all 2.349; high 7.967; ratio 3.39
- EUROC_TO_UZH Best frozen Ridge: all 2.065; high 7.700; ratio 3.73
- EUROC_TO_UZH TCN: all 2.050; high 7.670; ratio 3.74
- EUROC_TO_UZH GRU: all 2.080; high 8.052; ratio 3.87
- EUROC_TO_UZH Transformer: all 2.198; high 8.240; ratio 3.75
- UZH_TO_EUROC Persistence: all 0.917; high 1.869; ratio 2.04
- UZH_TO_EUROC Best frozen Ridge: all 0.815; high 1.990; ratio 2.44
- UZH_TO_EUROC TCN: all 0.825; high 1.984; ratio 2.40
- UZH_TO_EUROC GRU: all 0.805; high 1.959; ratio 2.43
- UZH_TO_EUROC Transformer: all 0.817; high 1.873; ratio 2.29

## Transfer gap (learned models)
- EUROC_TO_UZH Best frozen Ridge: 0.122 m/s^2 (6.3% of within-target)
- UZH_TO_EUROC Best frozen Ridge: 0.087 m/s^2 (12.0% of within-target)
- EUROC_TO_UZH TCN: 0.130 m/s^2 (6.8% of within-target)
- UZH_TO_EUROC TCN: 0.090 m/s^2 (12.3% of within-target)
- EUROC_TO_UZH GRU: 0.134 m/s^2 (6.9% of within-target)
- UZH_TO_EUROC GRU: 0.073 m/s^2 (10.0% of within-target)
- EUROC_TO_UZH Transformer: 0.302 m/s^2 (16.0% of within-target)
- UZH_TO_EUROC Transformer: 0.073 m/s^2 (9.8% of within-target)

## q_w ablation
- EUROC GRU: q_f 0.731 vs q_f+q_w 0.730
- UZH_FPV TRANSFORMER: q_f 1.920 vs q_f+q_w 1.908

## Parameters / inference
- EUROC TCN TCN12: 16212 params, 0.055 ms/window median
- EUROC GRU GRU07: 14356 params, 0.147 ms/window median
- EUROC TRANSFORMER TR12: 13460 params, 0.136 ms/window median
- UZH_FPV TCN TCN07: 26516 params, 0.082 ms/window median
- UZH_FPV GRU GRU07: 14356 params, 0.148 ms/window median
- UZH_FPV TRANSFORMER TR05: 13460 params, 0.214 ms/window median
