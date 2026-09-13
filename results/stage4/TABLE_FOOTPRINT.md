# Table 5. Computational footprint

| Source | Model | Config | Parameters | Mean train s | Median inference ms/window | Artifact bytes |
|---|---|---|---|---|---|---|
| EUROC | TCN | TCN12 | 16212 | 3.1 | 0.055 | 70005 |
| EUROC | GRU | GRU07 | 14356 | 17.2 | 0.147 | 61321 |
| EUROC | TRANSFORMER | TR12 | 13460 | 23.4 | 0.136 | 60973 |
| UZH_FPV | TCN | TCN07 | 26516 | 9.6 | 0.082 | 112589 |
| UZH_FPV | GRU | GRU07 | 14356 | 11.4 | 0.148 | 61537 |
| UZH_FPV | TRANSFORMER | TR05 | 13460 | 27.5 | 0.214 | 61209 |

CPU environment only. Not a real-time onboard claim.
