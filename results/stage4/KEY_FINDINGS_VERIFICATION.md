# KEY FINDINGS VERIFICATION

Each item is checked against recomputed sequence-level RMSE.

## A. Within EuRoC: Ridge approximately as good as advanced models
Status: SUPPORTED
Ridge 0.7283; TCN 0.7351; GRU 0.7316; Transformer 0.7439 m/s^2.
All three advanced models have higher mean RMSE than Ridge; paired CIs overlap zero; wins are a minority.

## B. Within UZH: Transformer provides the clearest advanced-model gain over Ridge
Status: SUPPORTED
Ridge 1.9431; Transformer 1.8952; mean Δ=-0.0479; wins 7/8; CI [-0.0726, -0.0223].
TCN is a smaller, CI-overlapping improvement; GRU is mixed.

## C. EuRoC -> UZH: TCN approximately competitive with Ridge; Transformer materially worse
Status: SUPPORTED
Ridge 2.0651; TCN 2.0499 (Δ=-0.0152, CI overlaps 0); Transformer 2.1976 (Δ=0.1324, 1–7).

## D. UZH -> EuRoC: GRU nominally best, improvement over Ridge modest
Status: SUPPORTED
Ridge 0.8153; GRU 0.8047; mean Δ=-0.0106; wins 4/6; CI includes 0.

## E. No advanced model dominates all four contexts
Status: SUPPORTED
Context-wise lowest mean RMSE among Ridge/TCN/GRU/Transformer:
- WITHIN_EUROC: Best frozen Ridge
- WITHIN_UZH: Transformer
- EUROC_TO_UZH: TCN
- UZH_TO_EUROC: GRU

Working conclusion check: compact nonlinear models can provide dataset-specific within-domain gains (UZH Transformer), but increased complexity does not consistently improve zero-shot transfer (EuRoC Transformer degrades); regularized Ridge remains competitive. Status: SUPPORTED.
