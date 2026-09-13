# Analysis hierarchy (frozen)

## PRIMARY
- Sequence-level multi-horizon q_f RMSE
- Four evaluation contexts: WITHIN_EUROC, WITHIN_UZH, EUROC_TO_UZH, UZH_TO_EUROC
- Persistence / Best frozen Ridge / TCN / GRU / Transformer
- RMSE at 50, 100, and 200 ms
- Zero-shot transfer results
- Paired advanced-versus-Ridge differences (Delta = RMSE_advanced - RMSE_Ridge)

## SECONDARY
- q_w ablation (authorized Stage-3 ablation only)
- High-specific-force target intervals (sequence-specific p95, unchanged)
- Horizon curves h=1..20
- Transfer gap (descriptive)
- Computational footprint
- Seed variability (not additional inferential units)
- Domain-shift characterization

## SUPPLEMENTARY
- Linear extrapolation baseline (B1)
- Full hyperparameter grids
- Per-sequence RMSE table
- Model-selection / coefficient details
- Filter and lock audits
- All Stage-0/1 manifests

Do not reclassify after manuscript writing begins.
