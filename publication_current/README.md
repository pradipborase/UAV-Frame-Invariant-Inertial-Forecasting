# publication_current

Authoritative manuscript-facing tables and figures for this package.

B3 is the primary Ridge comparator in both zero-shot transfer directions because it was selected exclusively by source-domain validation. B2 is retained only as a supplementary sensitivity baseline.

These files are recomputed from frozen `results/stage2/` and `results/stage3/` CSVs. No model was retrained.
Ridge is always `B3_RIDGE_QF_QW`. Target-domain RMSE is never used to choose the comparator.

UZH_TO_EUROC primary Ridge mean RMSE = 0.830803772150
50/100/200 ms = 0.697230937167 / 0.834738248050 / 0.920150728817

Filter-order labels in `FILTER_SPECIFICATION_CORRECTED.md` and `RESAMPLING_AUDIT_CORRECTED.csv` are publication-facing: EuRoC/200 Hz = order 5, UZH-FPV/500 Hz = order 6. Historical Stage-1 files keep the original (incorrect) order-6 labels for both designs. SOS coefficients are unchanged.

Fixed 100-lag Ridge B3 CSVs here are a **post hoc supplementary** equal-history sensitivity. They do not replace the primary source-selected-lag B3 comparator.

Recommended filenames (`PRIMARY_RESULTS.csv`, `PERSISTENCE_SKILL.csv`, `TRANSFER_GAPS.csv`, `ADVANCED_VS_RIDGE.csv`, `HORIZON_RMSE_10_TO_200MS.csv`) are copies of the `TABLE_*` / `HORIZON_PUBLICATION_DATA.csv` files.

Historical Stage-4 outputs under `results/stage4/` retain the pre-correction (descriptive best-frozen Ridge, UZH→EuRoC B2) audit.
Figure 6 (high $q_f$) is not part of the current main manuscript and is not regenerated here.
