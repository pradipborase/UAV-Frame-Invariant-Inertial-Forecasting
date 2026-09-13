# Supplementary files

Manuscript-facing names below are generated in `publication_current/` from **frozen** Stage-1/2/3 files (no retraining, no invented numbers). Copies with the same names are also written here when `scripts/build_publication_current.py` is run.

B3 is the primary Ridge comparator in both zero-shot transfer directions because it was selected exclusively by source-domain validation. B2 is retained only as a supplementary sensitivity baseline.

| Requested name | How it is produced |
| --- | --- |
| `Supplementary_PerFlight_Primary_RMSE.csv` | Recomputed per-sequence RMSE for Persistence, source-selected B3 Ridge, TCN, GRU, Transformer |
| `Supplementary_FlightLevel_Paired_Differences.csv` | Per-flight Δ = RMSE_advanced − RMSE_B3 Ridge |
| `Supplementary_Paired_Tests.csv` | Sequence-bootstrap CIs, win/loss/tie, exploratory sign-flip p vs B3 |
| `Supplementary_Baseline_and_Ridge_Sensitivity.csv` | Frozen `results/stage2/BASELINE_SUMMARY.csv` plus a `role` label (B2 = supplementary sensitivity only) |
| `Supplementary_Resampling_Audit.csv` | Publication-facing copy with corrected filter-order metadata (EuRoC order 5; UZH-FPV order 6); historical original remains under `results/stage1/` and `audit_original/` |
| `Supplementary_Ridge_Hyperparameter_Selection.csv` | Copy of frozen `results/stage2/HYPERPARAMETER_SELECTION.csv` |
| `Supplementary_Selected_Advanced_Models.csv` | Copy of frozen `results/stage3/SELECTED_MODELS.csv` |

Historical copies with the original study filenames remain in this folder (`SUPPLEMENT_SEQUENCE_RESULTS.csv`, `FIG4_paired_delta.csv`, `RIDGE_COMPARATOR_MAP.csv`, …). Those historical Ridge tables still reflect the pre-correction Stage-4 audit (UZH→EuRoC B2). Use `publication_current/` for manuscript numbers.

Full frozen trees remain under `../results/stage1`, `../results/stage2`, `../results/stage3`, and historical `../results/stage4`.
