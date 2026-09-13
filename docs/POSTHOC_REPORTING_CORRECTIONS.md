# Post-hoc reporting corrections

This package contains two documentation corrections. Neither retrained a model, retuned a hyperparameter on target data, or regenerated processed IMU streams. Historical audit files that record the original pipeline are retained.

See also the individual notes:

- `docs/POSTHOC_REPORTING_CORRECTION.md` (Ridge comparator)
- `docs/POSTHOC_FILTER_ORDER_CORRECTION.md` (filter order)

## F1. Ridge comparator correction

Historical Stage-2/Stage-3 artifacts contained a descriptive best-frozen-Ridge comparison where B2 happened to give lower UZH→EuRoC **target** error (mean RMSE ≈ 0.815321).

Final primary reporting uses **B3_RIDGE_QF_QW** in all four contexts because B3 was selected exclusively by source-domain validation:

- EuRoC B3: `[q_f, q_w]`, lag = 25, alpha = 1e-4
- UZH-FPV B3: `[q_f, q_w]`, lag = 10, alpha = 1e-2

UZH-FPV → EuRoC **primary** source-selected B3 mean RMSE ≈ **0.830804**.

B2 may appear only as a clearly labelled supplementary sensitivity baseline. No model was retrained.

Manuscript-facing tables: `publication_current/` (including `RIDGE_COMPARATOR_MAP.csv`).  
Historical pre-correction Stage-4 audit: `results/stage4/` (UZH→EuRoC still records descriptive B2 there).

## F2. Filter-order metadata correction

Historical Stage-1 reports labelled **both** filters as order 6 because order was inferred as twice the number of SOS rows.

Final reporting:

- EuRoC / 200 Hz design = **order 5** (three SOS rows, one first-order denominator section with `a2 = 0`)
- UZH-FPV / 500 Hz design = **order 6**

The Stage-1 filtering calculations and processed data were correct, but the filter order was misreported because metadata inferred order as twice the number of SOS rows. The 200-Hz EuRoC design is fifth order and is represented by three SOS rows, one containing a first-order denominator section. The 500-Hz UZH-FPV design is sixth order. No filter coefficients, processed signals, trained models, predictions, or forecasting results were changed by this documentation correction.

Original files that show the historical state were not deleted:

- `results/stage1/FILTER_SPECIFICATION.md`
- `results/stage1/RESAMPLING_AUDIT.csv`
- `results/stage1/STAGE1_REPORT.md`
- `audit_original/results/stage1/`
