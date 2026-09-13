# Historical Stage-1 metadata snapshot

These files are an explicit copy of the **original** Stage-1 audit metadata shipped under `results/stage1/`.

They document the historical (incorrect) filter-order **labels**:

- both EuRoC/200 Hz and UZH-FPV/500 Hz designs labelled order 6
- because metadata used `order = 2 * n_SOS_sections`

Executed SOS **coefficients** in `FILTER_SPECIFICATION.md` are the frozen truth and were not changed.

Do not treat this folder as the manuscript-facing specification. Use:

- `publication_current/FILTER_SPECIFICATION_CORRECTED.md`
- `publication_current/RESAMPLING_AUDIT_CORRECTED.csv`

The copies under `results/stage1/` are the same historical files and were not overwritten in this public package.
