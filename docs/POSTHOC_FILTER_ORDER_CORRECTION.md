# Post-hoc filter-order metadata correction

This document records a **documentation correction**. It is not a filter redesign, a reprocessing of IMU streams, or a retraining step.

The Stage-1 filtering calculations and processed data were correct, but the filter order was misreported because metadata inferred order as twice the number of SOS rows. The 200-Hz EuRoC design is fifth order and is represented by three SOS rows, one containing a first-order denominator section. The 500-Hz UZH-FPV design is sixth order. No filter coefficients, processed signals, trained models, predictions, or forecasting results were changed by this documentation correction.

## What was wrong

Historical Stage-1 writers used the equivalent of:

```text
order = 2 * sos.shape[0]
```

Both designs have three SOS rows, so both were labelled order 6. That rule is wrong for an odd-order elliptic filter stored in padded SOS form: the EuRoC 200 Hz design has one first-order denominator section (`a2 = 0`).

## Actual executed designs (unchanged coefficients)

Locked elliptic specification (both datasets):

- passband edge 30 Hz, stopband edge 45 Hz
- gpass 1 dB, gstop 60 dB
- `ftype="ellip"`, `output="sos"`
- forward causal `scipy.signal.sosfilt` only (no `filtfilt` / zero-phase)

| Native rate | Dataset | True IIR order | SOS rows | Historical label |
| --- | --- | ---: | ---: | ---: |
| 200 Hz | EuRoC | **5** | 3 | 6 |
| 500 Hz | UZH-FPV | **6** | 3 | 6 |

Executed SOS coefficients remain exactly those stored in `results/stage1/FILTER_SPECIFICATION.md`.

## Where to read which file

- **Historical (original labels):** `results/stage1/FILTER_SPECIFICATION.md`, `results/stage1/RESAMPLING_AUDIT.csv`, `results/stage1/STAGE1_REPORT.md`, and the snapshot under `audit_original/results/stage1/`.
- **Publication-facing (corrected labels, same coefficients):** `publication_current/FILTER_SPECIFICATION_CORRECTED.md`, `publication_current/RESAMPLING_AUDIT_CORRECTED.csv`.

Future Stage-1 metadata writers in this package count denominator order from SOS (`a2 = 0` ⇒ first-order section) instead of `2 * n_sections`. Re-running Stage 1 is **not** required to use the frozen forecasts.

## What was not done

- No processed IMU stream was regenerated.
- No Stage-2 or Stage-3 model was retrained or retuned.
- No target-domain data were used.
- Historical Stage-1 audit files were not overwritten.
