# UAV inertial forecasting — reproducibility package

This directory is the GitHub/Zenodo reproducibility package for:

**A Frame-Invariant Evaluation Protocol for Cross-Platform Short-Horizon UAV Inertial Forecasting**

Authors: Pradip Diwan Borase and Vijyant Agarwal.

Intended repository: `https://github.com/pradipborase/UAV-Frame-Invariant-Inertial-Forecasting`

It contains the **actual source code, locked configuration, tests, and frozen machine-readable results** used for the study. It is a copy of selected project files; it is not a rewritten or simplified substitute.

## Research question

Can a **frame-invariant**, leakage-safe protocol compare short-horizon inertial forecasts of accelerometer specific-force magnitude across two official UAV IMU platforms, using sequence-level statistics, without fitting the target dataset?

The study compares Persistence, a source-selected B3 Ridge comparator, and compact TCN / GRU / Transformer models in four contexts: within EuRoC, within UZH-FPV, zero-shot EuRoC→UZH-FPV, and zero-shot UZH-FPV→EuRoC.

## Datasets (raw files are not redistributed)

**EuRoC MAV and UZH-FPV raw data are not included.** Obtain them from the official hosts. See `docs/DATASET_DOWNLOAD_INSTRUCTIONS.md`.

### EuRoC MAV (n = 6, Vicon rooms)

- `V1_01_easy`
- `V1_02_medium`
- `V1_03_difficult`
- `V2_01_easy`
- `V2_02_medium`
- `V2_03_difficult`

### UZH-FPV Snapdragon (n = 8)

- `indoor_forward_6_snapdragon`
- `indoor_forward_9_snapdragon`
- `indoor_forward_10_snapdragon`
- `indoor_45_2_snapdragon`
- `indoor_45_4_snapdragon`
- `indoor_45_13_snapdragon`
- `indoor_45_14_snapdragon`
- `outdoor_forward_1_snapdragon`

This package also does **not** include Python virtual environments, training checkpoints (`*.pt`), processed IMU streams (`data/processed/`), passwords, tokens, or `.env` files. Frozen **numeric results** are included so tables and figures can be rebuilt without retraining.

## Quantity

Primary target:

\[
q_f(t)=\|f(t)\|_2 \quad (\mathrm{m/s}^2)
\]

Auxiliary:

\[
q_\omega(t)=\|\omega(t)\|_2 \quad (\mathrm{rad/s})
\]

## Input-history contract

Do not omit the Ridge lag qualification: neural models consume the full 100-sample window; Ridge uses a source-selected suffix (the most recent samples) of that window. Details: `docs/INPUT_HISTORY_CLARIFICATION.md`.

| Role | Samples | Physical |
| --- | ---: | --- |
| Candidate history (every origin) | 100 | 1.0 s |
| Neural effective history (TCN, GRU, Transformer) | 100 | 1.0 s |
| Ridge effective history | source-selected lag from `[10, 25, 50, 100]` | 0.10–1.0 s |
| Forecast horizon | 20 | 0.20 s |
| Origin stride | 5 | 0.05 s |
| Model input rate | 100 Hz | after causal anti-alias + integer decimation |

Primary B3 Ridge lags: EuRoC **25**, UZH-FPV **10**.

## Causal anti-alias filters (executed orders)

Elliptic IIR SOS, pass 30 Hz / stop 45 Hz, gpass 1 dB / gstop 60 dB, forward `sosfilt` only.

- **EuRoC native 200 Hz design: order 5** (three SOS rows; one first-order denominator section)
- **UZH-FPV native 500 Hz design: order 6**

Historical Stage-1 files labelled both as order 6 because metadata used `2 * n_SOS_rows`. Coefficients, processed streams, and forecasts were not changed. See `docs/POSTHOC_FILTER_ORDER_CORRECTION.md`.

## Primary Ridge comparator

`B3_RIDGE_QF_QW` in **all four** contexts, selected exclusively by source-domain validation.

- EuRoC: `[q_f, q_w]`, lag = 25, alpha = 1e-4
- UZH-FPV: `[q_f, q_w]`, lag = 10, alpha = 1e-2

UZH-FPV → EuRoC primary B3 mean RMSE ≈ **0.830804** m/s².

The older descriptive B2 target result ≈ 0.815321 is **not** the primary comparator. B2 is supplementary sensitivity only.

B3 is the primary Ridge comparator in both zero-shot transfer directions because it was selected exclusively by source-domain validation. See `docs/POSTHOC_REPORTING_CORRECTIONS.md`.

## Fixed 100-lag Ridge (supplementary)

A post hoc equal-history sensitivity forces B3 lag = 100 and retunes only alpha on source data. It **does not replace** primary source-selected-lag Ridge. Forced lag-100 **raises** aggregate Ridge RMSE in all four contexts. Within EuRoC it modestly favors nonlinear models versus that worse Ridge; **no architecture dominates all domains**. See `docs/RIDGE_FIXED100_HISTORY_SENSITIVITY.md`.

## Inferential unit, bootstrap, zero-shot

- Inferential unit = **flight recording** (sequence), not sliding windows (EuRoC n = 6, UZH-FPV n = 8).
- Bootstrap: **10,000** sequence-level resamples, seed **20260912**.
- Zero-shot transfer: no target-domain fitting, scaling, alignment, early stopping, or architecture selection.

## How to reproduce manuscript tables and figures from frozen CSVs

Exact command sequence for a full from-raw rerun: `docs/REPRODUCTION_INSTRUCTIONS.md`.

To regenerate **only** the paper tables and figures (no training):

```text
python -m pip install -r requirements.txt
python scripts/build_publication_current.py --verify
```

This reads frozen `results/stage2/` and `results/stage3/` (and copies `results/sensitivity/` Ridge-lag100 files). It does **not** train TCN/GRU/Transformer and **never** selects Ridge by target-domain RMSE.

Manuscript-facing outputs: `publication_current/` (`PRIMARY_RESULTS.csv`, `PERSISTENCE_SKILL.csv`, `TRANSFER_GAPS.csv`, `ADVANCED_VS_RIDGE.csv`, `HORIZON_RMSE_10_TO_200MS.csv`, `RIDGE_COMPARATOR_MAP.csv`, corrected filter metadata, Fig01–Fig05).

Do **not** use historical `scripts/build_publication_evidence.py` for manuscript numbers.

Locked protocol hashes (must match `PROTOCOL_LOCK.md` / `docs/protocol/PROTOCOL_LOCK.md`):

- `PROTOCOL_LOCK.md` SHA-256: `b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed`
- `configs/protocol_lock.yaml` SHA-256: `8fd75b59e35aac291f597fb856ff1debfc9609cf7a52adf6fc0e3502bf3c0d7d`

## Python and dependencies

- Python **3.12.6** (study environment)
- NumPy, SciPy, PyYAML, scikit-learn, matplotlib, joblib, pytest
- PyTorch **CPU** (`torch>=2.0` is sufficient to install)

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Tests (skip cleanly if raw/processed IMU files are absent):

```text
python -m pytest -q
```

## Layout

| Path | Contents |
| --- | --- |
| `src/stage1/` | Native IMU loaders, \(q_f\)/\(q_\omega\), causal elliptic SOS, 100 Hz pipeline, windows/folds |
| `src/stage2/` | Ridge / persistence / linear baselines, source-only scaler, bootstrap, transfer, lag-100 sensitivity |
| `src/stage3/` | TCN, GRU, Transformer, training, freeze, transfer |
| `src/stage4/` | Historical Stage-4 audit plus `publication_current` recompute (no training) |
| `configs/` | Frozen YAML for protocol and Stages 1–3 |
| `tests/` | Causality, leakage, filter order, Ridge B3, history contract, lag-100, publication tests |
| `results/stage2/` `results/stage3/` | Frozen modelling CSVs (do not rewrite) |
| `results/stage1/` | Historical Stage-1 audit (original order labels retained) |
| `results/sensitivity/` | Fixed 100-lag Ridge B3 supplementary CSVs |
| `publication_current/` | Authoritative manuscript-facing tables and Fig01–Fig05 |
| `supplementary/` | Requested machine-readable supplementary CSVs |
| `docs/` | Dataset placement, reproduction, post-hoc corrections, history contract |
| `paper_support/` | Placeholder: Revision-12 manuscript `.tex` files were not available at packaging |

## Licence

Our code in this package is MIT (see `LICENSE`). **EuRoC MAV and UZH-FPV remain under their original dataset licences** and are not granted by the MIT licence.

## Citation

See `CITATION.cff`. No Zenodo DOI is inserted in this package.
