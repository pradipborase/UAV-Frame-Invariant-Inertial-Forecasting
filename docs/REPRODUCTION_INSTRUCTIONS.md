# Reproduction instructions

Treat this `GITHUB_RELEASE_V3` folder as the project root. Commands below are for Windows PowerShell; on Linux/macOS use `python3` and `source .venv/bin/activate`.

**Do not change** `PROTOCOL_LOCK.md`, `configs/protocol_lock.yaml`, `configs/stage2_baselines.yaml`, `configs/stage3_advanced.yaml`, or the Stage-2/3 freeze files if the goal is to match the frozen study.

Stage 3 retraining can take many hours on CPU. Frozen CSVs under `results/stage2/` and `results/stage3/` already contain the reported numbers. `publication_current/` **does not train**; it recomputes manuscript tables and figures from those CSVs using the source-selected B3 Ridge comparator in all four contexts. Historical `results/stage4/` is retained as a pre-correction audit (UZH→EuRoC used descriptive B2 there). See `docs/POSTHOC_REPORTING_CORRECTIONS.md`.

Candidate history is 100 samples / 1.0 s for every origin. Neural models use all 100 samples. Ridge uses a source-selected lag from `[10, 25, 50, 100]` (primary B3: EuRoC 25, UZH 10). See `docs/INPUT_HISTORY_CLARIFICATION.md`. Publication-facing filter orders are EuRoC **5** and UZH-FPV **6** (`docs/POSTHOC_FILTER_ORDER_CORRECTION.md`).

## 0. Environment

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Study Python: **3.12.6**.

## 1. Official data

Follow `docs/DATASET_DOWNLOAD_INSTRUCTIONS.md`.

Optional automated EuRoC download + text extraction, and UZH `imu.txt` cache extraction:

```text
.\.venv\Scripts\python.exe scripts/download_stage0c_data.py
.\.venv\Scripts\python.exe scripts/run_stage0c_audit.py
.\.venv\Scripts\python.exe scripts/verify_euroc_sources.py
```

## 2. Preprocessing (Stage 1)

Causal elliptic anti-alias (pass 30 Hz / stop 45 Hz; EuRoC/200 Hz order 5, UZH-FPV/500 Hz order 6), integer decimation to **100 Hz**, \(q_f\) and \(q_\omega\), 1.0 s lookback / 0.20 s horizon / 0.05 s stride windows:

```text
.\.venv\Scripts\python.exe scripts/run_stage1.py
```

Writes `data/processed/stage1/` and would rewrite `results/stage1/` if executed. **The public package already includes the historical Stage-1 audit files; do not overwrite them if the goal is to keep the original audit labels.** Processed streams are intentionally not shipped.

## 3. Baseline results, including transfer (Stage 2)

Persistence, linear extrapolation (supplementary), Ridge \(q_f\), Ridge \(q_f+q_\omega\); within-domain LORO and frozen zero-shot transfer:

```text
.\.venv\Scripts\python.exe scripts/run_stage2_baselines.py
```

Writes sequence-level RMSE, horizon curves, paired comparisons, bootstrap summaries, and `results/stage2/SOURCE_MODELS_FROZEN.md`. **Do not rerun this to match the paper; use the frozen CSVs.**

## 4. Advanced-model results and transfer (Stage 3)

Compact TCN / GRU / Transformer. Source-only nested selection, freeze, then EuRoC→UZH and UZH→EuRoC with **no target-fitted scaler, alignment, or model selection**:

```text
.\.venv\Scripts\python.exe scripts/run_stage3_advanced.py
```

This **trains** the compact networks (three seeds). Checkpoints would be written under `artifacts/stage3/` at runtime; they are **not** shipped. **Do not retrain to match the paper.**

Optional supplementary (does not train neural models; does not rewrite Stage-2/3 primaries):

```text
.\.venv\Scripts\python.exe scripts/run_ridge_fixed_lag100_sensitivity.py
```

The package already includes `results/sensitivity/RIDGE_B3_FIXED_LAG100_*.csv`.

## 5. Manuscript tables and figures (`publication_current`)

Recompute manuscript-facing statistics from frozen Stage-2/3 result CSVs. Ridge is **hardcoded** to source-selected `B3_RIDGE_QF_QW` in every context (including UZH→EuRoC). This step never inspects target-domain RMSE to choose Ridge:

```text
.\.venv\Scripts\python.exe scripts/build_publication_current.py --verify
```

`--verify` fails if the primary UZH→EuRoC Ridge is the old B2 value (~0.815321), if the EuRoC filter is labelled order 6 in publication-facing files, if fixed-100-lag files are missing, or if target-selected Ridge logic is detected.

This step must **not** call a training loop. It regenerates `publication_current/` (primary table, skill table, transfer gap, 10–200 ms horizon data, advanced-vs-Ridge paired differences, bootstrap intervals, flight-level paired data, corrected filter metadata, Fig01–Fig05).

Included frozen modelling outputs that this step reads (do not rewrite):

- Stage-2 sequence RMSE: `results/stage2/*_SEQUENCE_RESULTS.csv`, `results/stage2/BOOTSTRAP_SUMMARY.csv`, `results/stage2/SOURCE_MODELS_FROZEN.md`
- Stage-3 advanced sequence RMSE: `results/stage3/*_ADVANCED.csv`, `results/stage3/SELECTED_MODELS.csv`
- Lag-100 sensitivity: `results/sensitivity/RIDGE_B3_FIXED_LAG100_*.csv`

The historical Stage-4 builder `scripts/build_publication_evidence.py` is **not** the manuscript generator.

## 6. Tests

```text
.\.venv\Scripts\python.exe -m pytest -q
```

Relevant groups:

- filter order: `tests/test_filter_actual_order.py`
- Ridge B3 primary: `tests/test_ridge_primary_source_selected.py`
- history contract: `tests/test_input_history_contract.py`
- lag-100 sensitivity: `tests/test_fixed100_sensitivity.py`
- preprocessing / causality: `tests/test_stage1_filter.py`, `tests/test_preprocessing_causality.py`
- leakage / source-only: `tests/test_stage2_leakage.py`
- statistics / reconciliation: `tests/test_bootstrap_unit.py`, `tests/test_numerical_reconciliation.py`

If you have **not** re-run Stages 1–3, keep the included `results/stage2` and `results/stage3` CSVs. Tests that need raw or processed IMU files skip when those trees are absent.

## Notes

- Inferential unit = flight recording (EuRoC n=6, UZH n=8), not windows.
- Bootstrap: 10,000 sequence resamples, seed 20260912.
- Zero-shot means no target-dataset data in parameters, hyperparameters, normalization, alignment, or model selection for the reported source→target evaluation.
