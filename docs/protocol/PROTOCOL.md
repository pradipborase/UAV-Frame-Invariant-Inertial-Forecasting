# PROTOCOL.md — Stage 0 frozen scientific rules

These principles are frozen for the project. They are not model hyperparameters.

## Frozen

1. **Raw files immutable.** Everything under `data/raw/` remains byte-identical after download. Parsing goes to memory or `data/cache/`.
2. **Official sources only.** Historical Stage 0/0B: MIT AERA / `blackbird-dataset.mit.edu` / `mit-aera/Blackbird-Dataset` for Blackbird (now retired from the experiment; audit files retained). Stage 0C: ETH Zurich ASL / ETH Research Collection EuRoC MAV (DOI 10.3929/ethz-b-000690084) and University of Zurich RPG / `fpv.ifi.uzh.ch` / `rpg.ifi.uzh.ch` for UZH-FPV. No Kaggle, no random GitHub mirrors, no third-party ML dumps. If an official host is down, record `OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE` and stop substituting.
3. **Session/flight is the future inferential unit.** An independent experimental unit is an actual recording, not an overlapping window.
4. **No window-level pseudo-replication for inferential statistics.** Nearby speeds of the same trajectory family remain a documented dependency group.
5. **Future preprocessing parameters must be learned only from training data.**
6. **Target-test data must never determine a zero-shot scaler** or a fitted frame/unit warp.
7. **Physical-unit and frame traceability required.** Every important statement carries an evidence class. `UNRESOLVED` is not replaced by a guess.
8. **Ground truth and sensor measurements remain conceptually distinct.** Pose GT is not accelerometer specific force. Differentiated GT acceleration is not authorized in Stage 0.
9. **Inclusion/exclusion is independent of forecasting performance.** Usability uses integrity, units, frames, overlap, and public GT only.

## Explicitly not frozen in Stage 0

Sampling rate for models, lookback, forecast horizon, architecture, train/test split, metric set, noise stress levels, statistical hypothesis test.

## Evidence classes

`VERIFIED_FROM_RAW_FILE`  
`VERIFIED_FROM_OFFICIAL_DOCUMENTATION`  
`VERIFIED_FROM_DATASET_PAPER`  
`VERIFIED_FROM_CALIBRATION_FILE`  
`VERIFIED_FROM_OFFICIAL_SOURCE_CODE`  
`DERIVED_FROM_RAW_TIMESTAMPS`  
`UNRESOLVED`

## Stage 0 numerical work

The only allowed numerical work is data auditing: hashes, timestamp statistics, channel descriptive statistics, overlap, integrity flags. No resampling, interpolation, smoothing, clipping, rotation, or merging of datasets.

## Inferential unit (frozen; restated for Stage 0C)

Even if later modelling creates many overlapping windows, the inferential unit remains the **flight / recording / sequence**, not the window. Windows are not independent experimental units. This rule is unchanged by replacing Blackbird with EuRoC MAV.


## Stage 1 locked additions

10. **Frame-invariant primary representation.** Cross-dataset principal target is accelerometer specific-force magnitude q_f, not native 3-axis components.
11. **Causal 100 Hz resampling.** Elliptic SOS anti-alias (30/45 Hz, <=1 dB / >=60 dB) then integer decimation. No zero-phase filtering.
12. **Inferential unit remains the sequence.** Window counts are operational, not statistical sample sizes.
13. **Source-train-only fitted transforms.** Physical Euclidean norms, unit conversion, locked causal filters, and integer decimation are deterministic (FIT_SCOPE = NONE). Scalers, if used later, are SOURCE_TRAIN_ONLY.
14. **Do not call this an OSF/Zenodo registration.** Use PRE-SPECIFIED / LOCKED PROTOCOL unless an external timestamped record exists before modelling.
