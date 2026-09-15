UAV inertial forecasting — reproducibility package

This repository is the reproducibility package for:

A Rotation-Invariant, Leakage-Controlled Evaluation Protocol for Cross-Platform Short-Horizon UAV Inertial Forecasting

Authors: Pradip Diwan Borase and Vijyant Agarwal.

Repository: https://github.com/pradipborase/UAV-Frame-Invariant-Inertial-Forecasting

The package contains the source code, locked protocol/configuration material, tests, frozen machine-readable results, supplementary outputs, and manuscript-facing publication artifacts used to support the study. Raw third-party EuRoC MAV and UZH-FPV data are not redistributed.

Important publication-current note

For manuscript numbers and figures, use the publication-current outputs in publication_current/ and the associated supplementary files. Historical Stage-1/Stage-4 audit files and historical orchestration code are preserved for provenance and may contain superseded reporting metadata. In particular:

the primary Ridge comparator is B3_RIDGE_QF_QW in all four evaluation contexts;

the EuRoC anti-alias filter is order 5, while the UZH-FPV filter is order 6;

historical files that show an older UZH-FPV→EuRoC B2 descriptive comparator are not authoritative for the manuscript;

historical Blackbird selection notes are retained only as provenance from an earlier retired dataset stage and are not part of the final EuRoC+UZH-FPV experiment.

The authoritative comparator mapping is publication_current/RIDGE_COMPARATOR_MAP.csv.

Research question

Can a rotation-invariant, leakage-controlled protocol compare short-horizon forecasts of accelerometer specific-force magnitude across two heterogeneous UAV IMU datasets while keeping all fitted transformations and model-selection decisions source-only during zero-shot transfer?

The study compares Persistence, source-selected B3 Ridge, and compact TCN / GRU / Transformer models in four contexts:

within EuRoC MAV,

within UZH-FPV,

zero-shot EuRoC→UZH-FPV, and

zero-shot UZH-FPV→EuRoC.

Datasets

EuRoC MAV — 6 retained Vicon-room recordings

V1_01_easy

V1_02_medium

V1_03_difficult

V2_01_easy

V2_02_medium

V2_03_difficult

UZH-FPV Snapdragon — 8 retained recordings

indoor_forward_6_snapdragon

indoor_forward_9_snapdragon

indoor_forward_10_snapdragon

indoor_45_2_snapdragon

indoor_45_4_snapdragon

indoor_45_13_snapdragon

indoor_45_14_snapdragon

outdoor_forward_1_snapdragon

The retained recording list was fixed independently of forecasting performance. Current selection/usability rules are documented in docs/protocol/SEQUENCE_SELECTION_RULES.md.

Physical quantities

Primary forecast quantity:

[
q_f(t)=|f(t)|_2 \quad (\mathrm{m/s}^2)
]

Auxiliary input:

[
q_\omega(t)=|\omega(t)|_2 \quad (\mathrm{rad/s})
]

The Euclidean magnitudes are invariant to a fixed proper orthonormal rotation of the sensor axes. This does not make the platforms physically identical and does not remove differences in sensor hardware, scale factors, bandwidth, bias, noise, dynamics, or operating regime.

Causal preprocessing

The locked pipeline is:

validate native-frame SI inertial channels;

compute q_f and q_omega at native sample times;

apply forward-only causal elliptic SOS anti-alias filtering;

integer-decimate to a nominal 100 Hz output rate;

discard a fixed 1.0 s warm-up interval.

Filter specification: passband 30 Hz, stopband 45 Hz, passband ripple ≤1 dB, stopband attenuation ≥60 dB.

EuRoC native 200 Hz design: order 5

UZH-FPV native 500 Hz design: order 6

No filtfilt, sosfiltfilt, zero-phase smoothing, target-fitted alignment, or target-fitted time warp is used.

Input-history contract

Role

Samples

Physical duration

Candidate history at each origin

100

1.0 s

TCN / GRU / Transformer history

100

1.0 s

Ridge effective history

selected from [10, 25, 50, 100]

0.10–1.0 s

Forecast horizon

20

0.20 s

Origin stride

5

0.05 s

Model input rate

100 Hz

after causal filtering + decimation

Primary B3 Ridge lags:

EuRoC: 25 samples, alpha = 1e-4

UZH-FPV: 10 samples, alpha = 1e-2

A supplementary fixed-100-lag Ridge sensitivity is retained to assess history-length fairness.

Statistical unit and zero-shot rule

inferential unit = flight recording / sequence, not overlapping window;

dataset-level performance = equal-sequence mean of sequence-level multi-horizon RMSE;

uncertainty intervals = 10,000 sequence-level bootstrap resamples, seed 20260912;

zero-shot transfer = no target-domain fitting, scaling, coordinate alignment, early stopping, architecture selection, or hyperparameter selection.

For TCN, GRU, and Transformer, three fixed training seeds (20260912, 20260913, 20260914) are evaluated separately. Reported sequence-level neural metrics are arithmetic means of the three seed-specific metrics; predictions are not first ensemble-averaged and seed replicates are not treated as independent flights.

Primary Ridge comparator

B3_RIDGE_QF_QW is the primary Ridge comparator in all four contexts because it is selected from source-domain validation only.

The historical B2 result with lower UZH-FPV→EuRoC target RMSE is retained only as a supplementary sensitivity result and is not used to define the primary comparator.

Authoritative file: publication_current/RIDGE_COMPARATOR_MAP.csv.

Reproduce manuscript-facing tables and figures

Install dependencies, then run:

python -m pip install -r requirements.txt
python scripts/build_publication_current.py --verify

This rebuilds manuscript-facing tables and figures from frozen numeric result files. It does not retrain the neural models and does not choose a Ridge comparator from target-domain RMSE.

Important outputs in publication_current/ include:

PRIMARY_RESULTS.csv

PERSISTENCE_SKILL.csv

TRANSFER_GAPS.csv

ADVANCED_VS_RIDGE.csv

HORIZON_RMSE_10_TO_200MS.csv

RIDGE_COMPARATOR_MAP.csv

corrected filter metadata

manuscript figures

Do not use historical publication builders or Stage-4 files as the source of final manuscript numbers unless they are explicitly marked publication-current.

Repository layout

Path

Contents

src/stage1/

Native IMU loading, magnitude construction, causal filtering, 100 Hz pipeline, windows/folds

src/stage2/

Persistence/Ridge baselines, source-only scaling, statistics, transfer, lag-100 sensitivity

src/stage3/

TCN, GRU, Transformer training and frozen transfer code; historical orchestration artifacts are retained for provenance

configs/

Frozen protocol/model configuration files

tests/

Causality, leakage, filter-order, Ridge-B3, history-contract, sensitivity, and publication checks

results/stage1/

Historical Stage-1 audit artifacts

results/stage2/, results/stage3/

Frozen modelling outputs

results/sensitivity/

Fixed-100-lag Ridge sensitivity outputs

publication_current/

Authoritative manuscript-facing outputs

supplementary/

Supplementary machine-readable results

docs/

Protocol, corrections, selection rules, and reproduction documentation

Historical provenance

Some historical files intentionally preserve earlier study states. They should not be silently deleted because they document the audit trail. Where a historical file conflicts with publication-current reporting, the current README, publication_current/, correction notes, and current manuscript take precedence. Historical files should carry a visible warning when practical.

Data and licence

EuRoC MAV and UZH-FPV remain under their original dataset licences. This repository's code licence does not relicense third-party raw data.

Citation and archive

See CITATION.cff for the current manuscript title and citation metadata.

The computational release v1.0.0 is archived on Zenodo under DOI 10.5281/zenodo.22751169. If repository documentation or manuscript-facing metadata are synchronized after v1.0.0, create a new Zenodo version and use that newer version-specific DOI in the final submitted manuscript metadata where appropriate.
