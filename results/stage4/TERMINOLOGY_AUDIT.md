# Terminology audit

Use **accelerometer specific force** for the IMU accelerometer measurand, not generic "acceleration".

Use **specific-force magnitude** for q_f = ||f||_2.

Use **reference trajectory** for pose products. "Ground truth" may appear when quoting dataset terminology; define that it is not independent IMU acceleration truth.

Bootstrap intervals are **sequence-level sampling uncertainty**, not a GUM measurement-uncertainty budget, and not "measurement uncertainty".

Protocol language: **pre-specified locked protocol**, not pre-registered (no OSF/Zenodo registration before modelling).

Zero-shot transfer: no target-dataset data were used to fit parameters, hyperparameters, normalization, coordinate alignment, or model-selection decisions for the reported source-to-target evaluation. Do **not** claim the target dataset was unseen throughout the entire study; datasets were audited before model evaluation.
