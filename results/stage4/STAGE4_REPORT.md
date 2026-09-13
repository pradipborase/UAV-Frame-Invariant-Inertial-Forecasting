# STAGE 4 REPORT

Lock verification:
PASS (all seven frozen artifacts match expected SHA-256)

Numerical reconciliation:
PASS (0 mismatches above tolerance of 430 checks)

Sequences:
EuRoC = 6
UZH = 8

Primary target:
q_f specific-force magnitude (m/s^2)

Primary contexts:
WITHIN_EUROC, WITHIN_UZH, EUROC_TO_UZH, UZH_TO_EUROC

Primary models:
Persistence; context-specific best frozen Ridge; TCN; GRU; Transformer
Linear extrapolation is supplementary only.

Primary results table:
PASS

Advanced-vs-Ridge verification:
PASS; Delta = RMSE_advanced - RMSE_Ridge (negative = advanced lower)

Horizon verification:
Recomputed h=1..20 equal-sequence means with sequence-bootstrap CIs (not 20 hypothesis tests)

High-q_f verification:
Sequence-specific p95 unchanged vs Stage-2 PREDICTABILITY_CHARACTERIZATION.csv

Transfer-gap verification:
Positive gaps for all learned models; Persistence omitted as structurally untrained

q_w ablation:
Modest auxiliary contribution (see QW_ABLATION_FINAL.csv)

Computational-footprint verification:
sub-millisecond median inference per window on reported CPU

Seed-variability verification:
Three seeds summarized separately; sequences remain inferential units

Major supported conclusions:
- Ridge remains competitive with compact nonlinear models.
- Transformer within-UZH gain does not transfer EuRoC→UZH.
- No architecture dominates all four contexts.
- Zero-shot transfer is feasible without target-fitted adaptation, with descriptive transfer degradation.

Claims weakened/rejected:
- TCN/Transformer as universal winners
- Deep learning significantly outperforms Ridge
- Domain shift eliminated
- True physical acceleration / GT acceleration
- Real-time onboard deployment
- Preregistration
- Bootstrap CIs as measurement uncertainty

Most important limitation:
Small independent-flight counts (n=6/8) and only two datasets.

Second most important limitation:
q_f magnitude discards direction; target is future sensor observation, not independent acceleration truth.

Gravity-dominance characterization:
Sequence-mean median q_f is 9.85 m/s^2 (EuRoC) and 9.49 m/s^2 (UZH), near gravitational acceleration, but dynamic variation is present: mean fraction within 9.81±2 m/s^2 is 96.8% (EuRoC) and 87.3% (UZH); UZH p95 values reach well above 13 m/s^2. The target is not a constant-gravity scalar.

Publication figures:
Fig01–Fig06 (PDF/SVG/PNG) under results/stage4/figures and publication_package/figures

Publication tables:
Table 1 datasets; Table 2 protocol; Table 3 primary RMSE; Table 4 advanced vs Ridge; Table 5 footprint

Reproducibility manifest:
results/stage4/REPRODUCIBILITY_MANIFEST.csv

Final results lock:
results/stage4/FINAL_RESULTS_LOCK.md sha256=25a0f4c6e512b8036dbe591e453f9cfff97e8a89389226d0b2fe5c61214cdb4c

Critical findings:
- None.

Major findings:
- None beyond documented rounding in narrative reports.

pytest:
117 passed, 0 failed

RECOMMENDED MANUSCRIPT EMPHASIS:
Frame-invariant cross-platform UAV inertial forecasting with emphasis on the complexity-generalization trade-off under zero-shot dataset transfer.

RECOMMENDED PRIMARY TITLE:
Frame-Invariant Short-Horizon UAV Inertial Forecasting: Cross-Dataset Evaluation of Complexity and Zero-Shot Generalization

READY FOR MANUSCRIPT WRITING:
YES

OVERALL STAGE-4 DECISION:
PASS

No new model was fitted in Stage 4. No hyperparameter was retuned. No target-domain adaptation was performed.
