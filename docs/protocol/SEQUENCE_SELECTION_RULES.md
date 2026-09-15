SEQUENCE_SELECTION_RULES.md

Current manuscript experiment: EuRoC MAV + UZH-FPV

These rules describe the final experiment reported in A Rotation-Invariant, Leakage-Controlled Evaluation Protocol for Cross-Platform Short-Horizon UAV Inertial Forecasting.

Selection is frozen against forecasting performance and is based on official-source provenance and measurement usability. An overlapping window is never an independent experimental unit; the inferential unit is an actual flight recording / sequence.

General eligibility / usability rule

A primary recording must satisfy the following quality and provenance conditions:

obtained from an official dataset source;

readable accelerometer and gyroscope measurements;

physical units verified from raw data / official documentation / calibration material;

sensor frame or calibration relation understood sufficiently for the scalar measurement protocol;

timestamps auditable and monotonic over the usable interval;

public reference data available for dataset integrity and overlap auditing;

sufficient valid IMU/reference overlap for the planned sequence-level analysis;

overwhelmingly finite IMU samples and no identified file corruption;

inclusion/exclusion determined independently of forecasting error.

Irregular sampling is reported and audited; it is not automatically an exclusion if the recording remains usable under the locked causal-resampling protocol.

EuRoC MAV primary subset

The final primary EuRoC subset is fixed to the six Vicon-room recordings:

V1_01_easy

V1_02_medium

V1_03_difficult

V2_01_easy

V2_02_medium

V2_03_difficult

This subset spans both Vicon rooms and the easy, medium, and difficult sequence labels. The two Vicon rooms define the EuRoC source-validation groups. Other EuRoC recordings are outside the frozen primary subset; they were not excluded because of observed forecasting performance.

UZH-FPV Snapdragon primary subset

The final primary UZH-FPV subset is fixed to the following eight Snapdragon recordings with public reference data:

indoor_forward_6_snapdragon

indoor_forward_9_snapdragon

indoor_forward_10_snapdragon

indoor_45_2_snapdragon

indoor_45_4_snapdragon

indoor_45_13_snapdragon

indoor_45_14_snapdragon

outdoor_forward_1_snapdragon

The source-validation groups are:

indoor_forward

indoor_45

outdoor_forward

Snapdragon and DAVIS streams from the same underlying UZH recording are not treated as independent flights. Recordings with withheld/unavailable public reference data are not promoted to primary experimental units through unofficial reconstruction.

Independence rule

DATASET → RECORDING / FLIGHT → OVERLAPPING WINDOWS

The recording / flight is the inferential unit. Overlapping windows are operational forecasting instances, not independent statistical samples.

Performance-blind selection rule

The retained recording list and group definitions were fixed before forecasting results were inspected. Forecasting RMSE, model ranking, transfer performance, or bootstrap results must never be used to decide whether a recording belongs to the primary set.

Historical Blackbird provenance

Earlier Stage-0/0B work considered the Blackbird dataset. Blackbird was subsequently retired from the final experiment and replaced by EuRoC MAV before the locked EuRoC+UZH forecasting analysis. Any older Blackbird-specific selection notes are historical provenance only and are not part of the final manuscript experiment.
