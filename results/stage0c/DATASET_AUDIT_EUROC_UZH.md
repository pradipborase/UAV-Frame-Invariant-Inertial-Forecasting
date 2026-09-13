# Stage 0C — EuRoC MAV + UZH-FPV Scientific Audit

## 1. Executive decision

**CONDITIONAL_PASS**

utc = 2026-09-12T16:56:32Z

EuRoC official source verified. Phase A Vicon sequences audited: 6. USABLE_PRIMARY: 6. UZH hash-verified retained sequences: 8/8. Recommended measurand: 3-axis accelerometer specific force in each dataset's native IMU sensor frame (EuRoC a_RS_S; UZH lin_acc_*), SI unit m/s^2. Overall compatibility class: COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY.

No modelling, resampling, normalization, full-dataset rotation, or GT acceleration derivation was performed.

## 2. Why Blackbird was retired

Blackbird was the Stage 0 / 0B second dataset. Two official-source acquisition attempts failed: `blackbird-dataset.mit.edu` did not resolve; the official S3 bucket `ijrr-blackbird-dataset.s3.amazonaws.com` answered AccessDenied (HTTP 403) for anonymous GetObject/ListObjects. Blackbird is closed for this study. Historical `results/stage0/` and `results/stage0b/` files are retained. EuRoC MAV replaces Blackbird as D1. UZH-FPV remains D2.

## 3. EuRoC authoritative source and provenance

- Institution: ETH Zurich ASL
- Dataset DOI: 10.3929/ethz-b-000690084
- Paper: Burri et al., IJRR, DOI 10.1177/0278364915620033
- Host: ETH Research Collection bitstream API
- License: In Copyright – Non-Commercial Use Permitted (InC-NC 1.0)
- Official tools: ethz-asl/dataset_tools (not used to alter raw files)
- See `EUROC_SOURCE_VERIFICATION.md`

## 4. EuRoC sequence inventory

Eleven standard flight recordings are documented on the ASL page and confirmed in official zip.json previews (5 Machine Hall + 6 Vicon). Calibration sessions exist separately and are not flight recordings. Remote inventory: `EUROC_REMOTE_SEQUENCE_INVENTORY.csv`.

## 5. EuRoC acquisition

Phase A downloaded official bundled archives `vicon_room1.zip` and `vicon_room2.zip` (total official size 12055648375 bytes, 11.228 GiB) within the 15 GiB additional budget. Machine Hall (`machine_hall.zip`, 12683729426 bytes) was not downloaded. Nested `.bag` files and stereo images were not extracted. Inner sequence zips were extracted to `data/raw/euroc/extracted/`; CSV/YAML working copies to `data/cache/euroc/`. Downloaded Vicon bytes (local): 12055648375.

Downloaded sequences: V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult

## 6. EuRoC IMU

### 6.1 Exact channels

Exact raw header strings are in `EUROC_IMU_AUDIT.csv`. Representative fields (first audited sequence): timestamp `timestamp [ns]`; gyro `w_RS_S_x [rad s^-1]`, `w_RS_S_y [rad s^-1]`, `w_RS_S_z [rad s^-1]`; accel `a_RS_S_x [m s^-2]`, `a_RS_S_y [m s^-2]`, `a_RS_S_z [m s^-2]`.

### 6.2 Physical interpretation

`RAW_ACCELEROMETER_SPECIFIC_FORCE`

Gravity remaining: `YES_CONSISTENT_WITH_SPECIFIC_FORCE_AT_REST`. Not labelled linear acceleration. Bias correction of the published IMU CSV: `NO_NOT_CLAIMED_IN_IMU_CSV`. Factory calibration of the published stream: `UNRESOLVED_FOR_PUBLISHED_STREAM`.

### 6.3 Units

Acceleration `m/s^2`; gyro `rad/s`; timestamps `nanoseconds`.

### 6.4 Sensor hardware

ASL/paper: MEMS IMU ADIS16448, angular rate and acceleration, 200 Hz. File YAML comment/model: `ADIS16448`. Noise densities from sensor.yaml are copied into `EUROC_IMU_AUDIT.csv` when present.

### 6.5 Actual sampling

Documented `200` Hz. Measured medians are per sequence in `EUROC_TIMESTAMP_AUDIT.csv`.

### 6.6 Integrity

See `EUROC_INTEGRITY_FINDINGS.csv`. Critical FAIL count among audited sequences: 0. Major FAIL count: 0.

## 7. EuRoC coordinate frames and calibration

IMU indications are in sensor frame S (`a_RS_S`, `w_RS_S` when those headers are present). `T_BS` in `imu0/sensor.yaml` is a 4x4 transform. Official MATLAB `dataset_plot_body.m` uses `p_BS_B = T_BS(1:3,4)` and `C_BS = T_BS(1:3,1:3)` to plot the sensor in the body, i.e. T_BS maps S to B. Stage 0C does not rotate streams. Details: `EUROC_FRAME_AUDIT.csv`.

## 8. EuRoC ground truth

Vicon sequences: 6-DoF motion-capture pose, published as a spatio-temporally aligned / post-processed state. `state_groundtruth_estimate0` also contains velocity and IMU biases that are **DERIVED_BY_DATASET_AUTHORS**, not independent tracker measurands. Leica Machine Hall 3D position is not a 6-DoF orientation measurement; those sequences were not downloaded. **NO_DIRECT_GT_ACCELERATION**. Representative GT: type `POSTPROCESSED_STATE_ESTIMATE`; frame `R (ASL world/reference; p_RS_R)`.

## 9. EuRoC usable sequences

USABLE_PRIMARY (6): V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult

Full table: `EUROC_USABLE_SEQUENCES.csv`. Machine Hall sequences: NOT_DOWNLOADED.

## 10. Retained UZH-FPV dataset

Eight Stage-0 USABLE_PRIMARY Snapdragon sequences were retained and not redownloaded.

## 11. UZH hash/schema re-verification

Hash PASS count = 8/8. Sequences: indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon. See `UZH_HASH_REVERIFY.csv`. Quantity/unit/frame facts were cross-checked against Stage-0 IMU_AUDIT and a lightweight `imu.txt` header parse. Full expensive UZH audit was not repeated.

## 12. IMU quantity comparability

Both published streams are interpreted as accelerometer specific force (gravity remains) when file evidence supports that label. Hardware differs. Quantity class can still be compared.

## 13. Unit comparability

Acceleration m/s^2 and angular rate rad/s are independently documented. Compatibility: DIRECT. No distribution-estimated scale factors.

## 14. Coordinate-frame comparability

Native IMU frames are each documented. They are not the same physical axes. EuRoC has a documented S->B transform. UZH vehicle body CAD is unresolved. A learned/test-fitted alignment is prohibited. See `EUROC_UZH_FRAME_COMPATIBILITY.csv`.

## 15. Sampling-rate comparability

EuRoC ~200 Hz vs UZH ~500 Hz. Future 100 Hz or 200 Hz conversion is discussed in `SAMPLING_COMPATIBILITY.md` without implementation.

## 16. Ground-truth comparability

Both provide reference trajectories, not direct IMU specific-force labels. Instruments differ (Vicon 6-DoF vs Leica + batch pose). Suitable for temporal segmentation and pose-reference characterization. GT acceleration derivation remains unauthorized. See `EUROC_UZH_GT_COMPATIBILITY.csv`.

## 17. Sequence independence and dependency groups

Each named recording is a separate flight. EuRoC V1_* share room and MAV; V2_* share the second room. UZH indoor_forward / indoor_45 / outdoor_forward are dependency groups. The inferential unit remains the flight/recording, not a window.

## 18. Candidate common measurands

A: native-frame 3-axis specific force. B: common body-frame 3-axis (not currently definable for UZH). C: ||f||. D: native-frame 3-axis angular velocity. E: ||omega||. See `MEASURAND_DECISION.md`.

## 19. Recommended common measurand

3-axis accelerometer specific force in each dataset's native IMU sensor frame (EuRoC a_RS_S; UZH lin_acc_*), SI unit m/s^2

confidence = HIGH

## 20. Recommended primary sequence set

See `PRIMARY_SEQUENCE_SET.csv`. Sequences were not selected by future prediction difficulty.

## 21. Future zero-shot transfer feasibility

Train on Dataset A, freeze, apply to Dataset B without target-test fitted scaling, PCA, coordinate alignment, bias, or time warp. Fixed documented unit conversion is allowed. Fixed documented T_BS on EuRoC is allowed if used. Inventing or learning a UZH body-from-test alignment is not allowed. Feasibility: CONDITIONAL (native-frame quantity; no common official body).

## 22. Remaining limitations

- UZH vehicle-body CAD unresolved
- EuRoC vs UZH IMU axes not officially co-registered
- Vicon synchronization residual (official known issue)
- Machine Hall not downloaded
- Published IMU factory/bias state not fully documented
- Different vehicles and flight styles (scientifically useful, not a defect)

## 23. Blocking issues before Stage 1

Independent researcher/ChatGPT review is required. Stage 1 modelling is not authorized by this file. If hashes or CRITICAL IMU/GT issues appear in the CSVs, they block PASS.

## 24. Final Stage-0C decision

**CONDITIONAL_PASS**
