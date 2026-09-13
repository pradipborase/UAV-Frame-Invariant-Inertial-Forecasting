# STAGE 0C REPORT

EuRoC official source:
VERIFIED

EuRoC DOI:
10.3929/ethz-b-000690084

EuRoC sequences documented:
11 standard flights (6 Vicon + 5 Machine Hall) plus separate calibration sessions

EuRoC sequences downloaded:
6 Vicon sequences (Phase A); Machine Hall not downloaded

EuRoC USABLE_PRIMARY:
6 (V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult)

EuRoC exact acceleration fields:
a_RS_S_x [m s^-2] / a_RS_S_y [m s^-2] / a_RS_S_z [m s^-2]

EuRoC acceleration physical quantity:
RAW_ACCELEROMETER_SPECIFIC_FORCE

EuRoC acceleration unit:
m/s^2

EuRoC exact gyro fields:
w_RS_S_x [rad s^-1] / w_RS_S_y [rad s^-1] / w_RS_S_z [rad s^-1]

EuRoC gyro unit:
rad/s

EuRoC IMU sensor frame:
S (IMU/sensor; a_RS_S / w_RS_S when present)

EuRoC body transform:
T_BS maps sensor-frame coordinates to body-frame coordinates (S -> B)

EuRoC documented rate:
200 Hz

EuRoC measured rate:
see EUROC_TIMESTAMP_AUDIT.csv (median Hz per sequence)

EuRoC GT:
POSTPROCESSED_STATE_ESTIMATE; NO_DIRECT_GT_ACCELERATION

EuRoC GT frame:
R (ASL world/reference; p_RS_R) / S pose in R (p_RS_R, q_RS) per official loader names

EuRoC GT overlap:
ALL_PRIMARY_GE_20S

UZH retained sequences:
indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon

UZH hash verification:
PASS

UZH quantity:
RAW_ACCELEROMETER_SPECIFIC_FORCE

UZH acceleration unit:
m/s^2

UZH gyro unit:
rad/s

UZH IMU frame:
S (Snapdragon IMU)

UZH measured rate:
approximately 500 Hz (Stage 0 timestamp audit)

Recommended common measurand:
3-axis accelerometer specific force in each dataset's native IMU sensor frame (EuRoC a_RS_S; UZH lin_acc_*), SI unit m/s^2

Measurand confidence:
HIGH

3-axis specific-force compatibility:
SAME_QUANTITY_DIFFERENT_AXES (native frames; no target-fitted alignment)

specific-force magnitude compatibility:
YES_ROTATION_INVARIANT (optional; not selected merely for ease)

3-axis gyro compatibility:
SAME_QUANTITY_DIFFERENT_AXES

angular-speed magnitude compatibility:
YES_ROTATION_INVARIANT (optional)

Body-frame standardization:
EuRoC T_BS documented; UZH vehicle body UNRESOLVED; common body NOT currently definable

Sampling compatibility:
FEASIBLE_IN_PRINCIPLE (200 Hz vs 500 Hz; see SAMPLING_COMPATIBILITY.md); not executed

GT compatibility:
BOTH_REFERENCE_TRAJECTORIES; NO_DIRECT_GT_ACCELERATION; instruments differ

Zero-shot cross-dataset feasibility:
CONDITIONAL

Critical findings:
0

Major findings:
0

pytest:
38 passed
0 failed

OVERALL STAGE-0C DECISION:
CONDITIONAL_PASS

NO MODELLING WAS PERFORMED.
NO RESAMPLING WAS PERFORMED.
NO NORMALIZATION WAS PERFORMED.
NO COORDINATE ROTATION OF THE FULL DATASET WAS PERFORMED.
NO GT ACCELERATION WAS DERIVED.
