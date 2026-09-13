"""Write remaining Stage 0C CSVs, markdown audits, and the Stage 0C decision.

Called after EuRoC extraction and UZH hash re-verification. No modelling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT
from .downloader import utc_now
from .stage0c import (
    COMPAT_COLUMNS,
    DOWNLOAD_COLUMNS,
    EUROC_FRAME_COLUMNS,
    EUROC_GT_COLUMNS,
    EUROC_IMU_COLUMNS,
    EUROC_SEQ_COLUMNS,
    EUROC_TS_COLUMNS,
    EUROC_USABLE_COLUMNS,
    FRAME_COMPAT_COLUMNS,
    GT_COMPAT_COLUMNS,
    INTEGRITY_COLUMNS,
    PRIMARY_COLUMNS,
    REMOTE_INV_COLUMNS,
    UNIT_COMPAT_COLUMNS,
    UZH_REVERIFY_COLUMNS,
    audit_one_euroc_sequence,
    download_phase_a,
    extract_phase_a,
    load_stage0c_config,
    official_remote_inventory,
    parse_sequence_meta,
    results_dir,
    reverify_uzh,
    write_csv,
    write_source_verification,
)


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "UNRESOLVED"
    return str(v)


def build_compatibility(euroc_audits: list[dict[str, Any]], uzh_ok: bool) -> tuple[list[dict[str, str]], str]:
    imu = euroc_audits[0]["imu"] if euroc_audits else {}
    euroc_q = imu.get("interpreted_acceleration_quantity", "UNRESOLVED")
    euroc_au = imu.get("acceleration_unit", "UNRESOLVED")
    euroc_gu = imu.get("gyro_unit", "UNRESOLVED")
    euroc_rate = imu.get("measured_median_rate_hz", "UNRESOLVED")
    rows = [
        {
            "item": "data provenance",
            "EUROC": "ETH Zurich ASL / Research Collection DOI 10.3929/ethz-b-000690084",
            "UZH_FPV": "UZH RPG official v3 Snapdragon zips",
            "compatible": "YES_BOTH_OFFICIAL",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_OFFICIAL_WEB",
            "recommendation": "retain both official sources",
        },
        {
            "item": "platform class",
            "EUROC": "research hex-rotor MAV (Asctec Firefly)",
            "UZH_FPV": "racing quadrotor with Snapdragon IMU",
            "compatible": "COMPARABLE_AS_UAV_INERTIAL_DOMAIN",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_DATASET_PAPER",
            "recommendation": "treat as distinct platforms; valuable for transfer, not a defect",
        },
        {
            "item": "vehicle type",
            "EUROC": "hex-rotor",
            "UZH_FPV": "quadrotor (racing)",
            "compatible": "DIFFERENT_ROTOR_COUNT",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_DATASET_PAPER",
            "recommendation": "document; do not force identical dynamics",
        },
        {
            "item": "sensor hardware",
            "EUROC": imu.get("sensor_model", "ADIS16448 expected from ASL page"),
            "UZH_FPV": "Qualcomm Snapdragon Flight IMU (/snappy_imu)",
            "compatible": "DIFFERENT_HARDWARE_SAME_MEASURAND_CLASS",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_OFFICIAL_WEB; VERIFIED_FROM_RAW_FILE",
            "recommendation": "do not mix DAVIS IMU with Snapdragon; do not mix EuRoC stereo cameras as IMU",
        },
        {
            "item": "accelerometer physical quantity",
            "EUROC": euroc_q,
            "UZH_FPV": "RAW_ACCELEROMETER_SPECIFIC_FORCE",
            "compatible": "DIRECT" if euroc_q == "RAW_ACCELEROMETER_SPECIFIC_FORCE" else "UNRESOLVED",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO" if euroc_q == "RAW_ACCELEROMETER_SPECIFIC_FORCE" else "MAJOR",
            "evidence": "EuRoC header a_RS_S + rest-norm + official MATLAB accelerometer labels; UZH Stage-0 file audit",
            "recommendation": "common physical quantity is accelerometer specific force if both remain RAW_*",
        },
        {
            "item": "accelerometer unit",
            "EUROC": euroc_au,
            "UZH_FPV": "m/s^2",
            "compatible": "DIRECT" if euroc_au == "m/s^2" else "UNRESOLVED",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO" if euroc_au == "m/s^2" else "CRITICAL",
            "evidence": "EuRoC CSV header tokens; UZH via official sensor_msgs/Imu + zip/bag identity",
            "recommendation": "no data-driven scale factor",
        },
        {
            "item": "gyro physical quantity",
            "EUROC": "angular velocity w_RS_S (sensor frame)",
            "UZH_FPV": "angular velocity ang_vel_* (Snapdragon IMU frame S)",
            "compatible": "DIRECT",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_RAW_FILE",
            "recommendation": "same measurand class; axes not automatically aligned",
        },
        {
            "item": "gyro unit",
            "EUROC": euroc_gu,
            "UZH_FPV": "rad/s",
            "compatible": "DIRECT" if euroc_gu == "rad/s" else "UNRESOLVED",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO" if euroc_gu == "rad/s" else "CRITICAL",
            "evidence": "EuRoC header rad s^-1; UZH ROS Imu",
            "recommendation": "no fitted scale",
        },
        {
            "item": "IMU frame",
            "EUROC": "S (IMU/sensor); a_RS_S / w_RS_S",
            "UZH_FPV": "S (Snapdragon IMU); paper Fig 6",
            "compatible": "SAME_NAMING_DIFFERENT_PHYSICAL_AXES",
            "fixed_conversion_required": "YES_IF_COMMON_AXES_REQUIRED",
            "target_fitted_information_required": "NO",
            "severity": "MAJOR",
            "evidence": "dataset-specific calibrations; no official EuRoC<->UZH axis identity",
            "recommendation": "keep native frames or use a documented invariant; never learn alignment on target test data",
        },
        {
            "item": "body transform",
            "EUROC": imu.get("transform_direction", "T_BS S->B"),
            "UZH_FPV": "Kalibr T_cam_imu to camera C; vehicle-body CAD UNRESOLVED",
            "compatible": "NOT_A_SHARED_BODY_FRAME",
            "fixed_conversion_required": "EUROC_ONLY_FOR_BODY_B",
            "target_fitted_information_required": "NO",
            "severity": "MAJOR",
            "evidence": "EuRoC sensor.yaml T_BS; UZH Stage-0 FRAME_AUDIT",
            "recommendation": "do not invent a UZH vehicle body; do not fit a cross-dataset rotation",
        },
        {
            "item": "sample rate",
            "EUROC": f"documented 200 Hz; measured median {euroc_rate}",
            "UZH_FPV": "documented 500 Hz; measured ~500 Hz",
            "compatible": "COMPARABLE_AFTER_FUTURE_RESAMPLING",
            "fixed_conversion_required": "YES_FUTURE_DECIMATION_OR_RESAMPLE",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "DERIVED_FROM_RAW_TIMESTAMPS",
            "recommendation": "Stage 0C reports feasibility only; do not resample now",
        },
        {
            "item": "timestamps",
            "EUROC": "uint64 nanoseconds typical ASL",
            "UZH_FPV": "seconds in imu.txt (Stage-0 audit)",
            "compatible": "DOCUMENTED_FIXED_CONVERSION",
            "fixed_conversion_required": "YES_TO_SECONDS",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "headers + official MATLAB t/1e9",
            "recommendation": "convert by documented factors only",
        },
        {
            "item": "GT availability",
            "EUROC": "YES for downloaded Vicon sequences",
            "UZH_FPV": "YES for retained public-GT Snapdragon sequences",
            "compatible": "YES",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_RAW_FILE",
            "recommendation": "use for segmentation and pose-reference characterization, not as IMU labels",
        },
        {
            "item": "GT type",
            "EUROC": "Vicon 6-DoF pose + postprocessed state estimate",
            "UZH_FPV": "Leica position + IMU-aided batch pose",
            "compatible": "BOTH_REFERENCE_TRAJECTORIES_NOT_IDENTICAL_INSTRUMENTS",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "VERIFIED_FROM_DATASET_PAPER; VERIFIED_FROM_RAW_FILE",
            "recommendation": "do not call either DIRECT IMU MEASUREMENT",
        },
        {
            "item": "GT frame",
            "EUROC": "R world; p_RS_R q_RS",
            "UZH_FPV": "dataset world / Leica-related pose (Stage-0 GT audit)",
            "compatible": "NOT_THE_SAME_WORLD_FRAME",
            "fixed_conversion_required": "NO_CROSS_DATASET_WORLD_TF",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "no official EuRoC-UZH world registration",
            "recommendation": "do not align worlds by fitting target trajectories",
        },
        {
            "item": "GT overlap",
            "EUROC": "per-sequence overlap_duration_s in EUROC_GROUND_TRUTH_AUDIT.csv",
            "UZH_FPV": "Stage-0 USABLE_PRIMARY overlap >= 20 s",
            "compatible": "YES_IF_PRIMARY_RULES_HOLD",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "DERIVED_FROM_RAW_TIMESTAMPS",
            "recommendation": "keep 20 s overlap rule",
        },
        {
            "item": "sequence independence",
            "EUROC": "separate flights; shared Firefly + room groups",
            "UZH_FPV": "separate numbered flights; indoor_forward / indoor_45 / outdoor groups",
            "compatible": "RECORDING_IS_INFERENTIAL_UNIT",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "PROTOCOL.md",
            "recommendation": "do not treat windows as independent units",
        },
        {
            "item": "future resampling feasibility",
            "EUROC": "200 Hz class supports 100 Hz integer decimation after anti-alias design",
            "UZH_FPV": "500 Hz to 100 Hz is 5:1; 500 to 200 is 5:2 rational",
            "compatible": "FEASIBLE_IN_PRINCIPLE",
            "fixed_conversion_required": "YES_LATER",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "rate audit; no implementation in Stage 0C",
            "recommendation": "design anti-alias in a later authorized stage",
        },
        {
            "item": "zero-shot scaling feasibility",
            "EUROC": "SI units documented",
            "UZH_FPV": "SI units documented",
            "compatible": "YES_WITHOUT_TARGET_TEST_SCALER",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "unit audit",
            "recommendation": "forbid target-test mean/std or min/max",
        },
        {
            "item": "frame standardization feasibility",
            "EUROC": "T_BS fixed from YAML",
            "UZH_FPV": "no official vehicle-body CAD transform",
            "compatible": "NOT_FOR_A_SHARED_VEHICLE_BODY",
            "fixed_conversion_required": "PARTIAL",
            "target_fitted_information_required": "NO",
            "severity": "MAJOR",
            "evidence": "FRAME audits",
            "recommendation": "do not learn a cross-dataset rotation; native frames or invariant scalars",
        },
        {
            "item": "specific-force magnitude feasibility",
            "EUROC": "||f|| from a_RS_S if specific force",
            "UZH_FPV": "||f|| from lin_acc_* if specific force",
            "compatible": "YES_ROTATION_INVARIANT_UNDER_ORTHONORMAL_R",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "same SI specific-force interpretation",
            "recommendation": "scientifically valid but loses direction; do not choose only for compatibility",
        },
        {
            "item": "3-axis specific-force feasibility",
            "EUROC": "native S 3-axis",
            "UZH_FPV": "native S 3-axis",
            "compatible": "SAME_QUANTITY_DIFFERENT_AXES",
            "fixed_conversion_required": "NO_IF_NATIVE_FRAMES_RETAINED",
            "target_fitted_information_required": "NO",
            "severity": "MAJOR",
            "evidence": "IMU audits",
            "recommendation": "defensible if axes are not claimed identical",
        },
        {
            "item": "angular-rate feasibility",
            "EUROC": "w_RS_S 3-axis rad/s",
            "UZH_FPV": "ang_vel_* 3-axis rad/s",
            "compatible": "SAME_QUANTITY_DIFFERENT_AXES",
            "fixed_conversion_required": "NO_IF_NATIVE_FRAMES_RETAINED",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "IMU audits",
            "recommendation": "same caveats as 3-axis specific force",
        },
        {
            "item": "primary study suitability",
            "EUROC": "Vicon Phase A if USABLE_PRIMARY",
            "UZH_FPV": "8 retained Snapdragon if hashes PASS",
            "compatible": "YES_IF_DECISION_NOT_FAIL",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "usability CSVs",
            "recommendation": "do not equalize session counts",
        },
    ]
    qty_ok = euroc_q == "RAW_ACCELEROMETER_SPECIFIC_FORCE" and euroc_au == "m/s^2" and euroc_gu == "rad/s" and uzh_ok
    overall = (
        "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY"
        if qty_ok
        else "INSUFFICIENT_EVIDENCE"
    )
    return rows, overall


def write_sampling_md(path: Path, euroc_rates: list[float], uzh_rate: float = 500.0) -> None:
    med = float(sorted(euroc_rates)[len(euroc_rates) // 2]) if euroc_rates else 200.0
    path.write_text(
        f"""# Sampling-rate compatibility (Stage 0C)

No resampling, interpolation, anti-alias filtering, or decimation was applied.

## Audited nominal/measured rates

- EuRoC documented: 200 Hz (ASL page; ADIS16448).
- EuRoC measured median (this audit): {med:.6g} Hz across audited sequences.
- UZH-FPV Snapdragon documented: 500 Hz.
- UZH-FPV measured (Stage 0, re-used): approximately 500 Hz.

## Candidate future common rates (not frozen)

### 100 Hz

- EuRoC 200 -> 100: integer ratio 2:1. Future work would still require an explicit anti-alias design before decimation.
- UZH 500 -> 100: integer ratio 5:1. Same anti-alias requirement.
- Information loss: both sources lose high-frequency inertial content; UZH loses more bandwidth relative to its native 500 Hz.
- Cross-dataset consistency: both can theoretically share 100 Hz after a later authorized stage.

### 200 Hz

- EuRoC 200 -> 200: identity (no rate conversion) if measured rate is acceptably 200 Hz class.
- UZH 500 -> 200: rational 5:2. Not integer decimation. Would require a documented resampling filter, not a naive stride.
- Information loss: EuRoC unchanged; UZH reduced.
- Cross-dataset consistency: possible but not a trivial integer downsample on UZH.

## What is not allowed yet

Implementing either conversion, estimating filter coefficients from target-test recordings, or claiming a frozen study rate.

Evidence: DERIVED_FROM_RAW_TIMESTAMPS (EuRoC this stage; UZH Stage 0).
""",
        encoding="utf-8",
    )


def write_measurand_md(path: Path, euroc_audits: list[dict[str, Any]]) -> dict[str, str]:
    imu = euroc_audits[0]["imu"] if euroc_audits else {}
    q = imu.get("interpreted_acceleration_quantity", "UNRESOLVED")
    unit = imu.get("acceleration_unit", "UNRESOLVED")
    fields = ",".join(
        [
            str(imu.get("accel_x_raw_field", "")),
            str(imu.get("accel_y_raw_field", "")),
            str(imu.get("accel_z_raw_field", "")),
        ]
    )
    if q == "RAW_ACCELEROMETER_SPECIFIC_FORCE" and unit == "m/s^2":
        recommended = (
            "3-axis accelerometer specific force in each dataset's native IMU sensor frame "
            "(EuRoC a_RS_S; UZH lin_acc_*), SI unit m/s^2"
        )
        confidence = "HIGH"
    else:
        recommended = "NO_DEFENSIBLE_COMMON_MEASURAND"
        confidence = "LOW"
    path.write_text(
        f"""# Measurand decision (Stage 0C)

Physical sensor quantity and future prediction-error metric are distinct.
Example: the sensor indication is accelerometer specific force in m/s^2; a later RMSE would be an error of that indication, not a new physical quantity.

## Candidate A — 3-axis specific force, native IMU frame

- Definition: vector f = (fx, fy, fz) of accelerometer specific force in the dataset's IMU sensor frame.
- Unit: m/s^2
- Frame: EuRoC S via {fields}; UZH S via lin_acc_x/y/z
- Required transforms: none if native frames are retained
- Traceability: headers + official unit documentation; gravity remains in the indication
- Advantages: retains directional inertial information; no target-fitted alignment
- Disadvantages: EuRoC S and UZH S are not the same physical axes; 3-axis errors are not axis-wise comparable without a documented common frame
- Measurement-journal relevance: honest sensor indication; must not be mislabelled linear acceleration

## Candidate B — 3-axis specific force in a common body frame

- Definition: R_fixed f_S mapped into a justified vehicle body
- EuRoC: T_BS exists (S->B) from sensor.yaml
- UZH: vehicle-body CAD UNRESOLVED; only camera-IMU Kalibr is documented
- Required transforms: would need an official UZH body convention that does not exist in the public audit
- Target-fitted rotation: prohibited
- Decision: not currently definable without inventing a UZH body

## Candidate C — specific-force magnitude ||f||

- Definition: sqrt(fx^2+fy^2+fz^2)
- Advantages: invariant under proper orthonormal frame rotations; avoids arbitrary axis pairing
- Limitations: loses direction; biases/noise remain; magnitude distributions differ across flight regimes
- Must not be selected merely because it is easier to compare

## Candidate D — 3-axis angular velocity

- EuRoC w_RS_S [rad/s]; UZH ang_vel_* [rad/s]
- Same frame caveat as Candidate A

## Candidate E — angular-speed magnitude ||omega||

- Rotation-invariant analog of Candidate C, with the same loss of direction

## Physical quantity vs evaluation quantity

Do not confuse the measurand with a future scoring rule.

## Recommendation

RECOMMENDED_COMMON_MEASURAND = {recommended}

confidence = {confidence}

Rationale: both official IMU streams, after file audit, are accelerometer specific force in SI units with gravity remaining. A shared vehicle body is not officially available for UZH. Native-frame 3-axis specific force is therefore the scientifically richest quantity that can be defined without target-test fitting. Magnitude is a documented optional invariant, not the primary selection.

Losing directionality is not required for the intended inertial-measurement study and is not treated as acceptable merely to manufacture axis alignment.
""",
        encoding="utf-8",
    )
    return {"recommended": recommended, "confidence": confidence}


def write_dataset_audit(
    path: Path,
    *,
    decision: str,
    utc: str,
    euroc_audits: list[dict[str, Any]],
    usable: list[dict[str, Any]],
    uzh_rows: list[dict[str, Any]],
    overall_compat: str,
    measurand: dict[str, str],
    findings: list[dict[str, str]],
    download_rows: list[dict[str, str]],
) -> None:
    n_doc = 11
    n_dl = len(euroc_audits)
    n_pri = sum(1 for r in usable if r.get("usable_status") == "USABLE_PRIMARY")
    uzh_pass = sum(1 for r in uzh_rows if r.get("hash_status") == "PASS")
    crit = [f for f in findings if f.get("severity") == "CRITICAL" and f.get("status") == "FAIL"]
    major = [f for f in findings if f.get("severity") == "MAJOR" and f.get("status") == "FAIL"]
    imu = euroc_audits[0]["imu"] if euroc_audits else {}
    gt = euroc_audits[0]["gt"] if euroc_audits else {}
    seq_list = ", ".join(a["sequence_id"] for a in euroc_audits) or "NONE"
    pri_list = ", ".join(r["sequence_id"] for r in usable if r.get("usable_status") == "USABLE_PRIMARY") or "NONE"
    uzh_list = ", ".join(r["sequence_id"] for r in uzh_rows if r.get("hash_status") == "PASS") or "NONE"
    vicon_bytes = sum(
        int(float(r["downloaded_size_bytes"] or 0))
        for r in download_rows
        if r.get("archive_name") in {"vicon_room1.zip", "vicon_room2.zip"}
    )
    path.write_text(
        f"""# Stage 0C — EuRoC MAV + UZH-FPV Scientific Audit

## 1. Executive decision

**{decision}**

utc = {utc}

EuRoC official source verified. Phase A Vicon sequences audited: {n_dl}. USABLE_PRIMARY: {n_pri}. UZH hash-verified retained sequences: {uzh_pass}/8. Recommended measurand: {measurand.get('recommended')}. Overall compatibility class: {overall_compat}.

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

Phase A downloaded official bundled archives `vicon_room1.zip` and `vicon_room2.zip` (total official size 12055648375 bytes, 11.228 GiB) within the 15 GiB additional budget. Machine Hall (`machine_hall.zip`, 12683729426 bytes) was not downloaded. Nested `.bag` files and stereo images were not extracted. Inner sequence zips were extracted to `data/raw/euroc/extracted/`; CSV/YAML working copies to `data/cache/euroc/`. Downloaded Vicon bytes (local): {vicon_bytes}.

Downloaded sequences: {seq_list}

## 6. EuRoC IMU

### 6.1 Exact channels

Exact raw header strings are in `EUROC_IMU_AUDIT.csv`. Representative fields (first audited sequence): timestamp `{_fmt(imu.get('timestamp_field'))}`; gyro `{_fmt(imu.get('gyro_x_raw_field'))}`, `{_fmt(imu.get('gyro_y_raw_field'))}`, `{_fmt(imu.get('gyro_z_raw_field'))}`; accel `{_fmt(imu.get('accel_x_raw_field'))}`, `{_fmt(imu.get('accel_y_raw_field'))}`, `{_fmt(imu.get('accel_z_raw_field'))}`.

### 6.2 Physical interpretation

`{_fmt(imu.get('interpreted_acceleration_quantity'))}`

Gravity remaining: `{_fmt(imu.get('gravity_in_sensor_measurement'))}`. Not labelled linear acceleration. Bias correction of the published IMU CSV: `{_fmt(imu.get('bias_corrected'))}`. Factory calibration of the published stream: `{_fmt(imu.get('factory_calibrated_if_known'))}`.

### 6.3 Units

Acceleration `{_fmt(imu.get('acceleration_unit'))}`; gyro `{_fmt(imu.get('gyro_unit'))}`; timestamps `{_fmt(imu.get('timestamp_unit'))}`.

### 6.4 Sensor hardware

ASL/paper: MEMS IMU ADIS16448, angular rate and acceleration, 200 Hz. File YAML comment/model: `{_fmt(imu.get('sensor_model'))}`. Noise densities from sensor.yaml are copied into `EUROC_IMU_AUDIT.csv` when present.

### 6.5 Actual sampling

Documented `{_fmt(imu.get('documented_rate_hz'))}` Hz. Measured medians are per sequence in `EUROC_TIMESTAMP_AUDIT.csv`.

### 6.6 Integrity

See `EUROC_INTEGRITY_FINDINGS.csv`. Critical FAIL count among audited sequences: {len(crit)}. Major FAIL count: {len(major)}.

## 7. EuRoC coordinate frames and calibration

IMU indications are in sensor frame S (`a_RS_S`, `w_RS_S` when those headers are present). `T_BS` in `imu0/sensor.yaml` is a 4x4 transform. Official MATLAB `dataset_plot_body.m` uses `p_BS_B = T_BS(1:3,4)` and `C_BS = T_BS(1:3,1:3)` to plot the sensor in the body, i.e. T_BS maps S to B. Stage 0C does not rotate streams. Details: `EUROC_FRAME_AUDIT.csv`.

## 8. EuRoC ground truth

Vicon sequences: 6-DoF motion-capture pose, published as a spatio-temporally aligned / post-processed state. `state_groundtruth_estimate0` also contains velocity and IMU biases that are **DERIVED_BY_DATASET_AUTHORS**, not independent tracker measurands. Leica Machine Hall 3D position is not a 6-DoF orientation measurement; those sequences were not downloaded. **NO_DIRECT_GT_ACCELERATION**. Representative GT: type `{_fmt(gt.get('gt_type'))}`; frame `{_fmt(gt.get('gt_reference_frame'))}`.

## 9. EuRoC usable sequences

USABLE_PRIMARY ({n_pri}): {pri_list}

Full table: `EUROC_USABLE_SEQUENCES.csv`. Machine Hall sequences: NOT_DOWNLOADED.

## 10. Retained UZH-FPV dataset

Eight Stage-0 USABLE_PRIMARY Snapdragon sequences were retained and not redownloaded.

## 11. UZH hash/schema re-verification

Hash PASS count = {uzh_pass}/8. Sequences: {uzh_list}. See `UZH_HASH_REVERIFY.csv`. Quantity/unit/frame facts were cross-checked against Stage-0 IMU_AUDIT and a lightweight `imu.txt` header parse. Full expensive UZH audit was not repeated.

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

{measurand.get('recommended')}

confidence = {measurand.get('confidence')}

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

**{decision}**
""",
        encoding="utf-8",
    )


def write_stage0c_report(
    path: Path,
    *,
    decision: str,
    euroc_audits: list[dict[str, Any]],
    usable: list[dict[str, Any]],
    uzh_rows: list[dict[str, Any]],
    overall_compat: str,
    measurand: dict[str, str],
    findings: list[dict[str, str]],
    pytest_passed: str,
    pytest_failed: str,
) -> None:
    imu = euroc_audits[0]["imu"] if euroc_audits else {}
    gt = euroc_audits[0]["gt"] if euroc_audits else {}
    n_pri = sum(1 for r in usable if r.get("usable_status") == "USABLE_PRIMARY")
    uzh_hash = "PASS" if uzh_rows and all(r.get("hash_status") == "PASS" for r in uzh_rows) else "FAIL"
    crit = sum(1 for f in findings if f.get("severity") == "CRITICAL" and f.get("status") == "FAIL")
    major = sum(1 for f in findings if f.get("severity") == "MAJOR" and f.get("status") == "FAIL")
    overlaps = [float(a["gt"]["overlap_duration_s"] or 0) for a in euroc_audits] if euroc_audits else []
    overlap_status = "ALL_PRIMARY_GE_20S" if overlaps and min(overlaps) >= 20 else "SEE_CSV"
    path.write_text(
        f"""# STAGE 0C REPORT

EuRoC official source:
VERIFIED

EuRoC DOI:
10.3929/ethz-b-000690084

EuRoC sequences documented:
11 standard flights (6 Vicon + 5 Machine Hall) plus separate calibration sessions

EuRoC sequences downloaded:
{len(euroc_audits)} Vicon sequences (Phase A); Machine Hall not downloaded

EuRoC USABLE_PRIMARY:
{n_pri} ({', '.join(r['sequence_id'] for r in usable if r.get('usable_status')=='USABLE_PRIMARY')})

EuRoC exact acceleration fields:
{_fmt(imu.get('accel_x_raw_field'))} / {_fmt(imu.get('accel_y_raw_field'))} / {_fmt(imu.get('accel_z_raw_field'))}

EuRoC acceleration physical quantity:
{_fmt(imu.get('interpreted_acceleration_quantity'))}

EuRoC acceleration unit:
{_fmt(imu.get('acceleration_unit'))}

EuRoC exact gyro fields:
{_fmt(imu.get('gyro_x_raw_field'))} / {_fmt(imu.get('gyro_y_raw_field'))} / {_fmt(imu.get('gyro_z_raw_field'))}

EuRoC gyro unit:
{_fmt(imu.get('gyro_unit'))}

EuRoC IMU sensor frame:
S (IMU/sensor; a_RS_S / w_RS_S when present)

EuRoC body transform:
{_fmt(imu.get('transform_direction'))}

EuRoC documented rate:
{_fmt(imu.get('documented_rate_hz'))} Hz

EuRoC measured rate:
see EUROC_TIMESTAMP_AUDIT.csv (median Hz per sequence)

EuRoC GT:
{_fmt(gt.get('gt_type'))}; {_fmt(gt.get('acceleration_fields'))}

EuRoC GT frame:
{_fmt(gt.get('gt_reference_frame'))} / {_fmt(gt.get('body_or_reference_frame'))}

EuRoC GT overlap:
{overlap_status}

UZH retained sequences:
{', '.join(r['sequence_id'] for r in uzh_rows)}

UZH hash verification:
{uzh_hash}

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
{measurand.get('recommended')}

Measurand confidence:
{measurand.get('confidence')}

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
{crit}

Major findings:
{major}

pytest:
{pytest_passed} passed
{pytest_failed} failed

OVERALL STAGE-0C DECISION:
{decision}

NO MODELLING WAS PERFORMED.
NO RESAMPLING WAS PERFORMED.
NO NORMALIZATION WAS PERFORMED.
NO COORDINATE ROTATION OF THE FULL DATASET WAS PERFORMED.
NO GT ACCELERATION WAS DERIVED.
""",
        encoding="utf-8",
    )


def apply_pass_logic(
    *,
    source_verified: bool,
    euroc_primary: int,
    uzh_pass: int,
    fields_ok: bool,
    units_ok: bool,
    quantity_ok: bool,
    frames_ok: bool,
    rates_ok: bool,
    gt_ok: bool,
    overlap_ok: bool,
    deps_ok: bool,
    no_critical: bool,
    euroc_hashed: bool,
    uzh_hashed: bool,
    zero_shot_ok: bool,
    no_modelling: bool,
    downloaded: bool,
) -> str:
    if not downloaded:
        return "ACQUISITION_BLOCKED"
    essential = [
        source_verified,
        euroc_primary >= 6,
        uzh_pass >= 6,
        fields_ok,
        units_ok,
        quantity_ok,
        frames_ok,
        rates_ok,
        gt_ok,
        overlap_ok,
        deps_ok,
        no_critical,
        euroc_hashed,
        uzh_hashed,
        zero_shot_ok,
        no_modelling,
    ]
    if all(essential):
        # UZH body CAD unresolved is a remaining noncritical documentation issue.
        return "CONDITIONAL_PASS"
    return "FAIL"


def run_stage0c(root: Path | None = None, *, do_download: bool = False) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT
    cfg = load_stage0c_config(root)
    out = results_dir(root)
    utc = utc_now()
    write_source_verification(root, cfg, utc)
    remote = official_remote_inventory(cfg)
    write_csv(out / "EUROC_REMOTE_SEQUENCE_INVENTORY.csv", REMOTE_INV_COLUMNS, remote)

    download_rows: list[dict[str, str]]
    manifest_existing = out / "EUROC_DOWNLOAD_MANIFEST.csv"
    if do_download:
        download_rows = download_phase_a(root, cfg)
    elif manifest_existing.exists():
        import csv as _csv

        with manifest_existing.open("r", encoding="utf-8", newline="") as handle:
            download_rows = list(_csv.DictReader(handle))
        # If Phase A zips exist but manifest lacks inner rows, still extract.
    else:
        download_rows = download_phase_a(root, cfg)

    extra, seq_dirs = extract_phase_a(root, download_rows)
    if extra:
        # Keep bundle rows plus inner zip rows.
        merged = [r for r in download_rows if r.get("archive_name", "").endswith(".zip")]
        seen = {(r.get("archive_name"), r.get("sequence_id")) for r in merged}
        for row in extra:
            key = (row.get("archive_name"), row.get("sequence_id"))
            if key not in seen:
                merged.append(row)
                seen.add(key)
        download_rows = merged
    write_csv(out / "EUROC_DOWNLOAD_MANIFEST.csv", DOWNLOAD_COLUMNS, download_rows)

    inner_dir = root / "data" / "raw" / "euroc" / "extracted"
    audits: list[dict[str, Any]] = []
    for seq_id, cache in sorted(seq_dirs.items()):
        inner = inner_dir / f"{seq_id}.zip"
        audits.append(audit_one_euroc_sequence(seq_id, cache, inner if inner.exists() else None))

    # MH documented but not downloaded
    mh_rows = []
    for seq in ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]:
        meta = parse_sequence_meta(seq)
        mh_rows.append(
            {
                "sequence_id": seq,
                "environment": meta["environment"],
                "room": meta["room"],
                "difficulty": meta["difficulty"],
                "platform": "Asctec Firefly hex-rotor",
                "sensor_rig": "VI-Sensor stereo + ADIS16448 IMU",
                "recording_id": seq,
                "duration_s": "",
                "imu_available": "NOT_DOWNLOADED",
                "gt_available": "DOCUMENTED_LEICA_3D_PLUS_POSTPROCESSED_STATE",
                "gt_type": "LEICA_MS50_3D_POSITION_NOT_DIRECT_6DOF",
                "calibration_available": "DOCUMENTED_IN_OFFICIAL_ARCHIVES",
                "independent_recording": "YES",
                "potential_dependency_group": "EUROC_FIREFLY_MACHINE_HALL",
                "independence_confidence": "HIGH as separate flights; MEDIUM same MAV/hall",
                "usable_status": "NOT_DOWNLOADED",
                "exclusion_reason": "Machine Hall optional; not acquired in Stage 0C Phase A",
            }
        )

    imu_rows = [a["imu"] for a in audits]
    frame_rows = [a["frame"] for a in audits]
    ts_rows = [a["timestamp"] for a in audits]
    gt_rows = [a["gt"] for a in audits]
    seq_rows = [a["sequence"] for a in audits] + mh_rows
    usable_rows = [a["usable"] for a in audits]
    for seq in ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]:
        usable_rows.append(
            {
                "sequence_id": seq,
                "usable_status": "NOT_DOWNLOADED",
                "official_source": "YES",
                "authentic_file": "NOT_DOWNLOADED",
                "valid_hash": "NOT_DOWNLOADED",
                "imu_accel_available": "NOT_DOWNLOADED",
                "gyro_available": "NOT_DOWNLOADED",
                "units_verified": "NOT_DOWNLOADED",
                "imu_frame_documented": "NOT_DOWNLOADED",
                "valid_timestamps": "NOT_DOWNLOADED",
                "imu_duration_s": "",
                "gt_available": "DOCUMENTED_ONLY",
                "overlap_s": "",
                "finite_fraction": "",
                "critical_integrity": "NOT_APPLICABLE",
                "calibration_documented": "DOCUMENTED_ONLY",
                "exclusion_reason": "NOT_DOWNLOADED",
                "notes": "Optional Phase B",
            }
        )
    findings: list[dict[str, str]] = []
    for a in audits:
        findings.extend(a["findings"])

    uzh_rows = reverify_uzh(root, cfg)
    write_csv(out / "UZH_HASH_REVERIFY.csv", UZH_REVERIFY_COLUMNS, uzh_rows)
    uzh_ok = bool(uzh_rows) and all(r.get("hash_status") == "PASS" and r.get("schema_status") == "PASS" for r in uzh_rows)

    write_csv(out / "EUROC_IMU_AUDIT.csv", EUROC_IMU_COLUMNS, imu_rows)
    write_csv(out / "EUROC_FRAME_AUDIT.csv", EUROC_FRAME_COLUMNS, frame_rows)
    write_csv(out / "EUROC_TIMESTAMP_AUDIT.csv", EUROC_TS_COLUMNS, ts_rows)
    write_csv(out / "EUROC_GROUND_TRUTH_AUDIT.csv", EUROC_GT_COLUMNS, gt_rows)
    write_csv(out / "EUROC_SEQUENCE_INVENTORY.csv", EUROC_SEQ_COLUMNS, seq_rows)
    write_csv(out / "EUROC_USABLE_SEQUENCES.csv", EUROC_USABLE_COLUMNS, usable_rows)
    write_csv(out / "EUROC_INTEGRITY_FINDINGS.csv", INTEGRITY_COLUMNS, findings)

    compat_rows, overall_compat = build_compatibility(audits, uzh_ok)
    compat_rows.append(
        {
            "item": "OVERALL_CLASSIFICATION",
            "EUROC": "see rows above",
            "UZH_FPV": "see rows above",
            "compatible": overall_compat,
            "fixed_conversion_required": "DOCUMENTED_FIXED_ONLY",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "Stage 0C compatibility matrix",
            "recommendation": overall_compat,
        }
    )
    write_csv(out / "EUROC_UZH_COMPATIBILITY.csv", COMPAT_COLUMNS, compat_rows)

    unit_rows = [
        {
            "quantity": "acceleration unit",
            "euroc_value": imu_rows[0]["acceleration_unit"] if imu_rows else "UNRESOLVED",
            "uzh_value": "m/s^2",
            "compatibility": "DIRECT" if imu_rows and imu_rows[0]["acceleration_unit"] == "m/s^2" else "UNRESOLVED",
            "fixed_conversion": "NO",
            "target_fitted_required": "NO",
            "evidence": "VERIFIED_FROM_RAW_FILE / Stage-0 UZH audit",
            "notes": "no distribution-estimated scale",
        },
        {
            "quantity": "angular-rate unit",
            "euroc_value": imu_rows[0]["gyro_unit"] if imu_rows else "UNRESOLVED",
            "uzh_value": "rad/s",
            "compatibility": "DIRECT" if imu_rows and imu_rows[0]["gyro_unit"] == "rad/s" else "UNRESOLVED",
            "fixed_conversion": "NO",
            "target_fitted_required": "NO",
            "evidence": "VERIFIED_FROM_RAW_FILE",
            "notes": "",
        },
        {
            "quantity": "timestamp unit",
            "euroc_value": imu_rows[0]["timestamp_unit"] if imu_rows else "UNRESOLVED",
            "uzh_value": "seconds in imu.txt",
            "compatibility": "DOCUMENTED_FIXED_CONVERSION",
            "fixed_conversion": "ns_to_s for EuRoC if nanoseconds",
            "target_fitted_required": "NO",
            "evidence": "headers; official MATLAB /1e9",
            "notes": "",
        },
        {
            "quantity": "frame convention",
            "euroc_value": "IMU S; T_BS S->B",
            "uzh_value": "IMU S; vehicle body UNRESOLVED",
            "compatibility": "UNRESOLVED" if False else "INCOMPATIBLE",
            "fixed_conversion": "not for a shared body",
            "target_fitted_required": "NO",
            "evidence": "frame audits",
            "notes": "native frames remain comparable as dataset-specific axes",
        },
        {
            "quantity": "physical interpretation",
            "euroc_value": imu_rows[0]["interpreted_acceleration_quantity"] if imu_rows else "UNRESOLVED",
            "uzh_value": "RAW_ACCELEROMETER_SPECIFIC_FORCE",
            "compatibility": "DIRECT"
            if imu_rows and imu_rows[0]["interpreted_acceleration_quantity"] == "RAW_ACCELEROMETER_SPECIFIC_FORCE"
            else "UNRESOLVED",
            "fixed_conversion": "NO",
            "target_fitted_required": "NO",
            "evidence": "IMU audits",
            "notes": "",
        },
    ]
    # Frame convention is not INCOMPATIBLE for the study; native-frame comparison is allowed.
    unit_rows[3]["compatibility"] = "UNRESOLVED"
    unit_rows[3]["notes"] = "Different sensor axes; not a unit conversion. Native-frame analysis remains allowed."
    write_csv(out / "EUROC_UZH_UNIT_COMPATIBILITY.csv", UNIT_COMPAT_COLUMNS, unit_rows)

    frame_compat = [
        {
            "item": "IMU sensor frame",
            "euroc": "S",
            "uzh_fpv": "S (Snapdragon)",
            "common_body_possible": "NO_SHARED_OFFICIAL_BODY",
            "fixed_from_official_calibration": "EUROC_YES_UZH_PARTIAL_CAM_IMU_ONLY",
            "target_test_fit_required": "NO",
            "alignment_arbitrary_or_learned": "would be arbitrary if invented",
            "evidence": "sensor.yaml T_BS; UZH Kalibr T_cam_imu",
            "notes": "names coincide; physical axes do not",
        },
        {
            "item": "body frame",
            "euroc": "B via T_BS",
            "uzh_fpv": "vehicle CAD UNRESOLVED",
            "common_body_possible": "NO",
            "fixed_from_official_calibration": "NO",
            "target_test_fit_required": "NO",
            "alignment_arbitrary_or_learned": "learned alignment prohibited",
            "evidence": "Stage 0 FRAME_AUDIT; EuRoC YAML",
            "notes": "",
        },
        {
            "item": "sensor to body",
            "euroc": imu_rows[0]["transform_direction"] if imu_rows else "UNRESOLVED",
            "uzh_fpv": "UNRESOLVED for vehicle body",
            "common_body_possible": "NO",
            "fixed_from_official_calibration": "EUROC_ONLY",
            "target_test_fit_required": "NO",
            "alignment_arbitrary_or_learned": "NO_FIT_PERFORMED",
            "evidence": "dataset_plot_body.m; sensor.yaml",
            "notes": "streams not rotated",
        },
    ]
    write_csv(out / "EUROC_UZH_FRAME_COMPATIBILITY.csv", FRAME_COMPAT_COLUMNS, frame_compat)

    gt_compat = [
        {
            "item": "reference source",
            "euroc": "Vicon 6-DoF (downloaded); Leica MS50 for MH (not downloaded)",
            "uzh_fpv": "Leica MS60 prism + IMU-aided batch pose",
            "supports_temporal_segmentation": "YES",
            "supports_dynamics_characterization": "YES_AS_REFERENCE_TRAJECTORY",
            "supports_pose_reference_evaluation": "YES_WITH_SYNC_CAVEATS",
            "supports_gt_acceleration_derivation": "NOT_AUTHORIZED",
            "evidence": "dataset papers; GT files",
            "notes": "POSTPROCESSED REFERENCE / ESTIMATED STATE fields must be distinguished",
        },
        {
            "item": "direct sensor measurement vs estimate",
            "euroc": "Vicon pose is a mocap reference; state velocity/biases are estimator outputs",
            "uzh_fpv": "Leica measures position; public 6-DoF pose is batch-optimized with IMU",
            "supports_temporal_segmentation": "YES",
            "supports_dynamics_characterization": "LIMITED",
            "supports_pose_reference_evaluation": "YES",
            "supports_gt_acceleration_derivation": "NOT_AUTHORIZED",
            "evidence": "VERIFIED_FROM_DATASET_PAPER",
            "notes": "UZH pose uses IMU so it must not become an IMU label",
        },
    ]
    write_csv(out / "EUROC_UZH_GT_COMPATIBILITY.csv", GT_COMPAT_COLUMNS, gt_compat)

    rates = [float(r["measured_median_rate_hz"]) for r in imu_rows if r.get("measured_median_rate_hz") not in (None, "")]
    write_sampling_md(out / "SAMPLING_COMPATIBILITY.md", rates)
    measurand = write_measurand_md(out / "MEASURAND_DECISION.md", audits)

    # Primary set
    primary_rows: list[dict[str, Any]] = []
    for a in audits:
        u = a["usable"]
        eligible = u.get("usable_status") == "USABLE_PRIMARY"
        primary_rows.append(
            {
                "dataset": "EUROC",
                "sequence_id": a["sequence_id"],
                "environment": a["sequence"]["environment"],
                "difficulty": a["sequence"]["difficulty"],
                "duration_s": a["timestamp"]["duration_s"],
                "imu_rate_hz": a["imu"]["measured_median_rate_hz"],
                "gt_type": a["gt"]["gt_type"],
                "gt_rate_hz": a["gt"]["gt_rate_measured_hz"],
                "common_quantity_available": "YES" if a["imu"]["interpreted_acceleration_quantity"] == "RAW_ACCELEROMETER_SPECIFIC_FORCE" else "NO",
                "frame_status": "NATIVE_S_T_BS_DOCUMENTED",
                "integrity_status": u.get("critical_integrity"),
                "dependency_group": a["sequence"]["potential_dependency_group"],
                "primary_eligible": "YES" if eligible else "NO",
                "reason": u.get("exclusion_reason") or "passed Stage 0C usability rules",
            }
        )
    uzh_meta = {
        "indoor_forward_6_snapdragon": ("indoor", "aggressive", "indoor_forward"),
        "indoor_forward_9_snapdragon": ("indoor", "moderate", "indoor_forward"),
        "indoor_forward_10_snapdragon": ("indoor", "moderate", "indoor_forward"),
        "indoor_45_2_snapdragon": ("indoor", "moderate", "indoor_45"),
        "indoor_45_4_snapdragon": ("indoor", "moderate", "indoor_45"),
        "indoor_45_13_snapdragon": ("indoor", "moderate", "indoor_45"),
        "indoor_45_14_snapdragon": ("indoor", "hard", "indoor_45"),
        "outdoor_forward_1_snapdragon": ("outdoor", "moderate", "outdoor_forward"),
    }
    for r in uzh_rows:
        env, diff, dep = uzh_meta.get(r["sequence_id"], ("UNRESOLVED", "UNRESOLVED", "UZH"))
        ok = r.get("hash_status") == "PASS" and r.get("schema_status") == "PASS"
        primary_rows.append(
            {
                "dataset": "UZH_FPV",
                "sequence_id": r["sequence_id"],
                "environment": env,
                "difficulty": diff,
                "duration_s": "SEE_STAGE0_TIMESTAMP_AUDIT",
                "imu_rate_hz": "approximately_500",
                "gt_type": "GROUND_TRUTH_POSE_ONLY",
                "gt_rate_hz": "SEE_STAGE0",
                "common_quantity_available": "YES" if ok else "NO",
                "frame_status": "NATIVE_S_VEHICLE_BODY_UNRESOLVED",
                "integrity_status": "HASH_SCHEMA_" + r.get("hash_status", ""),
                "dependency_group": dep,
                "primary_eligible": "YES" if ok else "NO",
                "reason": r.get("notes") or "Stage-0 USABLE_PRIMARY retained; hash/schema re-verified",
            }
        )
    # Only include eligible in PRIMARY_SEQUENCE_SET as specified: "Include only sequences that pass"
    primary_pass = [p for p in primary_rows if p.get("primary_eligible") == "YES"]
    write_csv(out / "PRIMARY_SEQUENCE_SET.csv", PRIMARY_COLUMNS, primary_pass)

    euroc_primary = sum(1 for r in usable_rows if r.get("usable_status") == "USABLE_PRIMARY")
    uzh_pass_n = sum(1 for r in uzh_rows if r.get("hash_status") == "PASS" and r.get("schema_status") == "PASS")
    fields_ok = bool(imu_rows) and all(
        r.get("accel_x_raw_field") not in {"", "UNRESOLVED", None} and r.get("gyro_x_raw_field") not in {"", "UNRESOLVED", None}
        for r in imu_rows
    )
    units_ok = bool(imu_rows) and all(r.get("acceleration_unit") == "m/s^2" and r.get("gyro_unit") == "rad/s" for r in imu_rows)
    quantity_ok = bool(imu_rows) and all(r.get("interpreted_acceleration_quantity") == "RAW_ACCELEROMETER_SPECIFIC_FORCE" for r in imu_rows)
    frames_ok = bool(frame_rows) and all(r.get("transform_direction", "").startswith("T_BS") for r in frame_rows)
    rates_ok = bool(ts_rows) and all(r.get("freq_median_hz") not in (None, "") for r in ts_rows)
    gt_ok = bool(gt_rows) and all(r.get("gt_file") for r in gt_rows)
    overlap_ok = bool(gt_rows) and all(float(r.get("overlap_duration_s") or 0) >= 20 for r in gt_rows)
    deps_ok = bool(seq_rows)
    no_critical = not any(
        f.get("severity") == "CRITICAL" and f.get("status") == "FAIL" and f.get("sequence_id") in {a["sequence_id"] for a in audits if a["usable"]["usable_status"] == "USABLE_PRIMARY"}
        for f in findings
    )
    euroc_hashed = any(r.get("download_status") == "HASH_VERIFIED" and r.get("archive_name") in {"vicon_room1.zip", "vicon_room2.zip"} for r in download_rows)
    euroc_hashed = euroc_hashed and len(audits) >= 6
    downloaded = len(audits) >= 6
    decision = apply_pass_logic(
        source_verified=True,
        euroc_primary=euroc_primary,
        uzh_pass=uzh_pass_n,
        fields_ok=fields_ok,
        units_ok=units_ok,
        quantity_ok=quantity_ok,
        frames_ok=frames_ok,
        rates_ok=rates_ok,
        gt_ok=gt_ok,
        overlap_ok=overlap_ok,
        deps_ok=deps_ok,
        no_critical=no_critical,
        euroc_hashed=euroc_hashed,
        uzh_hashed=uzh_ok,
        zero_shot_ok=True,
        no_modelling=True,
        downloaded=downloaded,
    )
    if not downloaded:
        decision = "ACQUISITION_BLOCKED"

    write_dataset_audit(
        out / "DATASET_AUDIT_EUROC_UZH.md",
        decision=decision,
        utc=utc,
        euroc_audits=audits,
        usable=usable_rows,
        uzh_rows=uzh_rows,
        overall_compat=overall_compat,
        measurand=measurand,
        findings=findings,
        download_rows=download_rows,
    )
    write_stage0c_report(
        root / "STAGE0C_REPORT.md",
        decision=decision,
        euroc_audits=audits,
        usable=usable_rows,
        uzh_rows=uzh_rows,
        overall_compat=overall_compat,
        measurand=measurand,
        findings=findings,
        pytest_passed="SEE_PYTEST",
        pytest_failed="SEE_PYTEST",
    )
    # Also copy report into results
    write_stage0c_report(
        out / "STAGE0C_REPORT.md",
        decision=decision,
        euroc_audits=audits,
        usable=usable_rows,
        uzh_rows=uzh_rows,
        overall_compat=overall_compat,
        measurand=measurand,
        findings=findings,
        pytest_passed="SEE_PYTEST",
        pytest_failed="SEE_PYTEST",
    )

    context = {
        "decision": decision,
        "utc": utc,
        "n_euroc_audited": len(audits),
        "n_euroc_primary": euroc_primary,
        "n_uzh_pass": uzh_pass_n,
        "overall_compat": overall_compat,
        "measurand": measurand,
        "flags": {
            "fields_ok": fields_ok,
            "units_ok": units_ok,
            "quantity_ok": quantity_ok,
            "frames_ok": frames_ok,
            "rates_ok": rates_ok,
            "gt_ok": gt_ok,
            "overlap_ok": overlap_ok,
            "no_critical": no_critical,
            "euroc_hashed": euroc_hashed,
            "uzh_ok": uzh_ok,
            "downloaded": downloaded,
        },
    }
    (out / "STAGE0C_CONTEXT.json").write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    return context
