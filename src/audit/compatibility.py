"""Cross-dataset compatibility. No coordinate conversion is executed."""

from __future__ import annotations

from typing import Any

COMPAT_COLUMNS = [
    "item",
    "blackbird",
    "uzh_fpv",
    "compatible",
    "conversion_required",
    "conversion_documented",
    "severity_if_unresolved",
    "scientific_interpretation",
]


def compatibility_table(ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    bb_imu = ctx.get("blackbird_imu_quantity", "UNRESOLVED")
    uzh_imu = ctx.get("uzh_imu_quantity", "UNRESOLVED")
    bb_unit = ctx.get("blackbird_accel_unit", "UNRESOLVED")
    uzh_unit = ctx.get("uzh_accel_unit", "UNRESOLVED")
    bb_frame = ctx.get("blackbird_imu_frame", "imu (settings.json)")
    uzh_frame = ctx.get("uzh_imu_frame", "S /snappy_imu")
    bb_gt = ctx.get("blackbird_gt", "OptiTrack 6D pose (documented)")
    uzh_gt = ctx.get("uzh_gt", "Leica-position + IMU batch-optimized pose")
    bb_dl = int(ctx.get("blackbird_downloaded", 0))
    uzh_dl = int(ctx.get("uzh_downloaded", 0))

    same_quantity = bb_imu == uzh_imu and "UNRESOLVED" not in (bb_imu, uzh_imu)
    units_ok = bb_unit == uzh_unit == "m/s^2"

    rows = [
        {
            "item": "accelerometer_information_present",
            "blackbird": "YES documented (Xsens MTi-3, /blackbird/imu)" if True else "YES",
            "uzh_fpv": "YES documented (Snapdragon /snappy_imu and DAVIS /dvs/imu)",
            "compatible": "YES",
            "conversion_required": "NO",
            "conversion_documented": "n/a",
            "severity_if_unresolved": "INFO",
            "scientific_interpretation": "Both datasets record an IMU accelerometer stream.",
        },
        {
            "item": "acceleration_quantity_physically_the_same",
            "blackbird": bb_imu,
            "uzh_fpv": uzh_imu,
            "compatible": "YES" if same_quantity else "UNRESOLVED",
            "conversion_required": "UNRESOLVED" if not same_quantity else "NO",
            "conversion_documented": "NO" if not same_quantity else "n/a",
            "severity_if_unresolved": "MAJOR",
            "scientific_interpretation": (
                "Do not treat ROS linear_acceleration as gravity-free kinematic acceleration. "
                "Kalibr + rest-period evidence may support specific force, but Stage 0 does not "
                "relabel without file-level confirmation."
            ),
        },
        {
            "item": "si_acceleration_units",
            "blackbird": bb_unit,
            "uzh_fpv": uzh_unit,
            "compatible": "YES" if units_ok else "UNRESOLVED",
            "conversion_required": "YES" if (bb_unit != uzh_unit and "UNRESOLVED" not in (bb_unit, uzh_unit)) else "UNRESOLVED",
            "conversion_documented": "NO conversion executed in Stage 0",
            "severity_if_unresolved": "MAJOR",
            "scientific_interpretation": "Future comparison requires SI m/s^2. No conversion was applied now.",
        },
        {
            "item": "sensor_frame_measurements",
            "blackbird": bb_frame,
            "uzh_fpv": uzh_frame,
            "compatible": "YES_BOTH_SENSOR_FRAMES",
            "conversion_required": "YES for a common body convention later",
            "conversion_documented": "NO official Blackbird↔UZH extrinsic",
            "severity_if_unresolved": "MAJOR",
            "scientific_interpretation": "Each IMU is in its own sensor frame. There is no official cross-dataset body convention.",
        },
        {
            "item": "sensor_to_body_extrinsics_known",
            "blackbird": "FRAME_RELATION_UNRESOLVED (commented TF not used; Kalibr file on downed host)",
            "uzh_fpv": "Kalibr T_cam_imu available for Snapdragon; IMU-to-vehicle-body CAD not separately published",
            "compatible": "NO",
            "conversion_required": "YES to a researcher-defined body",
            "conversion_documented": "PARTIAL_UZH_ONLY",
            "severity_if_unresolved": "MAJOR",
            "scientific_interpretation": "A later common body frame would be a convention, not a unique official transform.",
        },
        {
            "item": "ground_truth_sufficient",
            "blackbird": bb_gt + (f"; downloaded={bb_dl}" ),
            "uzh_fpv": uzh_gt + (f"; downloaded={uzh_dl}"),
            "compatible": "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY",
            "conversion_required": "YES conceptually (different GT generation)",
            "conversion_documented": "YES as documentation, not as a numeric transform",
            "severity_if_unresolved": "MAJOR",
            "scientific_interpretation": (
                "Blackbird GT is independent OptiTrack pose. UZH public 6-DoF GT uses IMU in the estimator. "
                "Neither provides ground-truth acceleration. Pose-only: GROUND_TRUTH_POSE_ONLY."
            ),
        },
        {
            "item": "timestamp_quality",
            "blackbird": "documented 100 Hz IMU, 360 Hz GT, ±5 ms sync; not measured from files if undownloaded",
            "uzh_fpv": "measured from downloaded timestamps when files exist; paper notes Leica dropouts",
            "compatible": "YES_IF_FILES_MEASURED",
            "conversion_required": "NO resampling in Stage 0",
            "conversion_documented": "n/a",
            "severity_if_unresolved": "WARNING",
            "scientific_interpretation": "Irregular sampling is reported, not interpolated.",
        },
        {
            "item": "sample_rate_bandwidth_known",
            "blackbird": "documented 100 Hz IMU (Xsens MTi-3); measured only if downloaded",
            "uzh_fpv": "documented 500 Hz Snapdragon / 1000 Hz DAVIS; measured from files when downloaded",
            "compatible": "RATES_DIFFER",
            "conversion_required": "YES if a later common rate is imposed",
            "conversion_documented": "NO rate conversion in Stage 0",
            "severity_if_unresolved": "INFO",
            "scientific_interpretation": "Different native rates are expected. Resampling is forbidden in Stage 0.",
        },
        {
            "item": "zero_shot_without_fitting_target_transform",
            "blackbird": "physical IMU in imu frame",
            "uzh_fpv": "physical IMU in S frame",
            "compatible": "UNRESOLVED",
            "conversion_required": "YES unless comparing native sensor-frame channels with explicit caveats",
            "conversion_documented": "NO fitted transform permitted on target-test data",
            "severity_if_unresolved": "MAJOR",
            "scientific_interpretation": (
                "A future zero-shot transfer cannot fit a frame/unit warp on the target test sequence. "
                "Only a pre-declared, documented convention would be defensible."
            ),
        },
    ]

    if bb_dl < 6:
        classification = "INSUFFICIENT_EVIDENCE"
        if report_blocked := True:
            classification = "INSUFFICIENT_EVIDENCE"
    elif not same_quantity or not units_ok:
        classification = "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY"
    else:
        classification = "COMPARABLE_AFTER_DOCUMENTED_FRAME_TRANSFORM"

    if bb_dl == 0:
        classification = "INSUFFICIENT_EVIDENCE"

    rows.append(
        {
            "item": "FINAL_CLASSIFICATION",
            "blackbird": f"downloaded_sequences={bb_dl}",
            "uzh_fpv": f"downloaded_sequences={uzh_dl}",
            "compatible": classification,
            "conversion_required": "NOT_EXECUTED",
            "conversion_documented": "see FRAME_AUDIT.md",
            "severity_if_unresolved": "CRITICAL" if classification == "INSUFFICIENT_EVIDENCE" else "MAJOR",
            "scientific_interpretation": (
                "Allowed labels: DIRECTLY_COMPARABLE / COMPARABLE_AFTER_DOCUMENTED_FRAME_TRANSFORM / "
                "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY / NOT_CURRENTLY_COMPARABLE / INSUFFICIENT_EVIDENCE. "
                f"Selected: {classification}."
            ),
        }
    )
    return rows, classification
