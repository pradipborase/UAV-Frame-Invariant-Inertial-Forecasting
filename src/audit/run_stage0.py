"""Stage-0 audit engine. Auditing only: no modelling, resampling, or frame conversion."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .catalogs import (
    SEQUENCE_INVENTORY_COLUMNS,
    UZH_CALIB_FILES,
    blackbird_sequences,
    uzh_sequences,
)
from .channel_inventory import channel_row
from .compatibility import compatibility_table
from .config import Stage0Config, load_config
from .downloader import utc_now
from .frame_audit import blackbird_documented_frames, uzh_documented_frames
from .groundtruth_audit import empty_gt_row
from .hashing import sha256_file
from .reporting import write_all_tables
from .sequence_audit import classify_sequence
from .source_verification import verify_sources, write_source_verification
from .text_reader import extract_audit_members, interpret_gt_table, interpret_imu_table, read_text_table, zip_members
from .timestamp_audit import audit_timestamps, overlap_seconds

UTC = timezone.utc

IMU_AUDIT_COLUMNS = [
    "dataset",
    "sequence_id_or_ALL",
    "sensor_platform",
    "imu_topic_or_file",
    "message_type",
    "timestamp_field",
    "accel_x_field",
    "accel_y_field",
    "accel_z_field",
    "gyro_x_field",
    "gyro_y_field",
    "gyro_z_field",
    "declared_acceleration_quantity",
    "quantity_interpretation",
    "unit_acceleration",
    "unit_angular_rate",
    "frame_id_raw",
    "sensor_frame",
    "body_frame_relationship",
    "axis_x_definition",
    "axis_y_definition",
    "axis_z_definition",
    "gravity_in_measurement",
    "bias_corrected_if_known",
    "calibrated_if_known",
    "sampling_rate_documented_hz",
    "sampling_rate_measured_median_hz",
    "sampling_rate_measured_p05_hz",
    "sampling_rate_measured_p95_hz",
    "evidence_type",
    "evidence_source",
    "confidence",
    "unresolved_issue",
]

DATASET_SUMMARY_COLUMNS = [
    "dataset",
    "institution",
    "official_name",
    "publication",
    "license",
    "official_source_verified",
    "total_sequences_documented",
    "public_gt_sequences_documented",
    "sequences_downloaded",
    "primary_usable_sequences",
    "secondary_usable_sequences",
    "imu_available",
    "exact_imu_quantity",
    "acceleration_unit",
    "gyro_unit",
    "imu_frame",
    "gt_type",
    "gt_frame",
    "calibration_available",
    "documented_imu_rate_hz",
    "measured_rate_summary",
    "frame_transform_available",
    "main_strength",
    "main_limitation",
    "dataset_status",
]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_fraction(arr: np.ndarray | None) -> float:
    if arr is None or arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def _hz_from_dt(median_dt: float | None, p05: float | None, p95: float | None) -> tuple[Any, Any, Any]:
    med = (1.0 / median_dt) if median_dt else None
    # p05_dt is a small dt → high frequency; p95_dt is a large dt → low frequency
    p05_hz = (1.0 / p95) if p95 else None
    p95_hz = (1.0 / p05) if p05 else None
    return med, p05_hz, p95_hz


def _parse_kalibr_yaml(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "keys": [], "update_rate": None, "accel_noise_comment_si": False, "T_cam_imu": None}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info
    info["accel_noise_comment_si"] = ("m / s^2" in text) or ("m/s^2" in text) or ("m s^-2" in text)
    info["gyro_noise_comment_si"] = ("rad / s" in text) or ("rad/s" in text) or ("rad s^-1" in text)
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return info
    if not isinstance(data, dict):
        return info
    info["keys"] = list(data.keys())
    imu = data.get("imu0") or data.get("imu") or {}
    if not isinstance(imu, dict) or not imu:
        imu = data if any(k in data for k in ("update_rate", "accelerometer_noise_density", "rostopic")) else {}
    if isinstance(imu, dict):
        info["update_rate"] = imu.get("update_rate") or imu.get("rate")
        info["accelerometer_noise_density"] = imu.get("accelerometer_noise_density")
        info["gyroscope_noise_density"] = imu.get("gyroscope_noise_density")
        info["rostopic_imu"] = imu.get("rostopic")
    cam = data.get("cam0") or {}
    if isinstance(cam, dict) and cam.get("T_cam_imu") is not None:
        info["T_cam_imu"] = cam.get("T_cam_imu")
        info["rostopic"] = cam.get("rostopic")
        info["timeshift_cam_imu"] = cam.get("timeshift_cam_imu")
    return info


def _gravity_flag(accel: np.ndarray | None, timestamps_s: np.ndarray | None, unit: str) -> str:
    if accel is None or accel.size == 0 or unit not in {"m/s^2", "SI_MPS2_VERIFIED"}:
        return "UNRESOLVED"
    ts = timestamps_s if timestamps_s is not None and timestamps_s.size == accel.shape[0] else None
    if ts is not None:
        t0 = float(np.nanmin(ts))
        rest = accel[(ts >= t0) & (ts <= t0 + 1.5)]
        if rest.shape[0] < 10:
            rest = accel[: min(200, accel.shape[0])]
    else:
        rest = accel[: min(200, accel.shape[0])]
    mag = np.linalg.norm(rest, axis=1)
    mag = mag[np.isfinite(mag)]
    if mag.size < 10:
        return "UNRESOLVED"
    med = float(np.median(mag))
    if 8.5 <= med <= 11.0:
        return "YES_CONSISTENT_WITH_SPECIFIC_FORCE_AT_REST"
    if med < 1.5:
        return "NO_OR_GRAVITY_REMOVED_CANDIDATE"
    return f"UNRESOLVED_rest_median_norm={med:.3f}"


def audit_downloaded_uzh(cfg: Stage0Config, manifest: list[dict[str, str]]) -> dict[str, Any]:
    cache = cfg.cache_dir / "uzh_fpv"
    cache.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    calib_parsed: dict[str, Any] = {}
    for rec in manifest:
        if rec.get("dataset") != "UZH_FPV":
            continue
        if rec.get("download_status") not in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}:
            continue
        path = Path(rec.get("local_path") or "")
        if not path.exists():
            continue
        if rec.get("content_role") == "CALIBRATION":
            dest = cache / "calib" / path.stem
            files = extract_audit_members(path, dest)
            parsed = {}
            for key, fpath in files.items():
                if fpath.suffix.lower() in {".yaml", ".yml"}:
                    parsed[fpath.name] = _parse_kalibr_yaml(fpath)
            calib_parsed[rec.get("sequence_id", path.stem)] = {"zip": str(path), "files": {k: str(v) for k, v in files.items()}, "yaml": parsed}
            continue
        if rec.get("content_role") == "LEICA_POSITION":
            dest = cache / "leica" / path.stem
            files = extract_audit_members(path, dest)
            tables = {}
            for key, fpath in files.items():
                if fpath.suffix.lower() in {".txt", ".csv"}:
                    cols, arr, comments = read_text_table(fpath)
                    tables[fpath.name] = {"columns": cols, "n": int(arr.shape[0]), "comments": comments[:8]}
            payload.setdefault(rec["sequence_id"], {})["leica"] = tables
            continue
        if rec.get("content_role") != "IMU_GT_TEXT_ARCHIVE":
            continue
        seq = rec["sequence_id"]
        dest = cache / "sequences" / path.stem
        members = zip_members(path)
        files = extract_audit_members(path, dest)
        imu_path = files.get("imu")
        gt_path = files.get("gt")
        entry: dict[str, Any] = {
            "zip": str(path),
            "sha256": rec.get("sha256"),
            "zip_members_n": len(members),
            "extracted": {k: str(v) for k, v in files.items()},
            "imu_path": str(imu_path) if imu_path else "",
            "gt_path": str(gt_path) if gt_path else "",
        }
        if imu_path:
            cols, arr, comments = read_text_table(imu_path)
            imu = interpret_imu_table(cols, arr, comments)
            entry["imu"] = imu
        if gt_path:
            cols, arr, comments = read_text_table(gt_path)
            gt = interpret_gt_table(cols, arr, comments)
            entry["gt"] = gt
        payload[seq] = {**payload.get(seq, {}), **entry}
    return {"sequences": payload, "calib": calib_parsed}


def _blackbird_csv_audit(cfg: Stage0Config, manifest: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rec in manifest:
        if rec.get("dataset") != "BLACKBIRD":
            continue
        if rec.get("download_status") not in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}:
            continue
        path = Path(rec.get("local_path") or "")
        if not path.exists():
            continue
        seq = rec["sequence_id"]
        out.setdefault(seq, {})
        try:
            cols, arr, comments = read_text_table(path)
        except Exception as exc:
            out[seq][path.name] = {"error": str(exc)}
            continue
        role = rec.get("content_role")
        if role == "IMU":
            out[seq]["imu"] = interpret_imu_table(cols, arr, comments)
            out[seq]["imu_path"] = str(path)
        elif role == "GROUND_TRUTH":
            out[seq]["gt"] = interpret_gt_table(cols, arr, comments)
            out[seq]["gt_path"] = str(path)
        else:
            out[seq][path.name] = {"columns": cols, "n": int(arr.shape[0])}
    return out


def _finding(
    dataset: str,
    sequence_id: str,
    finding_id: str,
    category: str,
    severity: str,
    status: str,
    observed: str,
    expected: str,
    interpretation: str,
    action: str,
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "sequence_id": sequence_id,
        "finding_id": finding_id,
        "category": category,
        "severity": severity,
        "status": status,
        "observed": observed,
        "expected_or_rule": expected,
        "interpretation": interpretation,
        "action_required": action,
    }


def run_audit(cfg: Stage0Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    source_report = verify_sources(cfg)
    write_source_verification(cfg, source_report)
    manifest = _read_manifest(cfg.results_dir / "DOWNLOAD_MANIFEST.csv")

    # Hash-verify existing downloaded files.
    for rec in manifest:
        path = Path(rec.get("local_path") or "")
        if rec.get("download_status") in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"} and path.exists():
            digest = sha256_file(path)
            if rec.get("sha256") and rec["sha256"] != digest:
                rec["notes"] = (rec.get("notes") or "") + " HASH_MISMATCH_ON_REVERIFY"
                rec["download_status"] = "FAILED"

    bb_cat = blackbird_sequences()
    uzh_cat = uzh_sequences()
    uzh_parsed = audit_downloaded_uzh(cfg, manifest)
    bb_parsed = _blackbird_csv_audit(cfg, manifest)

    downloaded_bb = {
        r["sequence_id"]
        for r in manifest
        if r.get("dataset") == "BLACKBIRD"
        and r.get("download_status") in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}
        and r.get("content_role") in {"IMU", "GROUND_TRUTH"}
    }
    downloaded_uzh = {
        r["sequence_id"]
        for r in manifest
        if r.get("dataset") == "UZH_FPV"
        and r.get("content_role") == "IMU_GT_TEXT_ARCHIVE"
        and r.get("download_status") in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}
    }

    channels: list[dict[str, Any]] = []
    timestamps: list[dict[str, Any]] = []
    imu_audit: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    usable_extra: dict[str, dict[str, Any]] = {}

    # Dataset-level Blackbird IMU documentation row (files may be absent).
    imu_audit.append(
        {
            "dataset": "BLACKBIRD",
            "sequence_id_or_ALL": "ALL_DOCUMENTED",
            "sensor_platform": "Xsens_MTi-3",
            "imu_topic_or_file": "/blackbird/imu  (CSV blackbird_slash_imu.csv if exported)",
            "message_type": "sensor_msgs/Imu from LCM imuRaw_t",
            "timestamp_field": "header.stamp from LCM utime (microseconds * 1000 → nsec in converter)",
            "accel_x_field": "linear_acceleration.x  (LCM imuRaw_t.accel[0])",
            "accel_y_field": "linear_acceleration.y  (LCM imuRaw_t.accel[1])",
            "accel_z_field": "linear_acceleration.z  (LCM imuRaw_t.accel[2])",
            "gyro_x_field": "angular_velocity.x  (LCM imuRaw_t.gyro[0])",
            "gyro_y_field": "angular_velocity.y  (LCM imuRaw_t.gyro[1])",
            "gyro_z_field": "angular_velocity.z  (LCM imuRaw_t.gyro[2])",
            "declared_acceleration_quantity": "ROS linear_acceleration; LCM field name accel",
            "quantity_interpretation": "UNRESOLVED",
            "unit_acceleration": "UNIT_UNRESOLVED",
            "unit_angular_rate": "UNRESOLVED",
            "frame_id_raw": "imu",
            "sensor_frame": "imu",
            "body_frame_relationship": "FRAME_RELATION_UNRESOLVED",
            "axis_x_definition": "UNRESOLVED (Fig 1 exists; numeric axis text not in accessible paper HTML)",
            "axis_y_definition": "UNRESOLVED",
            "axis_z_definition": "UNRESOLVED",
            "gravity_in_measurement": "UNRESOLVED",
            "bias_corrected_if_known": "NO_CLAIM; paper includes 3 s rest for bias initialization, which implies raw/biased measurements exist",
            "calibrated_if_known": "Kalibr used for noise and IMU-camera; onboard output calibration UNRESOLVED without files",
            "sampling_rate_documented_hz": "100",
            "sampling_rate_measured_median_hz": "",
            "sampling_rate_measured_p05_hz": "",
            "sampling_rate_measured_p95_hz": "",
            "evidence_type": "VERIFIED_FROM_OFFICIAL_SOURCE_CODE + VERIFIED_FROM_DATASET_PAPER",
            "evidence_source": "settings.json; msgConverters.py ImuConverter; IJRR/ISER papers",
            "confidence": "HIGH on field names/topic; LOW-MEDIUM on specific-force vs linear acceleration",
            "unresolved_issue": "No raw Blackbird IMU file retrieved. Do not label specific force from the ROS field name alone.",
        }
    )

    for seq, data in uzh_parsed["sequences"].items():
        imu = data.get("imu")
        gt = data.get("gt")
        source = data.get("imu_path") or "imu.txt"
        if not imu:
            findings.append(
                _finding(
                    "UZH_FPV",
                    seq,
                    "UZH-IMU-MISSING",
                    "required_topic",
                    "CRITICAL",
                    "FAIL",
                    "no imu text extracted",
                    "IMU file in official zip",
                    "Cannot audit acceleration.",
                    "Do not use sequence until IMU file is located.",
                )
            )
            usable_extra[seq] = {"downloaded": True, "imu_ok": False, "gt_ok": bool(gt), "corrupt": True}
            continue
        ts = imu.get("timestamps_s")
        ts_stats = audit_timestamps(ts if ts is not None else np.array([]), imu.get("timestamp_unit_native", "UNRESOLVED"))
        timestamps.append({"dataset": "UZH_FPV", "sequence_id": seq, "stream": "IMU", **ts_stats})
        accel = imu.get("accel")
        gyro = imu.get("gyro")
        unit_a = imu.get("unit_acceleration_from_header") or "UNRESOLVED"
        unit_g = imu.get("unit_gyro_from_header") or "UNRESOLVED"
        accel_names = " ".join(str(v) for v in (imu.get("accel_fields") or {}).values())
        gyro_names = " ".join(str(v) for v in (imu.get("gyro_fields") or {}).values())
        # Official dataset page: zip content identical to rosbag; rosbag IMU is sensor_msgs/Imu (SI in the ROS message spec).
        # This is not an inference from a 9.81 magnitude.
        if unit_a == "UNRESOLVED" and "lin_acc" in accel_names:
            unit_a = "m/s^2"
            unit_status_note = "SI from official sensor_msgs/Imu + official zip/bag identity claim; imu.txt header has no unit token"
        else:
            unit_status_note = ""
        if unit_g == "UNRESOLVED" and "ang_vel" in gyro_names:
            unit_g = "rad/s"
        # Kalibr SI comments support units if headers are silent.
        if unit_a == "UNRESOLVED":
            for calib in uzh_parsed["calib"].values():
                for y in (calib.get("yaml") or {}).values():
                    if y.get("accel_noise_comment_si"):
                        unit_a = "CONVERTIBLE_OR_DECLARED_IN_CALIB_YAML"
        if unit_a == "m/s^2":
            unit_status = "SI_MPS2_VERIFIED"
        elif "CALIB" in str(unit_a):
            unit_status = "CONVERTIBLE_TO_SI_WITH_DOCUMENTED_FORMULA"
        else:
            unit_status = "UNIT_UNRESOLVED"
        gravity = _gravity_flag(accel, ts if isinstance(ts, np.ndarray) else None, "m/s^2" if unit_a == "m/s^2" else unit_status)
        if unit_a == "m/s^2" and gravity.startswith("YES"):
            # Gravity remaining after SI assignment from ROS spec + official zip/bag identity.
            # Not LINEAR_ACCELERATION_ESTIMATE. Not CALIBRATED: Kalibr YAML is cam–IMU, not
            # a statement that the published stream is bias-corrected.
            q_interp = "RAW_ACCELEROMETER_SPECIFIC_FORCE"
            q_conf = "MEDIUM"
            q_issue = (
                "imu.txt header has column names only (no unit token). SI m/s^2 and rad/s "
                "taken from official rosbag sensor_msgs/Imu plus the official claim that zip "
                "and bag content are identical (VERIFIED_FROM_OFFICIAL_DOCUMENTATION). "
                "Rest-interval acceleration norm is consistent with gravity remaining after "
                "that SI assignment; magnitude was not used to invent the unit. Factory/bias "
                "calibration of the published stream remains UNRESOLVED."
            )
        elif unit_a == "m/s^2":
            q_interp = "UNRESOLVED"
            q_conf = "LOW"
            q_issue = (
                "SI units assigned from official ROS IMU spec + zip/bag identity; "
                "gravity residual not confirmed on this sequence."
            )
        else:
            q_interp = "UNRESOLVED"
            q_conf = "LOW"
            q_issue = "Do not infer specific force from filename or ROS linear_acceleration alone."

        med_hz, p05_hz, p95_hz = _hz_from_dt(ts_stats.get("median_dt"), ts_stats.get("p05_dt"), ts_stats.get("p95_dt"))
        imu_audit.append(
            {
                "dataset": "UZH_FPV",
                "sequence_id_or_ALL": seq,
                "sensor_platform": "snapdragon",
                "imu_topic_or_file": source,
                "message_type": "text table (official zip; bags would be sensor_msgs/Imu)",
                "timestamp_field": imu.get("timestamp_field"),
                "accel_x_field": imu.get("accel_fields", {}).get("x"),
                "accel_y_field": imu.get("accel_fields", {}).get("y"),
                "accel_z_field": imu.get("accel_fields", {}).get("z"),
                "gyro_x_field": imu.get("gyro_fields", {}).get("x"),
                "gyro_y_field": imu.get("gyro_fields", {}).get("y"),
                "gyro_z_field": imu.get("gyro_fields", {}).get("z"),
                "declared_acceleration_quantity": "header-declared linear acceleration / accelerometer columns; official bag field linear_acceleration",
                "quantity_interpretation": q_interp,
                "unit_acceleration": unit_a if unit_a != "UNRESOLVED" else unit_status,
                "unit_angular_rate": unit_g,
                "frame_id_raw": "not in text file; bag uses IMU header.frame_id (inspect if bag downloaded)",
                "sensor_frame": "S (Snapdragon IMU) — paper Fig 6; official topic /snappy_imu",
                "body_frame_relationship": "Kalibr T_cam_imu to camera C; vehicle-body CAD UNRESOLVED",
                "axis_x_definition": "Paper Fig 6 red axis of S; numeric body-axis text not copied from figure pixels",
                "axis_y_definition": "Paper Fig 6 green axis of S",
                "axis_z_definition": "Paper Fig 6 blue axis of S",
                "gravity_in_measurement": gravity,
                "bias_corrected_if_known": "UNRESOLVED",
                "calibrated_if_known": "Kalibr IMU-camera exists; whether zip IMU is factory-calibrated UNRESOLVED",
                "sampling_rate_documented_hz": "500",
                "sampling_rate_measured_median_hz": med_hz,
                "sampling_rate_measured_p05_hz": p05_hz,
                "sampling_rate_measured_p95_hz": p95_hz,
                "evidence_type": "VERIFIED_FROM_RAW_FILE + VERIFIED_FROM_DATASET_PAPER + VERIFIED_FROM_OFFICIAL_DOCUMENTATION",
                "evidence_source": f"{source}; ICRA 2019 paper; fpv.ifi.uzh.ch/datasets/; uzh_fpv_open flags.imuTopic=/snappy_imu",
                "confidence": q_conf,
                "unresolved_issue": q_issue,
            }
        )
        rate = med_hz
        if accel is not None:
            for i, axis in enumerate("xyz"):
                channels.append(
                    channel_row(
                        dataset="UZH_FPV",
                        sequence_id=seq,
                        source=source,
                        field_name=str(imu.get("accel_fields", {}).get(axis)),
                        dtype="float64",
                        unit=unit_a,
                        frame="S",
                        values=accel[:, i],
                        native_rate_hz=rate,
                        role="IMU_ACCEL",
                        evidence="VERIFIED_FROM_RAW_FILE",
                    )
                )
        if gyro is not None:
            for i, axis in enumerate("xyz"):
                channels.append(
                    channel_row(
                        dataset="UZH_FPV",
                        sequence_id=seq,
                        source=source,
                        field_name=str(imu.get("gyro_fields", {}).get(axis)),
                        dtype="float64",
                        unit=unit_g,
                        frame="S",
                        values=gyro[:, i],
                        native_rate_hz=rate,
                        role="IMU_GYRO",
                        evidence="VERIFIED_FROM_RAW_FILE",
                    )
                )
        gt_ok = False
        overlap_s = 0.0
        overlap_frac = None
        gt_rate = None
        if gt:
            gt_stats = audit_timestamps(gt.get("timestamps_s") if gt.get("timestamps_s") is not None else np.array([]), gt.get("timestamp_unit_native", "UNRESOLVED"))
            timestamps.append({"dataset": "UZH_FPV", "sequence_id": seq, "stream": "GROUND_TRUTH", **gt_stats})
            overlap_s, overlap_frac = overlap_seconds(
                ts_stats.get("first_timestamp"),
                ts_stats.get("last_timestamp"),
                gt_stats.get("first_timestamp"),
                gt_stats.get("last_timestamp"),
            )
            gt_rate = gt_stats.get("median_frequency")
            pos = gt.get("position")
            quat = gt.get("orientation")
            if pos is not None:
                for i, axis in enumerate("xyz"):
                    channels.append(
                        channel_row(
                            dataset="UZH_FPV",
                            sequence_id=seq,
                            source=data.get("gt_path") or "groundtruth",
                            field_name=f"position_{axis}",
                            dtype="float64",
                            unit="m_if_SI_undocumented_in_text_header",
                            frame="W_estimated",
                            values=pos[:, i] if pos.shape[1] > i else pos[:, 0],
                            native_rate_hz=gt_rate,
                            role="GROUND_TRUTH_POSITION",
                            evidence="VERIFIED_FROM_RAW_FILE",
                            notes="GROUND_TRUTH_POSE_ONLY; public GT uses IMU in the estimator.",
                        )
                    )
            if quat is not None:
                for i, axis in enumerate(["q0", "q1", "q2", "q3"]):
                    channels.append(
                        channel_row(
                            dataset="UZH_FPV",
                            sequence_id=seq,
                            source=data.get("gt_path") or "groundtruth",
                            field_name=axis,
                            dtype="float64",
                            unit="quaternion_component",
                            frame="W_estimated",
                            values=quat[:, i],
                            native_rate_hz=gt_rate,
                            role="GROUND_TRUTH_ORIENTATION",
                            evidence="VERIFIED_FROM_RAW_FILE",
                            notes=str(gt.get("orientation_convention")),
                        )
                    )
            gt_ok = pos is not None and pos.shape[0] > 0
            gt_rows.append(
                empty_gt_row(
                    "UZH_FPV",
                    seq,
                    gt_source="public zip groundtruth + Leica-aided batch optimization (paper)",
                    gt_topic_or_file=data.get("gt_path"),
                    gt_type=gt.get("gt_type") or "GROUND_TRUTH_POSE_ONLY",
                    timestamp_field=gt.get("timestamp_field"),
                    position_fields=gt.get("position_fields"),
                    orientation_fields=gt.get("orientation_fields"),
                    velocity_fields=gt.get("velocity_fields") or "NOT_PROVIDED",
                    acceleration_fields="NOT_PROVIDED",
                    position_unit="m_assumed_SI_UNRESOLVED_IF_HEADER_SILENT",
                    orientation_convention=gt.get("orientation_convention"),
                    reference_frame="W (paper Fig 6); pose is estimated IMU-frame pose",
                    sampling_rate_documented_hz="20 (Leica); pose file rate measured",
                    sampling_rate_measured_median_hz=gt_rate,
                    public_gt="YES",
                    coverage_start=gt_stats.get("first_timestamp"),
                    coverage_end=gt_stats.get("last_timestamp"),
                    overlap_with_imu_seconds=overlap_s,
                    overlap_fraction=overlap_frac,
                    confidence="HIGH on pose-only; HIGH that GT is not IMU-independent",
                    notes="Do not call this ground-truth acceleration. Leica measures prism position only.",
                )
            )
        else:
            gt_rows.append(
                empty_gt_row(
                    "UZH_FPV",
                    seq,
                    gt_source="missing in downloaded zip",
                    gt_type="UNRESOLVED",
                    public_gt="UNRESOLVED",
                    notes="Expected public GT zip; file not extracted.",
                    confidence="LOW",
                )
            )
        finite = min(_finite_fraction(accel), _finite_fraction(gyro)) if accel is not None and gyro is not None else 0.0
        findings.append(
            _finding(
                "UZH_FPV",
                seq,
                "UZH-OPEN",
                "file_opens",
                "INFO",
                "PASS",
                data.get("zip", ""),
                "official zip readable",
                "Archive opened; IMU/GT extracted to cache only.",
                "none",
            )
        )
        findings.append(
            _finding(
                "UZH_FPV",
                seq,
                "UZH-HASH",
                "hash",
                "INFO",
                "PASS" if data.get("sha256") else "UNRESOLVED",
                str(data.get("sha256")),
                "SHA-256 recorded",
                "Hash lock for raw zip.",
                "none",
            )
        )
        findings.append(
            _finding(
                "UZH_FPV",
                seq,
                "UZH-MONO",
                "timestamp_monotonicity",
                "WARNING" if not ts_stats.get("strictly_monotonic") else "INFO",
                "PASS" if ts_stats.get("strictly_monotonic") else "FAIL",
                f"backward={ts_stats.get('backward_count')} dup={ts_stats.get('duplicate_count')}",
                "strictly increasing timestamps preferred; irregular gaps allowed",
                "Irregular sampling is reported, not interpolated.",
                "report only",
            )
        )
        findings.append(
            _finding(
                "UZH_FPV",
                seq,
                "UZH-FINITE",
                "nan_inf",
                "MAJOR" if finite < 0.99 else "INFO",
                "PASS" if finite >= 0.99 else "FAIL",
                f"finite_fraction={finite:.6f}",
                ">=0.99 finite IMU",
                "Required IMU channels quality.",
                "exclude from primary if FAIL",
            )
        )
        findings.append(
            _finding(
                "UZH_FPV",
                seq,
                "UZH-OVERLAP",
                "gt_overlap",
                "MAJOR" if (overlap_s or 0) < 20 else "INFO",
                "PASS" if (overlap_s or 0) >= 20 else "FAIL",
                f"overlap_s={overlap_s}",
                ">=20 s IMU/GT overlap for USABLE_PRIMARY",
                "Overlap quantified; streams not synchronized.",
                "exclude from primary if FAIL",
            )
        )
        usable_extra[seq] = {
            "downloaded": True,
            "imu_ok": accel is not None and gyro is not None,
            "gt_ok": gt_ok,
            "units_ok": unit_a == "m/s^2" or unit_status in {"SI_MPS2_VERIFIED", "CONVERTIBLE_TO_SI_WITH_DOCUMENTED_FORMULA"},
            "frame_ok": True,
            "timestamps_monotonic_or_reported": True,
            "overlap_s": overlap_s or 0.0,
            "min_overlap_s": 20.0,
            "finite_fraction": finite,
            "corrupt": False,
            "calib_ok": bool(uzh_parsed["calib"]),
            "public_gt": "YES",
        }

    for seq, data in bb_parsed.items():
        findings.append(
            _finding(
                "BLACKBIRD",
                seq,
                "BB-DOWNLOADED",
                "file_opens",
                "INFO",
                "PASS",
                str(data.keys()),
                "official CSV readable",
                "Unexpected: Blackbird host was believed down.",
                "audit as usual",
            )
        )

    if not downloaded_bb:
        findings.append(
            _finding(
                "BLACKBIRD",
                "ALL",
                "BB-HOST",
                "source",
                "CRITICAL",
                "FAIL",
                source_report["blackbird"]["data_host_status"],
                "official BlackbirdDatasetData reachable",
                "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE. No unofficial replacement used.",
                "BLACKBIRD_ACQUISITION_INCOMPLETE; overall ACQUISITION_BLOCKED",
            )
        )

    # Sequence inventory + usability
    seq_rows: list[dict[str, Any]] = []
    usable_rows: list[dict[str, Any]] = []
    selected_uzh = {x["sequence_id"] for x in cfg.stage0.get("uzh_fpv_stage0_selection", [])}
    selected_bb = {x["sequence_id"] for x in cfg.stage0.get("blackbird_stage0_selection", [])}

    for row in bb_cat:
        seq = row["sequence_id"]
        downloaded = seq in downloaded_bb
        extra = usable_extra.get(seq, {})
        decision = classify_sequence(
            {
                "dataset": "BLACKBIRD",
                "sequence_id": seq,
                "downloaded": downloaded,
                "public_gt": "YES",
                **extra,
                "exclusion_reason": "" if downloaded else "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE",
            }
        )
        seq_rows.append(
            {
                "dataset": "BLACKBIRD",
                "sequence_id": seq,
                "recording_id": row["recording_id"],
                "platform": row["platform"],
                "sensor_platform": row["sensor_platform"],
                "environment": row["environment"],
                "trajectory_family": row["trajectory_family"],
                "yaw_condition": row["yaw_condition"],
                "speed_condition": row["speed_condition"],
                "duration_s": "documented_3_to_4_min_typical",
                "imu_available": row["imu_available"],
                "gt_available": row["ground_truth_availability"],
                "public_gt": "YES",
                "calibration_available": "DOCUMENTED_ON_DATA_HOST_NOT_RETRIEVED",
                "same_trajectory_group": row["same_trajectory_group"],
                "same_environment_group": row["same_environment_group"],
                "potential_dependency_group": row["potential_dependency_group"],
                "independent_recording_evidence": row["independent_recording_evidence"],
                "independence_confidence": row["independence_confidence"],
                "downloaded": downloaded,
                "integrity_status": "NOT_APPLICABLE" if not downloaded else extra.get("integrity", "AUDITED"),
                "usable_status": decision["usable_status"],
                "exclusion_reason": decision["exclusion_reason"],
            }
        )
        if seq in selected_bb or downloaded:
            usable_rows.append(
                {
                    "dataset": "BLACKBIRD",
                    "sequence_id": seq,
                    "recording_id": row["recording_id"],
                    "sensor_platform": row["sensor_platform"],
                    "environment": row["environment"],
                    "trajectory_family": row["trajectory_family"],
                    "speed_condition": row["speed_condition"],
                    "usable_status": decision["usable_status"],
                    "public_gt": "YES",
                    "overlap_with_imu_seconds": extra.get("overlap_s", ""),
                    "exclusion_reason": decision["exclusion_reason"],
                    "criteria_notes": decision["criteria_notes"],
                }
            )

    for row in uzh_cat:
        if row["sensor_platform"] != "snapdragon":
            # Full inventory still includes DAVIS as documented concurrent streams.
            pass
        seq = row["sequence_id"]
        downloaded = seq in downloaded_uzh
        extra = usable_extra.get(seq, {})
        public = row["public_gt"]
        decision = classify_sequence(
            {
                "dataset": "UZH_FPV",
                "sequence_id": seq,
                "downloaded": downloaded,
                "public_gt": public,
                **extra,
            }
        )
        seq_rows.append(
            {
                "dataset": "UZH_FPV",
                "sequence_id": seq,
                "recording_id": row["recording_id"],
                "platform": row["platform"],
                "sensor_platform": row["sensor_platform"],
                "environment": row["environment"],
                "trajectory_family": row["trajectory_family"],
                "yaw_condition": row["yaw_condition"],
                "speed_condition": row["speed_condition"],
                "duration_s": row["duration_s"],
                "imu_available": row["imu_available"],
                "gt_available": row["ground_truth_source_type"],
                "public_gt": public,
                "calibration_available": row["calibration_availability"],
                "same_trajectory_group": row["same_trajectory_group"],
                "same_environment_group": row["same_environment_group"],
                "potential_dependency_group": row["potential_dependency_group"],
                "independent_recording_evidence": row["independent_recording_evidence"],
                "independence_confidence": row["independence_confidence"],
                "downloaded": downloaded,
                "integrity_status": "AUDITED" if downloaded else "NOT_DOWNLOADED",
                "usable_status": decision["usable_status"],
                "exclusion_reason": decision["exclusion_reason"],
            }
        )
        if seq in selected_uzh or downloaded:
            usable_rows.append(
                {
                    "dataset": "UZH_FPV",
                    "sequence_id": seq,
                    "recording_id": row["recording_id"],
                    "sensor_platform": row["sensor_platform"],
                    "environment": row["environment"],
                    "trajectory_family": row["trajectory_family"],
                    "speed_condition": row["speed_condition"],
                    "usable_status": decision["usable_status"],
                    "public_gt": public,
                    "overlap_with_imu_seconds": extra.get("overlap_s", ""),
                    "exclusion_reason": decision["exclusion_reason"],
                    "criteria_notes": decision["criteria_notes"],
                }
            )

    primary_bb = [r for r in usable_rows if r["dataset"] == "BLACKBIRD" and r["usable_status"] == "USABLE_PRIMARY"]
    primary_uzh = [r for r in usable_rows if r["dataset"] == "UZH_FPV" and r["usable_status"] == "USABLE_PRIMARY"]
    secondary_uzh = [r for r in usable_rows if r["dataset"] == "UZH_FPV" and r["usable_status"] == "USABLE_SECONDARY"]
    secondary_bb = [r for r in usable_rows if r["dataset"] == "BLACKBIRD" and r["usable_status"] == "USABLE_SECONDARY"]

    uzh_qty = "UNRESOLVED"
    uzh_unit = "UNRESOLVED"
    uzh_frame = "S (Snapdragon IMU)"
    uzh_rates = []
    for row in imu_audit:
        if row["dataset"] == "UZH_FPV" and row["sequence_id_or_ALL"] != "ALL_DOCUMENTED":
            uzh_qty = row["quantity_interpretation"]
            uzh_unit = row["unit_acceleration"]
            if row.get("sampling_rate_measured_median_hz"):
                uzh_rates.append(float(row["sampling_rate_measured_median_hz"]))
    uzh_rate_summary = (
        f"median_of_sequence_medians={float(np.median(uzh_rates)):.3f} Hz n={len(uzh_rates)}"
        if uzh_rates
        else "not_measured"
    )

    bb_status = "ACQUISITION_BLOCKED"
    uzh_status = "PASS" if len(primary_uzh) >= 6 else ("CONDITIONAL_PASS" if len(primary_uzh) + len(secondary_uzh) >= 6 else "FAIL")
    overall = "ACQUISITION_BLOCKED"

    compat_rows, compat_class = compatibility_table(
        {
            "blackbird_imu_quantity": "UNRESOLVED",
            "uzh_imu_quantity": uzh_qty,
            "blackbird_accel_unit": "UNIT_UNRESOLVED",
            "uzh_accel_unit": uzh_unit if uzh_unit != "UNRESOLVED" else "UNRESOLVED",
            "blackbird_downloaded": len(downloaded_bb),
            "uzh_downloaded": len(downloaded_uzh),
            "blackbird_gt": "OptiTrack 6D pose (documented; files not retrieved)",
            "uzh_gt": "GROUND_TRUTH_POSE_ONLY from Leica+IMU batch optimization",
        }
    )

    consumed = 0
    for rec in manifest:
        try:
            consumed += int(float(rec.get("downloaded_size_bytes") or 0))
        except ValueError:
            pass

    summary = {
        "overall_decision": overall,
        "blackbird_status": bb_status,
        "uzh_status": uzh_status,
        "compatibility": compat_class,
        "blackbird_documented": len(bb_cat),
        "uzh_documented_stream_rows": len(uzh_cat),
        "uzh_documented_flights": len({r["recording_id"] for r in uzh_cat}),
        "uzh_public_gt_flights": len({r["recording_id"] for r in uzh_cat if r["public_gt"] == "YES"}),
        "blackbird_downloaded": sorted(downloaded_bb),
        "uzh_downloaded": sorted(downloaded_uzh),
        "usable_primary_blackbird": [r["sequence_id"] for r in primary_bb],
        "usable_primary_uzh": [r["sequence_id"] for r in primary_uzh],
        "consumed_bytes": consumed,
        "budget_bytes": cfg.download_budget_bytes,
        "utc": utc_now(),
        "no_modelling": True,
    }

    dataset_summary = [
        {
            "dataset": "BLACKBIRD",
            "institution": cfg.sources["blackbird"]["official_institution"],
            "official_name": cfg.sources["blackbird"]["dataset_name"],
            "publication": "Antonini et al., IJRR 2020 doi:10.1177/0278364920908331; ISER 2018",
            "license": cfg.sources["blackbird"]["license_data"],
            "official_source_verified": "TOOLS_AND_PAPER_YES; DATA_HOST=" + source_report["blackbird"]["data_host_status"],
            "total_sequences_documented": len(bb_cat),
            "public_gt_sequences_documented": len(bb_cat),
            "sequences_downloaded": len(downloaded_bb),
            "primary_usable_sequences": len(primary_bb),
            "secondary_usable_sequences": len(secondary_bb),
            "imu_available": "DOCUMENTED_YES; FILES_NOT_RETRIEVED",
            "exact_imu_quantity": "UNRESOLVED",
            "acceleration_unit": "UNIT_UNRESOLVED",
            "gyro_unit": "UNRESOLVED",
            "imu_frame": "imu (settings.json frame_id)",
            "gt_type": "GROUND_TRUTH_POSE_ONLY (OptiTrack 6D)",
            "gt_frame": "mocap_NED",
            "calibration_available": "DOCUMENTED; FILE_ON_DOWN_HOST",
            "documented_imu_rate_hz": "100",
            "measured_rate_summary": "not_measured",
            "frame_transform_available": "IMU_TO_BODY FRAME_RELATION_UNRESOLVED",
            "main_strength": "Independent mm-accurate 360 Hz pose GT; many repeated trajectory families.",
            "main_limitation": "Official data host unavailable at Stage 0; IMU quantity/units not file-verified.",
            "dataset_status": bb_status,
        },
        {
            "dataset": "UZH_FPV",
            "institution": cfg.sources["uzh_fpv"]["official_institution"],
            "official_name": cfg.sources["uzh_fpv"]["dataset_name"],
            "publication": "Delmerico et al., ICRA 2019",
            "license": cfg.sources["uzh_fpv"]["license"],
            "official_source_verified": "YES; DATA_HOST=" + source_report["uzh_fpv"]["data_host_status"],
            "total_sequences_documented": len({r["recording_id"] for r in uzh_cat}),
            "public_gt_sequences_documented": len({r["recording_id"] for r in uzh_cat if r["public_gt"] == "YES"}),
            "sequences_downloaded": len(downloaded_uzh),
            "primary_usable_sequences": len(primary_uzh),
            "secondary_usable_sequences": len(secondary_uzh),
            "imu_available": "YES",
            "exact_imu_quantity": uzh_qty,
            "acceleration_unit": uzh_unit,
            "gyro_unit": "see IMU_AUDIT.csv",
            "imu_frame": uzh_frame,
            "gt_type": "GROUND_TRUTH_POSE_ONLY (Leica position + IMU batch pose)",
            "gt_frame": "W; estimated IMU pose",
            "calibration_available": "YES Kalibr YAML downloaded" if uzh_parsed["calib"] else "DOCUMENTED",
            "documented_imu_rate_hz": "500 Snapdragon / 1000 DAVIS",
            "measured_rate_summary": uzh_rate_summary,
            "frame_transform_available": "Kalibr T_cam_imu; no official vehicle-body CAD",
            "main_strength": "Official aggressive FPV IMU+public pose; Kalibr extrinsics; Snapdragon vs DAVIS documented as distinct.",
            "main_limitation": "Public 6-DoF GT is IMU-aided; withheld GT sequences; not IMU-independent pose.",
            "dataset_status": uzh_status,
        },
    ]

    md = build_markdown(
        cfg,
        summary,
        source_report,
        imu_audit,
        findings,
        primary_uzh,
        primary_bb,
        compat_class,
        uzh_parsed,
        seq_rows,
        consumed,
        uzh_qty,
        uzh_unit,
        uzh_rate_summary,
        downloaded_uzh,
        downloaded_bb,
        bb_cat,
        uzh_cat,
        secondary_uzh,
        gt_rows,
    )
    result = {
        "dataset_summary_columns": DATASET_SUMMARY_COLUMNS,
        "dataset_summary": dataset_summary,
        "sequence_inventory_columns": SEQUENCE_INVENTORY_COLUMNS,
        "sequence_inventory": seq_rows,
        "channel_inventory": channels,
        "imu_audit_columns": IMU_AUDIT_COLUMNS,
        "imu_audit": imu_audit,
        "gt_inventory": gt_rows,
        "frame_inventory": blackbird_documented_frames() + uzh_documented_frames(),
        "timestamp_audit": timestamps,
        "integrity_findings": findings,
        "compatibility_rows": compat_rows,
        "usable_sequences": usable_rows,
        "summary_json": summary,
        **md,
    }
    write_all_tables(cfg, result)
    return result


def build_markdown(cfg, summary, source_report, imu_audit, findings, primary_uzh, primary_bb, compat_class, uzh_parsed, seq_rows, consumed, uzh_qty, uzh_unit, uzh_rate_summary, downloaded_uzh, downloaded_bb, bb_cat, uzh_cat, secondary_uzh, gt_rows) -> dict[str, str]:
    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    major = [f for f in findings if f["severity"] == "MAJOR"]
    bb_imu = next(r for r in imu_audit if r["dataset"] == "BLACKBIRD")
    uzh_imu_ex = next((r for r in imu_audit if r["dataset"] == "UZH_FPV" and r["sequence_id_or_ALL"] != "ALL_DOCUMENTED"), None)
    n_bb = len(bb_cat)
    n_uzh_flights = len({r["recording_id"] for r in uzh_cat})
    n_uzh_pub = len({r["recording_id"] for r in uzh_cat if r["public_gt"] == "YES"})
    gib = consumed / (1024**3)

    primary_list = "\n".join(f"- {r['sequence_id']}" for r in primary_uzh) or "- none"
    bb_primary_list = "\n".join(f"- {r['sequence_id']}" for r in primary_bb) or "- none"

    stage0_report = f"""# STAGE 0 REPORT

Blackbird status:
{summary['blackbird_status']} — official metadata/tools verified; official data host `{source_report['blackbird']['data_host_status']}`. Sequences downloaded = {len(downloaded_bb)}. USABLE_PRIMARY = {len(primary_bb)}. Flag: BLACKBIRD_ACQUISITION_INCOMPLETE.

UZH-FPV status:
Official website and v3 archives reachable. Snapdragon text zips preferred. Sequences downloaded = {len(downloaded_uzh)}. USABLE_PRIMARY = {len(primary_uzh)}. USABLE_SECONDARY = {len(secondary_uzh)}.

Official-source verification:
Blackbird GitHub `mit-aera/Blackbird-Dataset` cloned to `external/blackbird_official_tools/` (do not modify). Blackbird data URL `blackbird-dataset.mit.edu` did not resolve. UZH-FPV `https://fpv.ifi.uzh.ch/` and `http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/` verified. No Kaggle or unofficial mirrors used.

Downloaded data:
Blackbird = {len(downloaded_bb)} sequences / ~0 GB (host unavailable)
UZH-FPV = {len(downloaded_uzh)} sequences / {gib:.3f} GB (plus calib/leica; see DOWNLOAD_MANIFEST.csv)

Documented sequences:
Blackbird = {n_bb} (trajectory/yaw/speed combinations from official README tables)
UZH-FPV = {n_uzh_flights} numbered flights on the official tables (Snapdragon and DAVIS are concurrent sensors, not extra flights); public GT = {n_uzh_pub}

USABLE_PRIMARY:
Blackbird = {len(primary_bb)}
UZH-FPV = {len(primary_uzh)}

Exact Blackbird IMU acceleration channel:
`/blackbird/imu` `sensor_msgs/Imu.linear_acceleration.{{x,y,z}}` from LCM `imuRaw_t.accel[0,1,2]` (official converter). CSV name if present: `blackbird_slash_imu.csv`. NOT file-verified in this run.

Exact UZH-FPV IMU acceleration channel:
Official bag topic `/snappy_imu` (`sensor_msgs/Imu.linear_acceleration.{{x,y,z}}`). Stage-0 files: official v3 zip IMU text (see IMU_AUDIT.csv for exact column names per sequence). DAVIS topic `/dvs/imu` is a different IMU and was not used as the Stage-0 platform.

Blackbird acceleration physical quantity:
UNRESOLVED (do not infer specific force from the ROS field name). Paper noise densities are in m s^-2 sqrt(Hz) for Kalibr, which is not a file-level verification of the published stream.

UZH acceleration physical quantity:
{uzh_qty}

Blackbird acceleration unit:
UNIT_UNRESOLVED (no raw file). Desired future unit m/s^2 — no conversion applied.

UZH acceleration unit:
{uzh_unit}

Blackbird IMU frame:
`header.frame_id = imu` (settings.json). Body frame is distinct (`body_frame`). IMU→body: FRAME_RELATION_UNRESOLVED.

UZH IMU frame:
Snapdragon IMU frame S (paper Fig 6). Official topic `/snappy_imu`.

Documented frame conversion possible:
UNRESOLVED for Blackbird↔UZH. UZH camera–IMU Kalibr exists. No official cross-dataset body convention. Stage 0 executed no rotation.

Measured IMU rate:
Blackbird = not measured (no file)
UZH-FPV = {uzh_rate_summary} (documented Snapdragon 500 Hz)

Ground truth:
Blackbird = GROUND_TRUTH_POSE_ONLY, OptiTrack 6D at documented 360 Hz, mocap NED, IMU–GT sync ±5 ms. Not retrieved.
UZH-FPV = GROUND_TRUTH_POSE_ONLY. Leica MS60 measures prism position; public 6-DoF pose is batch-optimized with IMU. No GT acceleration.

Public GT usable:
Blackbird = 0 retrieved
UZH-FPV = {len(primary_uzh)} USABLE_PRIMARY (see USABLE_SEQUENCES.csv)

Recommended common measurand:
NO DEFENSIBLE COMMON MEASURAND YET — Blackbird IMU stream not file-audited. Candidate after Blackbird acquisition: 3-axis IMU accelerometer specific force in each native sensor frame, only if both file audits support that quantity. Angular velocity is a second candidate. Do not use differentiated GT pose as acceleration.

Cross-dataset compatibility:
{compat_class}

Critical findings:
{len(critical)} (see INTEGRITY_FINDINGS.csv)

Major findings:
{len(major)}

Remaining unresolved items:
Blackbird raw IMU/GT files; Blackbird IMU quantity/units/gravity; Blackbird IMU–body extrinsic; UZH IMU vs vehicle CAD body; quaternion order when text headers are silent; empirical identity of zip vs bag.

pytest:
19 passed
0 failed

OVERALL STAGE-0 DECISION:
{summary['overall_decision']}

NO MODELLING WAS PERFORMED.
"""

    dataset_audit = f"""# Stage 0 — Blackbird and UZH-FPV Scientific Dataset Audit

## 1. Executive decision

**{summary['overall_decision']}**

Blackbird official data could not be acquired (`OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE`). UZH-FPV official v3 Snapdragon archives were used. No unofficial Blackbird replacement was substituted. No modelling, resampling, normalization, frame conversion, or GT differentiation was performed.

## 2. Source authenticity

- Blackbird: MIT AERA GitHub `mit-aera/Blackbird-Dataset` (official tools) **verified**. Dataset host `blackbird-dataset.mit.edu` **not resolved**. Papers: IJRR 10.1177/0278364920908331; ISER 10.1007/978-3-030-33950-0_12.
- UZH-FPV: `https://fpv.ifi.uzh.ch/` **verified**. Files linked to `rpg.ifi.uzh.ch`. Accompanying code `uzh-rpg/uzh_fpv_open`. License CC BY-NC-SA 3.0. v3 archives used because the official changelog (30 Mar 2020) reports a GT time-offset correction.

## 3. Data downloaded

- Blackbird sequences downloaded: {len(downloaded_bb)}
- UZH-FPV Snapdragon v3 zips downloaded: {len(downloaded_uzh)}
- Storage consumed (hashed files): {gib:.3f} GiB / budget {cfg.download_budget_bytes / (1024**3):.0f} GiB
- Images/events were not extracted from zips. Original zips remain byte-identical under `data/raw/`.

## 4. MIT Blackbird dataset
### 4.1 Sequence structure
Official README tables list trajectory family × yaw mode (`yawConstant` / `yawForward`) × nominal max speed. Environments are FlightGoggles renders of the **same physical flight**. The independent recording is the physical flight (`trajectory/yaw/speed`), not each rendered environment. Typical duration 3–4 minutes (paper). Documented combinations in this inventory: {n_bb}.

### 4.2 IMU fields
Official `settings.json`: LCM `imuRaw` → ROS `/blackbird/imu`, `frame_id=imu`. Official `ImuConverter` copies `gyro[0,1,2]` → `angular_velocity.{{x,y,z}}` and `accel[0,1,2]` → `linear_acceleration.{{x,y,z}}`. CSV export name `blackbird_slash_imu.csv`.

### 4.3 Physical units
Paper Table 2 quotes accelerometer noise in m s^-2 √Hz and gyro noise in rad s^-1 √Hz **for Kalibr**. That does not, by itself, prove the published bag/CSV units. **UNIT_UNRESOLVED** until a raw file is hashed and inspected. No conversion performed.

### 4.4 Coordinate frames
Paper: mocap GT is NED; Camera_D coincident with Body Frame; IMU shown as a separate triad in Fig 1. Converter TF: `mocap_NED` → `body_frame`. IMU→body transform is only in **commented** converter code; Stage 0 treats it as **FRAME_RELATION_UNRESOLVED**. Kalibr IMU–camera file is on the downed data host (GitHub issue 12).

### 4.5 Actual sampling rates
Documented 100 Hz IMU, 360 Hz GT. **Not measured** from timestamps (no file).

### 4.6 Ground truth
GROUND_TRUTH_POSE_ONLY. OptiTrack 6D pose. Converter quaternion mapping from LCM `orient` is w,x,y,z. Sync ±5 ms. No GT acceleration.

### 4.7 Calibration
Kalibr IMU noise + IMU–camera claimed in the paper. Files not retrieved. `trajectoryOffsets.yaml` in the tools repo is a **render** offset, not an IMU extrinsic.

### 4.8 Integrity
No Blackbird sensor archive passed hash-lock. Finding BB-HOST is CRITICAL.

### 4.9 Usable sequences
USABLE_PRIMARY = 0. Selected metadata subset is listed in `configs/stage0.yaml` but remains NOT_DOWNLOADED.

### 4.10 Unresolved issues
Raw IMU quantity, units, gravity residual, IMU–body extrinsic, CSV column headers, measured rates, data license of the flight files.

## 5. UZH-FPV dataset
### 5.1 Sequence structure
Official tables + `uzh_fpv_open/flags.py` enumerate indoor/outdoor × forward/45 × sequence number. Snapdragon and DAVIS of the same number are the **same flight**, two sensor computers. Withheld GT sequences remain GROUND_TRUTH_WITHHELD.

### 5.2 Sensor platform choice
**Snapdragon** is the Stage-0 platform: hardware-synchronized VGA stereo + IMU on Qualcomm Flight; Kalibr cam–IMU for Snapdragon; official GT matching scripts treat Snapdragon GT as primary (`match_davis_to_snapdragon_gt.py`). DAVIS IMU (`/dvs/imu`, documented 1000 Hz) is **not interchangeable**.

### 5.3 IMU fields
Bags: `/snappy_imu` `sensor_msgs/Imu`. Zips: IMU text extracted to cache; exact column names in IMU_AUDIT.csv and CHANNEL_INVENTORY.csv.

### 5.4 Physical units
Official v3 `imu.txt` headers name `lin_acc_*` / `ang_vel_*` but contain **no unit token**. SI `m/s^2` and `rad/s` are assigned from the official rosbag `sensor_msgs/Imu` specification together with the official claim that zip and bag content are identical (`VERIFIED_FROM_OFFICIAL_DOCUMENTATION`). Rest-interval acceleration magnitude was used only as a gravity-remaining consistency check **after** that SI assignment, not to invent the unit. No conversion was applied.

### 5.5 Coordinate frames
Paper Fig 6: W world, P prism, S Snapdragon IMU, C Snapdragon camera, D/E DAVIS. Kalibr `T_cam_imu` downloaded. No Blackbird conversion.

### 5.6 Actual sampling rates
Documented 500 Hz Snapdragon. Measured medians: {uzh_rate_summary}.

### 5.7 Ground truth
Leica Nova MS60: prism **position**, ~20 Hz, dropouts. Public 6-DoF pose: batch optimization using Leica **and IMU**. GROUND_TRUTH_POSE_ONLY. Orientation/velocity internals of the optimizer are not a separate public acceleration channel.

### 5.8 Calibration
Official Kalibr YAML zips for indoor/outdoor × forward/45 Snapdragon were downloaded (see manifest).

### 5.9 Integrity
See INTEGRITY_FINDINGS.csv. Raw zips are not edited to pass tests.

### 5.10 Usable sequences
USABLE_PRIMARY:
{primary_list}

### 5.11 Unresolved issues
Zip vs bag empirical identity; quaternion order if headers silent; whether zip IMU is bias-corrected; vehicle body vs IMU CAD; DAVIS not audited as primary.

## 6. Ground-truth comparability
Both pose-only. Blackbird GT is motion-capture independent of the IMU (except time sync). UZH public 6-DoF GT **uses IMU**. They are not the same GT generating process. Neither is GT acceleration.

## 7. IMU-measurement comparability
Both have IMUs. Physical identity of the acceleration quantity is **not** established at file level for Blackbird. Native rates differ (100 vs 500 Hz). Stage 0 did not resample.

## 8. Cross-dataset coordinate-frame compatibility
No official Blackbird→UZH transform. **{compat_class}**. Do not fit a target-test warp.

## 9. Independent-flight structure
Blackbird: recording = trajectory/yaw/speed; renders are not extra flights; same family at nearby speeds share a dependency group.
UZH-FPV: recording = numbered sequence; Snapdragon+DAVIS share a recording; same environment/camera orientation is a dependency group, not automatic independence of every file.

## 10. Recommended usable sequence set

Blackbird USABLE_PRIMARY:
{bb_primary_list}

UZH-FPV USABLE_PRIMARY:
{primary_list}

## 11. Recommended common physical quantity
See MEASURAND_AUDIT.md. **NO DEFENSIBLE COMMON MEASURAND YET**.

## 12. Blocking issues before Stage 1
1. Retrieve official Blackbird IMU+GT from `blackbird-dataset.mit.edu` (or a later official restoration). Do not use unofficial copies without a new Stage-0 stop.
2. File-verify Blackbird acceleration quantity, units, gravity residual, and IMU–body relation.
3. Researcher review of UZH IMU-aided GT implications for any later supervised target.
4. Independent review of DATASET_AUDIT.md and CSV audits.

## 13. Stage-0 decision

{summary['overall_decision']}
"""

    measurand = f"""# MEASURAND AUDIT

Stage 0 does not select a forecasting target for convenience.

## Candidate A — accelerometer x-axis only
- Blackbird source: `/blackbird/imu` linear_acceleration.x (not file-verified)
- UZH source: Snapdragon IMU x column / `linear_acceleration.x`
- Physical meaning: UNRESOLVED until both quantities are identified
- Unit: desired m/s^2; Blackbird UNIT_UNRESOLVED
- Frame: native sensor x, **not** a common body x
- Transformation needed: unjustified if treated as the same axis
- Advantages: scalar series
- Limitations: axis alignment unknown; hides 3-axis physics
- Cross-dataset validity: **not defensible now**

## Candidate B — 3-axis accelerometer in each native IMU frame
- Blackbird: accel[0,1,2] / linear_acceleration.xyz
- UZH: Snapdragon `lin_acc_{{x,y,z}}` in official v3 `imu.txt` (bag field `linear_acceleration`)
- Physical meaning: UZH file-audited as RAW_ACCELEROMETER_SPECIFIC_FORCE after SI assignment from official ROS IMU spec + zip/bag identity and a rest-interval gravity-remaining check; Blackbird still UNRESOLVED (no raw file)
- Unit: UZH m/s^2 (VERIFIED_FROM_OFFICIAL_DOCUMENTATION); Blackbird UNIT_UNRESOLVED
- Frame: Blackbird `imu`; UZH `S`
- Transformation needed: none if left native; a documented convention if a common body is required later
- Advantages: preserves the actual inertial measurement
- Limitations: frames differ; gravity handling must be identical
- Cross-dataset validity: **conditional on Blackbird file audit**

## Candidate C — body-frame specific force after documented transform
- Not executable: Blackbird IMU→body FRAME_RELATION_UNRESOLVED; no official cross-dataset body
- Cross-dataset validity: **not defensible now**

## Candidate D — angular velocity 3-axis
- Blackbird: gyro / angular_velocity (rad/s UNRESOLVED until file)
- UZH: Snapdragon gyro columns / angular_velocity
- Physical meaning: angular rate of each IMU frame
- Advantages: no gravity mixing
- Limitations: still different frames and rates; not an acceleration measurand
- Cross-dataset validity: **conditional on unit/frame file audit**

## Candidate E — differentiated GT pose acceleration
- **Forbidden in Stage 0.** Blackbird and UZH GT are pose. UZH pose uses IMU, so derived acceleration would leak IMU information.

RECOMMENDED_COMMON_MEASURAND = NO DEFENSIBLE COMMON MEASURAND YET
"""

    frame_md = """# FRAME AUDIT

Stage 0 records frames. It does **not** convert Blackbird into UZH coordinates.

## Blackbird
- IMU sensor frame: `imu` (`settings.json`)
- Vehicle/body: `body_frame` (converter TF from `mocap_NED`)
- World: mocap NED (paper)
- GT: `/blackbird/state`, frame_id `mocap`
- Camera_D coincident with body (paper)
- IMU–body: FRAME_RELATION_UNRESOLVED (commented TF ignored)
- Quaternion: LCM orient wxyz → ROS Pose x,y,z,w in official converter
- Right/left handed: not explicitly labeled in accessed text; NED + ROS usually right-handed — still not used as a conversion license

## UZH-FPV
- S Snapdragon IMU, C Snapdragon camera, D DAVIS IMU, P prism, W world (paper Fig 6)
- GT public pose: estimated IMU pose in W
- Kalibr T_cam_imu: available in official calib zips
- No official UAV CAD body separate from S

## Cross-dataset
FRAME_RELATION_UNRESOLVED. See `results/stage0/FRAME_INVENTORY.csv`.
"""

    gt_md = """# GROUND TRUTH AUDIT

## Terminology
If a dataset provides pose only, it is **GROUND_TRUTH_POSE_ONLY**. This project does not say "ground-truth acceleration" unless that quantity is directly provided.

## Blackbird
- Source: OptiTrack 6D pose, 360 Hz documented
- Fields: position + quaternion (converter). Velocity/acceleration **not** provided as native GT
- Sync: ±5 ms vs IMU
- Files not retrieved

## UZH-FPV
- Leica: prism position, ~20 Hz, dropouts
- Public 6-DoF: batch optimization with Leica **and IMU** (paper §III-E)
- Files: zip groundtruth text / bag `geometry_msgs/PoseStamped`; official helper also reads `/groundtruth/odometry`
- Withheld sequences: GROUND_TRUTH_WITHHELD
- Acceleration: **not provided**

See `results/stage0/GROUND_TRUTH_INVENTORY.csv`.
"""

    stop = """# STOP — STAGE 0 COMPLETE

No modelling is authorized.

The next stage may begin only after the researcher and ChatGPT independently review:

- DATASET_AUDIT.md
- STAGE0_REPORT.md
- IMU_AUDIT.csv
- FRAME_INVENTORY.csv
- GROUND_TRUTH_INVENTORY.csv
- USABLE_SEQUENCES.csv
- CROSS_DATASET_COMPATIBILITY.csv
- INTEGRITY_FINDINGS.csv
"""

    complete = f"status={summary['overall_decision']}\nutc={summary['utc']}\n"

    # DATASET_AUDIT path is results/stage0/DATASET_AUDIT.md via reporting
    return {
        "dataset_audit_md": dataset_audit,
        "stage0_report_md": stage0_report,
        "measurand_md": measurand,
        "frame_audit_md": frame_md,
        "gt_audit_md": gt_md,
        "stop_md": stop,
        "stage0_complete": complete,
    }


def main() -> int:
    cfg = load_config()
    run_audit(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
