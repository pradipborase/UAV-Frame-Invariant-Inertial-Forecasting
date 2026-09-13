"""Load native SI IMU streams for Stage 1. Raw archives remain unmodified."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audit.euroc_schema import (
    as_uint64,
    map_euroc_imu_header,
    read_euroc_csv,
    relative_seconds_from_ns,
)
from audit.text_reader import interpret_imu_table, read_text_table

from .invariants import angular_speed_magnitude, specific_force_magnitude

EUROC_PRIMARY = (
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
)
UZH_PRIMARY = (
    "indoor_forward_6_snapdragon",
    "indoor_forward_9_snapdragon",
    "indoor_forward_10_snapdragon",
    "indoor_45_2_snapdragon",
    "indoor_45_4_snapdragon",
    "indoor_45_13_snapdragon",
    "indoor_45_14_snapdragon",
    "outdoor_forward_1_snapdragon",
)


def euroc_imu_path(root: Path, sequence_id: str) -> Path:
    return root / "data" / "cache" / "euroc" / sequence_id / "mav0" / "imu0" / "data.csv"


def uzh_imu_path(root: Path, sequence_id: str) -> Path:
    return root / "data" / "cache" / "uzh_fpv" / "sequences" / f"{sequence_id}_with_gt" / "imu.txt"


def uzh_zip_path(root: Path, sequence_id: str) -> Path:
    return root / "data" / "raw" / "uzh_fpv" / "sequences" / f"{sequence_id}_with_gt.zip"


def euroc_inner_zip_path(root: Path, sequence_id: str) -> Path:
    return root / "data" / "raw" / "euroc" / "extracted" / f"{sequence_id}.zip"


def load_euroc_native(root: Path, sequence_id: str) -> dict[str, Any]:
    path = euroc_imu_path(root, sequence_id)
    headers, array, header_line = read_euroc_csv(path)
    mapping = map_euroc_imu_header(headers)
    ts_idx = int(mapping["timestamp_index"] or 0)
    gx, gy, gz = mapping["gyro_indices"]
    ax, ay, az = mapping["accel_indices"]
    ts_s, _origin = relative_seconds_from_ns(array[:, ts_idx])
    fx = np.asarray(array[:, ax], dtype=np.float64)
    fy = np.asarray(array[:, ay], dtype=np.float64)
    fz = np.asarray(array[:, az], dtype=np.float64)
    wx = np.asarray(array[:, gx], dtype=np.float64)
    wy = np.asarray(array[:, gy], dtype=np.float64)
    wz = np.asarray(array[:, gz], dtype=np.float64)
    q_f = specific_force_magnitude(fx, fy, fz)
    q_w = angular_speed_magnitude(wx, wy, wz)
    return {
        "dataset": "EUROC",
        "sequence_id": sequence_id,
        "path": str(path),
        "header_line": header_line,
        "timestamp_s": ts_s,
        "q_f": q_f,
        "q_w": q_w,
        "accel_fields": ",".join(
            [mapping["accel_x_raw_field"], mapping["accel_y_raw_field"], mapping["accel_z_raw_field"]]
        ),
        "gyro_fields": ",".join(
            [mapping["gyro_x_raw_field"], mapping["gyro_y_raw_field"], mapping["gyro_z_raw_field"]]
        ),
        "q_f_unit": "m/s^2",
        "q_w_unit": "rad/s",
        "native_first_ns": str(int(as_uint64(array[:, ts_idx])[0])),
    }


def load_uzh_native(root: Path, sequence_id: str) -> dict[str, Any]:
    path = uzh_imu_path(root, sequence_id)
    columns, array, comments = read_text_table(path)
    parsed = interpret_imu_table(columns, array, comments)
    ts_s = np.asarray(parsed["timestamps_s"], dtype=np.float64)
    accel = np.asarray(parsed["accel"], dtype=np.float64)
    gyro = np.asarray(parsed["gyro"], dtype=np.float64)
    q_f = specific_force_magnitude(accel[:, 0], accel[:, 1], accel[:, 2])
    q_w = angular_speed_magnitude(gyro[:, 0], gyro[:, 1], gyro[:, 2])
    return {
        "dataset": "UZH_FPV",
        "sequence_id": sequence_id,
        "path": str(path),
        "header_line": comments[-1] if comments else ",".join(columns),
        "timestamp_s": ts_s,
        "q_f": q_f,
        "q_w": q_w,
        "accel_fields": "lin_acc_x,lin_acc_y,lin_acc_z",
        "gyro_fields": "ang_vel_x,ang_vel_y,ang_vel_z",
        "q_f_unit": "m/s^2",
        "q_w_unit": "rad/s",
        "native_first_ns": "",
    }
