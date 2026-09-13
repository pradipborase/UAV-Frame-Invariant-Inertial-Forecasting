"""Synthetic tests for EuRoC ASL CSV header and sensor.yaml parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from audit.euroc_schema import (
    load_sensor_yaml,
    map_euroc_gt_header,
    map_euroc_imu_header,
    parse_opencv_matrix,
    read_euroc_csv,
    split_euroc_header,
    timestamps_to_seconds,
)


def test_split_euroc_header_keeps_units() -> None:
    line = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
    headers = split_euroc_header(line)
    assert headers[0] == "timestamp [ns]"
    assert headers[4] == "a_RS_S_x [m s^-2]"
    mapped = map_euroc_imu_header(headers)
    assert mapped["layout"] == "HEADER_MAPPED"
    assert mapped["gyro_x_raw_field"] == "w_RS_S_x [rad s^-1]"
    assert mapped["accel_x_raw_field"] == "a_RS_S_x [m s^-2]"
    assert mapped["acceleration_unit_from_header"] == "m/s^2"
    assert mapped["gyro_unit_from_header"] == "rad/s"
    assert mapped["timestamp_unit_from_header"] == "nanoseconds"


def test_accel_mapping_does_not_use_bare_letter_a() -> None:
    headers = [
        "timestamp [ns]",
        "w_RS_S_x [rad s^-1]",
        "w_RS_S_y [rad s^-1]",
        "w_RS_S_z [rad s^-1]",
        "a_RS_S_x [m s^-2]",
        "a_RS_S_y [m s^-2]",
        "a_RS_S_z [m s^-2]",
    ]
    mapped = map_euroc_imu_header(headers)
    assert mapped["gyro_indices"] == (1, 2, 3)
    assert mapped["accel_indices"] == (4, 5, 6)


def test_read_euroc_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text(
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        "1403638128145099776,0.1,0.2,0.3,9.8,0.1,-0.2\n"
        "1403638128150099776,0.1,0.2,0.3,9.7,0.0,-0.1\n",
        encoding="utf-8",
    )
    headers, array, header_line = read_euroc_csv(path)
    assert "a_RS_S_x" in header_line
    assert array.shape == (2, 7)
    assert array[0, 0] == 1403638128145099776
    mapped = map_euroc_imu_header(headers)
    ts_s, unit = timestamps_to_seconds(array[:, 0], mapped["timestamp_unit_from_header"])
    assert unit == "nanoseconds"
    assert abs(float(ts_s[1] - ts_s[0]) - 0.005) < 1e-12
    assert float(ts_s[0]) == 0.0


def test_sensor_yaml_t_bs(tmp_path: Path) -> None:
    data = {
        "sensor_type": "imu",
        "comment": "VI-Sensor IMU (ADIS16448)",
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": [1.0, 0.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.02, 0.0, 0.0, 1.0, 0.03, 0.0, 0.0, 0.0, 1.0],
        },
        "rate_hz": 200,
        "gyroscope_noise_density": 1.7e-4,
        "gyroscope_random_walk": 1.9e-5,
        "accelerometer_noise_density": 2.0e-3,
        "accelerometer_random_walk": 3.0e-3,
    }
    path = tmp_path / "sensor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    parsed = load_sensor_yaml(path)
    assert parsed["sensor_type"] == "imu"
    assert parsed["rate_hz"] == 200
    assert parsed["T_BS"].shape == (4, 4)
    np.testing.assert_allclose(parsed["translation"], [0.01, 0.02, 0.03])
    mat = parse_opencv_matrix(data["T_BS"])
    assert mat is not None
    assert mat[0, 3] == 0.01


def test_gt_header_bias_is_not_direct_acceleration() -> None:
    headers = [
        "timestamp",
        "p_RS_R_x [m]",
        "p_RS_R_y [m]",
        "p_RS_R_z [m]",
        "q_RS_w []",
        "q_RS_x []",
        "q_RS_y []",
        "q_RS_z []",
        "v_RS_R_x [m s^-1]",
        "v_RS_R_y [m s^-1]",
        "v_RS_R_z [m s^-1]",
        "b_w_RS_S_x [rad s^-1]",
        "b_w_RS_S_y [rad s^-1]",
        "b_w_RS_S_z [rad s^-1]",
        "b_a_RS_S_x [m s^-2]",
        "b_a_RS_S_y [m s^-2]",
        "b_a_RS_S_z [m s^-2]",
    ]
    mapped = map_euroc_gt_header(headers)
    assert mapped["quaternion_order_if_applicable"] == "wxyz"
    assert mapped["acceleration_fields"] == "NO_DIRECT_GT_ACCELERATION"
    assert mapped["bias_accel_present"] is True
    assert "v_RS_R_x" in mapped["velocity_fields"]
