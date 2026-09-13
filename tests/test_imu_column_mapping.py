"""Synthetic IMU header mapping — must not confuse ang_vel with lin_acc."""

from __future__ import annotations

import numpy as np

from audit.text_reader import interpret_imu_table


def test_uzh_style_header_maps_lin_acc_not_ang_vel() -> None:
    columns = [
        "index",
        "timestamp",
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
        "lin_acc_x",
        "lin_acc_y",
        "lin_acc_z",
    ]
    array = np.array(
        [
            [0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 9.8],
            [1.0, 1.002, 0.11, 0.21, 0.31, 0.41, 0.51, 9.81],
        ]
    )
    parsed = interpret_imu_table(columns, array, comments=["# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z"])
    assert parsed["accel_fields"] == {"x": "lin_acc_x", "y": "lin_acc_y", "z": "lin_acc_z"}
    assert parsed["gyro_fields"] == {"x": "ang_vel_x", "y": "ang_vel_y", "z": "ang_vel_z"}
    assert parsed["timestamp_field"] == "timestamp"
    np.testing.assert_allclose(parsed["accel"][0], [0.4, 0.5, 9.8])
    np.testing.assert_allclose(parsed["gyro"][0], [0.1, 0.2, 0.3])
    assert parsed["unit_acceleration_from_header"] == "UNRESOLVED"


def test_header_unit_token_is_not_invented() -> None:
    columns = ["timestamp", "wx", "wy", "wz", "ax", "ay", "az"]
    array = np.zeros((3, 7))
    parsed = interpret_imu_table(columns, array, comments=["# no units here"])
    assert parsed["unit_acceleration_from_header"] == "UNRESOLVED"
    assert parsed["unit_gyro_from_header"] == "UNRESOLVED"
