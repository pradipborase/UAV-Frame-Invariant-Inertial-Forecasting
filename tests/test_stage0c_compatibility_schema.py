"""Compatibility table schema tests (synthetic rows, no real datasets)."""

from __future__ import annotations

import csv
from pathlib import Path

from audit.stage0c import COMPAT_COLUMNS, UNIT_COMPAT_COLUMNS, write_csv
from audit.stage0c_run import build_compatibility


def test_compatibility_columns_stable() -> None:
    assert COMPAT_COLUMNS == [
        "item",
        "EUROC",
        "UZH_FPV",
        "compatible",
        "fixed_conversion_required",
        "target_fitted_information_required",
        "severity",
        "evidence",
        "recommendation",
    ]


def test_build_compatibility_schema() -> None:
    fake_audit = {
        "imu": {
            "interpreted_acceleration_quantity": "RAW_ACCELEROMETER_SPECIFIC_FORCE",
            "acceleration_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "measured_median_rate_hz": 200.0,
            "sensor_model": "ADIS16448",
            "transform_direction": "T_BS maps sensor-frame coordinates to body-frame coordinates (S -> B)",
        }
    }
    rows, overall = build_compatibility([fake_audit], uzh_ok=True)
    assert overall == "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY"
    for row in rows:
        assert set(COMPAT_COLUMNS).issubset(row.keys())
    items = [r["item"] for r in rows]
    assert "accelerometer physical quantity" in items
    assert "zero-shot scaling feasibility" in items


def test_write_compat_csv(tmp_path: Path) -> None:
    rows = [
        {
            "item": "OVERALL_CLASSIFICATION",
            "EUROC": "x",
            "UZH_FPV": "y",
            "compatible": "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY",
            "fixed_conversion_required": "NO",
            "target_fitted_information_required": "NO",
            "severity": "INFO",
            "evidence": "fixture",
            "recommendation": "ok",
        }
    ]
    path = tmp_path / "compat.csv"
    write_csv(path, COMPAT_COLUMNS, rows)
    with path.open("r", encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["compatible"] == "COMPARABLE_FOR_SELECTED_QUANTITIES_ONLY"
    assert UNIT_COMPAT_COLUMNS[0] == "quantity"
