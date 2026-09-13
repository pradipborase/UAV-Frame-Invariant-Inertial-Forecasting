from __future__ import annotations

from audit.catalogs import SEQUENCE_INVENTORY_COLUMNS, blackbird_sequences, uzh_sequences
from audit.channel_inventory import CHANNEL_COLUMNS, channel_row
from audit.sequence_audit import ALLOWED_STATUS, classify_sequence
import numpy as np


def test_sequence_inventory_columns_stable() -> None:
    required = {
        "dataset",
        "sequence_id",
        "recording_id",
        "platform",
        "sensor_platform",
        "environment",
        "trajectory_family",
        "yaw_condition",
        "speed_condition",
        "duration_s",
        "imu_available",
        "gt_available",
        "public_gt",
        "calibration_available",
        "same_trajectory_group",
        "potential_dependency_group",
        "independent_recording_evidence",
        "independence_confidence",
    }
    assert required.issubset(set(SEQUENCE_INVENTORY_COLUMNS))


def test_blackbird_and_uzh_catalogs_nonempty() -> None:
    bb = blackbird_sequences()
    uzh = uzh_sequences()
    assert len(bb) >= 100
    assert any(r["public_gt"] == "YES" for r in uzh)
    assert any(r["public_gt"] == "NO" for r in uzh)
    withheld = [r for r in uzh if r["public_gt"] == "NO"]
    assert all("WITHHELD" in r["ground_truth_source_type"] for r in withheld)


def test_usability_statuses() -> None:
    not_dl = classify_sequence({"dataset": "BLACKBIRD", "sequence_id": "x", "downloaded": False, "public_gt": "YES"})
    assert not_dl["usable_status"] in ALLOWED_STATUS
    assert not_dl["usable_status"] == "NOT_DOWNLOADED"
    withheld = classify_sequence({"dataset": "UZH_FPV", "sequence_id": "y", "downloaded": False, "public_gt": "NO"})
    assert withheld["usable_status"] == "GT_WITHHELD"


def test_channel_schema_and_nan_count() -> None:
    values = np.array([1.0, 2.0, np.nan, np.inf, 2.0])
    row = channel_row(
        dataset="UZH_FPV",
        sequence_id="fixture",
        source="imu.txt",
        field_name="ax",
        dtype="float64",
        unit="UNRESOLVED",
        frame="S",
        values=values,
        native_rate_hz=500.0,
        role="IMU_ACCEL",
        evidence="fixture",
    )
    for col in CHANNEL_COLUMNS:
        assert col in row
    assert row["nonfinite_count"] == 2
    assert row["sample_count"] == 5
