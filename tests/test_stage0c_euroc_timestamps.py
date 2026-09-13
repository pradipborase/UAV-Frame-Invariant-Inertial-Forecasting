"""Timestamp, overlap, and SHA-256 tests for Stage 0C (synthetic only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audit.euroc_schema import compute_gt_overlap, extra_dt_frequency_stats, timestamps_to_seconds
from audit.hashing import sha256_bytes, sha256_file
from audit.timestamp_audit import audit_timestamps


def test_nanosecond_timestamps_to_seconds() -> None:
    ts = np.array([1_000_000_000.0, 1_005_000_000.0, 1_010_000_000.0])
    ts_s, unit = timestamps_to_seconds(ts, "nanoseconds")
    assert unit == "nanoseconds"
    np.testing.assert_allclose(ts_s, [0.0, 0.005, 0.010])


def test_duplicate_and_backward_timestamps() -> None:
    ts = np.array([1.0, 1.0, 0.9, 1.2])
    stats = audit_timestamps(ts, "seconds")
    assert stats["duplicate_count"] == 1
    assert stats["backward_count"] == 1
    assert stats["strictly_monotonic"] is False


def test_frequency_from_dt() -> None:
    dt = 1.0 / 200.0
    ts = np.arange(0.0, 1.0, dt)
    extra = extra_dt_frequency_stats(ts)
    assert extra["measured_median_rate_hz"] is not None
    assert abs(extra["measured_median_rate_hz"] - 200.0) < 1e-6
    stats = audit_timestamps(ts, "seconds")
    assert abs(float(stats["median_frequency"]) - 200.0) < 1e-6


def test_gt_overlap_computation() -> None:
    imu = np.array([0.0, 10.0, 30.0, 40.0])
    gt = np.array([5.0, 15.0, 25.0, 35.0])
    ov = compute_gt_overlap(imu, gt)
    assert abs(ov["overlap_duration_s"] - 30.0) < 1e-9
    empty = compute_gt_overlap(np.array([]), gt)
    assert empty["overlap_duration_s"] == 0.0


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"euroc-stage0c-fixture"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)
