from __future__ import annotations

import numpy as np

from audit.timestamp_audit import audit_timestamps, overlap_seconds


def test_duplicate_and_backward_detection() -> None:
    ts = np.array([0.0, 0.1, 0.1, 0.05, 0.3], dtype=np.float64)
    stats = audit_timestamps(ts, "seconds")
    assert stats["sample_count"] == 5
    assert stats["duplicate_count"] == 1
    assert stats["backward_count"] == 1
    assert stats["strictly_monotonic"] is False


def test_large_gap_counts() -> None:
    ts = np.array([0.0, 0.01, 0.02, 0.03, 0.50], dtype=np.float64)
    stats = audit_timestamps(ts, "seconds")
    assert stats["number_gaps_gt_2x_median"] >= 1
    assert stats["number_gaps_gt_10x_median"] >= 1
    assert abs(stats["maximum_gap"] - 0.47) < 1e-12


def test_gap_threshold_exact() -> None:
    ts = np.array([0.0, 1.0, 2.0, 3.0, 14.0])
    stats = audit_timestamps(ts, "seconds")
    assert stats["median_dt"] == 1.0
    assert stats["number_gaps_gt_2x_median"] == 1
    assert stats["number_gaps_gt_5x_median"] == 1
    assert stats["number_gaps_gt_10x_median"] == 1


def test_nan_inf_not_in_timestamp_array_handling() -> None:
    ts = np.array([0.0, np.nan, 0.2, np.inf, 0.4])
    stats = audit_timestamps(ts, "seconds")
    assert stats["sample_count"] == 3
    assert np.isfinite(stats["first_timestamp"])


def test_overlap() -> None:
    ov, frac = overlap_seconds(0.0, 10.0, 5.0, 20.0)
    assert ov == 5.0
    assert frac == 5.0 / 20.0
