"""Timestamp integrity statistics computed from raw timestamp arrays.

No resampling, interpolation, or synchronization is performed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

REQUIRED_FIELDS = (
    "sample_count",
    "first_timestamp",
    "last_timestamp",
    "duration",
    "timestamp_unit",
    "strictly_monotonic",
    "backward_count",
    "duplicate_count",
    "median_dt",
    "mean_dt",
    "std_dt",
    "p01_dt",
    "p05_dt",
    "p95_dt",
    "p99_dt",
    "median_frequency",
    "maximum_gap",
    "number_gaps_gt_2x_median",
    "number_gaps_gt_5x_median",
    "number_gaps_gt_10x_median",
)


def _empty_stats(timestamp_unit: str) -> dict[str, Any]:
    return {
        "sample_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "duration": None,
        "timestamp_unit": timestamp_unit,
        "strictly_monotonic": False,
        "backward_count": 0,
        "duplicate_count": 0,
        "median_dt": None,
        "mean_dt": None,
        "std_dt": None,
        "p01_dt": None,
        "p05_dt": None,
        "p95_dt": None,
        "p99_dt": None,
        "median_frequency": None,
        "maximum_gap": None,
        "number_gaps_gt_2x_median": 0,
        "number_gaps_gt_5x_median": 0,
        "number_gaps_gt_10x_median": 0,
    }


def audit_timestamps(timestamps: np.ndarray, timestamp_unit: str) -> dict[str, Any]:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        return _empty_stats(timestamp_unit)
    order = np.argsort(ts, kind="mergesort")
    ts_sorted_check = ts[order]
    if not np.array_equal(ts, ts_sorted_check):
        # Statistics are computed on the provided order, which is the native stream order.
        pass
    diffs = np.diff(ts)
    backward = int(np.sum(diffs < 0))
    duplicate = int(np.sum(diffs == 0))
    strictly_monotonic = bool(ts.size >= 2 and np.all(diffs > 0))
    positive = diffs[diffs > 0]
    stats = _empty_stats(timestamp_unit)
    stats.update(
        {
            "sample_count": int(ts.size),
            "first_timestamp": float(ts[0]),
            "last_timestamp": float(ts[-1]),
            "duration": float(ts[-1] - ts[0]),
            "strictly_monotonic": strictly_monotonic,
            "backward_count": backward,
            "duplicate_count": duplicate,
        }
    )
    if positive.size == 0:
        return stats
    median_dt = float(np.median(positive))
    stats.update(
        {
            "median_dt": median_dt,
            "mean_dt": float(np.mean(positive)),
            "std_dt": float(np.std(positive, ddof=0)),
            "p01_dt": float(np.percentile(positive, 1)),
            "p05_dt": float(np.percentile(positive, 5)),
            "p95_dt": float(np.percentile(positive, 95)),
            "p99_dt": float(np.percentile(positive, 99)),
            "median_frequency": (1.0 / median_dt) if median_dt > 0 else None,
            "maximum_gap": float(np.max(positive)),
            "number_gaps_gt_2x_median": int(np.sum(positive > 2.0 * median_dt)),
            "number_gaps_gt_5x_median": int(np.sum(positive > 5.0 * median_dt)),
            "number_gaps_gt_10x_median": int(np.sum(positive > 10.0 * median_dt)),
        }
    )
    return stats


def infer_timestamp_unit_from_values(timestamps: np.ndarray) -> str:
    """Heuristic unit label for reporting only. Not a physical conversion."""
    ts = np.asarray(timestamps, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    if ts.size < 2:
        return "UNRESOLVED"
    span = float(np.nanmax(ts) - np.nanmin(ts))
    median_dt = float(np.median(np.diff(np.sort(ts))))
    if median_dt <= 0:
        return "UNRESOLVED"
    if 1e6 <= median_dt <= 1e10:
        return "nanoseconds_candidate"
    if 1e3 <= median_dt <= 1e6:
        return "microseconds_candidate"
    if 0.0001 <= median_dt <= 1.0:
        return "seconds_candidate"
    if span > 1e12:
        return "nanoseconds_candidate"
    return "UNRESOLVED"


def overlap_seconds(a_first: float | None, a_last: float | None, b_first: float | None, b_last: float | None) -> tuple[float | None, float | None]:
    if None in (a_first, a_last, b_first, b_last):
        return None, None
    start = max(float(a_first), float(b_first))
    end = min(float(a_last), float(b_last))
    if end <= start:
        return 0.0, 0.0
    overlap = end - start
    union = max(float(a_last), float(b_last)) - min(float(a_first), float(b_first))
    frac = (overlap / union) if union > 0 else None
    return overlap, frac
