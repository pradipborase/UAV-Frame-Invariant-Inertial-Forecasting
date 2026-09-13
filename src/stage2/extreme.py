"""Pre-modelling extreme-value audit of canonical q_f. No clipping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stage1.io_native import load_euroc_native, load_uzh_native

from .windows import SequenceWindows


def _duration_above(ts: np.ndarray, qf: np.ndarray, thresh: float) -> float:
    if ts.size < 2:
        return 0.0
    dt = float(np.median(np.diff(ts)))
    return float(np.sum(qf > thresh) * dt)


def _neighbor_consistency(qf: np.ndarray, i: int, half: int = 5) -> tuple[str, float]:
    lo = max(0, i - half)
    hi = min(int(qf.size), i + half + 1)
    neigh = np.concatenate([qf[lo:i], qf[i + 1 : hi]]) if hi > lo else np.asarray([], dtype=np.float64)
    if neigh.size == 0:
        return "NO_NEIGHBORS", float("nan")
    med = float(np.median(neigh))
    peak = float(qf[i])
    ratio = peak / med if med != 0 else float("inf")
    if ratio < 3.0:
        return "YES", ratio
    if ratio < 8.0:
        return "PARTIAL", ratio
    return "NO", ratio


def audit_sequence(
    pack: SequenceWindows,
    native: dict[str, Any] | None,
) -> dict[str, Any]:
    qf = pack.q_f
    qw = pack.q_w
    ts = pack.timestamp_s
    i = int(np.argmax(qf))
    cons, ratio = _neighbor_consistency(qf, i)
    n_eq_max = int(np.sum(np.abs(qf - qf[i]) <= 1e-9))
    spike = cons == "NO" and _duration_above(ts, qf, 100.0) < 0.05
    sat = bool(qf[i] >= 150.0 and n_eq_max >= 3)
    native_max = ""
    native_consistent = "NA"
    if native is not None:
        nqf = np.asarray(native["q_f"], dtype=np.float64)
        native_max = float(np.max(nqf)) if nqf.size else ""
        if nqf.size:
            native_consistent = "YES" if np.all(np.isfinite(nqf)) else "NO"
    finite = bool(np.all(np.isfinite(qf)) and np.all(np.isfinite(qw)))
    if not finite or native_consistent == "NO":
        action = "RAW_INTEGRITY_REVIEW_REQUIRED"
        interpretation = "non-finite or native stream inconsistency"
    else:
        action = "KEEP"
        interpretation = (
            "extreme values retained as observed specific-force magnitude; "
            "no clipping; not excluded for modelling"
        )
    return {
        "dataset": pack.dataset,
        "sequence_id": pack.sequence_id,
        "q_f_min": float(np.min(qf)),
        "q_f_p001": float(np.percentile(qf, 0.1)),
        "q_f_p01": float(np.percentile(qf, 1)),
        "q_f_p99": float(np.percentile(qf, 99)),
        "q_f_p999": float(np.percentile(qf, 99.9)),
        "q_f_max": float(qf[i]),
        "timestamp_of_max": float(ts[i]),
        "duration_above_30_mps2": _duration_above(ts, qf, 30.0),
        "duration_above_50_mps2": _duration_above(ts, qf, 50.0),
        "duration_above_100_mps2": _duration_above(ts, qf, 100.0),
        "q_w_at_qf_max": float(qw[i]),
        "neighboring_values_consistent": cons,
        "neighbor_median_ratio": ratio,
        "possible_sensor_saturation": "YES" if sat else "NO",
        "possible_single_sample_spike": "YES" if spike else "NO",
        "raw_source_consistent": native_consistent,
        "native_q_f_max": native_max,
        "n_samples_at_processed_max": n_eq_max,
        "integrity_interpretation": interpretation,
        "action": action,
    }


def load_native(root: Path, dataset: str, sequence_id: str) -> dict[str, Any]:
    if dataset == "EUROC":
        return load_euroc_native(root, sequence_id)
    return load_uzh_native(root, sequence_id)
