"""Canonical Stage-1 processing: invariant magnitudes, causal anti-alias, integer decimation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .causal_filter import apply_causal_sos, design_antialias_sos, integer_decimate


def process_invariant_stream(
    timestamp_s: np.ndarray,
    q_f: np.ndarray,
    q_w: np.ndarray,
    *,
    native_rate_hz: float,
    decimation_factor: int,
    output_rate_hz: float,
    warmup_s: float,
    sos: np.ndarray | None = None,
) -> dict[str, Any]:
    ts = np.asarray(timestamp_s, dtype=np.float64).reshape(-1)
    qf = np.asarray(q_f, dtype=np.float64).reshape(-1)
    qw = np.asarray(q_w, dtype=np.float64).reshape(-1)
    if ts.size != qf.size or ts.size != qw.size:
        raise ValueError("timestamp and channel lengths differ")
    if sos is None:
        sos = design_antialias_sos(native_rate_hz)
    qf_f = apply_causal_sos(sos, qf)
    qw_f = apply_causal_sos(sos, qw)
    qf_d, ts_d, idx = integer_decimate(qf_f, ts, decimation_factor)
    qw_d, _, _ = integer_decimate(qw_f, ts, decimation_factor)
    t0 = float(ts[0]) if ts.size else 0.0
    keep = ts_d >= (t0 + float(warmup_s) - 1e-15)
    return {
        "timestamp_s": ts_d[keep],
        "q_f": qf_d[keep],
        "q_w": qw_d[keep],
        "native_index": idx[keep],
        "n_native": int(ts.size),
        "n_decimated": int(ts_d.size),
        "n_processed": int(np.sum(keep)),
        "warmup_s": float(warmup_s),
        "decimation_factor": int(decimation_factor),
        "output_rate_hz": float(output_rate_hz),
        "sos": sos,
    }


def rate_from_timestamps(ts: np.ndarray) -> dict[str, float | None]:
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return {"median_hz": None, "max_abs_dev_from_100": None, "strictly_monotonic": False}
    dt = np.diff(ts)
    pos = dt[dt > 0]
    if pos.size == 0:
        return {"median_hz": None, "max_abs_dev_from_100": None, "strictly_monotonic": False}
    med = float(np.median(pos))
    hz = 1.0 / med if med > 0 else None
    dev = abs(hz - 100.0) if hz else None
    return {
        "median_hz": hz,
        "max_abs_dev_from_100": dev,
        "strictly_monotonic": bool(np.all(dt > 0)),
        "median_dt": med,
        "max_dt": float(np.max(pos)),
    }
