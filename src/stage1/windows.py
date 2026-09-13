"""Window counting only. No predictions are generated."""

from __future__ import annotations

from typing import Any

import numpy as np


def count_windows(
    timestamp_s: np.ndarray,
    *,
    lookback_samples: int = 100,
    horizon_samples: int = 20,
    stride_samples: int = 5,
    gap_factor: float = 2.0,
) -> dict[str, Any]:
    ts = np.asarray(timestamp_s, dtype=np.float64).reshape(-1)
    n = int(ts.size)
    L = int(lookback_samples)
    H = int(horizon_samples)
    S = int(stride_samples)
    if n < 2:
        dt_med = None
    else:
        pos = np.diff(ts)
        pos = pos[pos > 0]
        dt_med = float(np.median(pos)) if pos.size else None
    candidates = list(range(0, n, S)) if n else []
    excluded_start = 0
    excluded_end = 0
    excluded_gap = 0
    excluded_other = 0
    valid = 0
    for k in candidates:
        if k < L - 1:
            excluded_start += 1
            continue
        if k + H >= n:
            excluded_end += 1
            continue
        sl = ts[k - L + 1 : k + H + 1]
        if sl.size != L + H:
            excluded_other += 1
            continue
        dts = np.diff(sl)
        if np.any(dts <= 0):
            excluded_other += 1
            continue
        if dt_med is not None and np.any(dts > gap_factor * dt_med):
            excluded_gap += 1
            continue
        valid += 1
    return {
        "processed_samples": n,
        "candidate_origins": len(candidates),
        "valid_origins": valid,
        "excluded_due_to_start": excluded_start,
        "excluded_due_to_end": excluded_end,
        "excluded_due_to_gap": excluded_gap,
        "excluded_other": excluded_other,
        "lookback_samples": L,
        "horizon_samples": H,
        "stride_samples": S,
    }
