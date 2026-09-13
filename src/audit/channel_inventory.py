"""Channel descriptive statistics. Audit only — no resampling or cleaning."""

from __future__ import annotations

from typing import Any

import numpy as np

CHANNEL_COLUMNS = [
    "dataset",
    "sequence_id",
    "source_topic_or_file",
    "field_name",
    "message_type_or_dtype",
    "unit_if_verified",
    "frame_if_verified",
    "sample_count",
    "missing_count",
    "nonfinite_count",
    "min",
    "p01",
    "median",
    "mean",
    "std",
    "p99",
    "max",
    "constant_fraction",
    "native_rate_hz",
    "role",
    "evidence",
    "notes",
]


def channel_row(
    *,
    dataset: str,
    sequence_id: str,
    source: str,
    field_name: str,
    dtype: str,
    unit: str,
    frame: str,
    values: np.ndarray,
    native_rate_hz: float | None,
    role: str,
    evidence: str,
    notes: str = "",
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    finite = np.isfinite(x)
    n_finite = int(np.sum(finite))
    xf = x[finite]
    if xf.size == 0:
        stats = {k: None for k in ("min", "p01", "median", "mean", "std", "p99", "max")}
        constant_fraction = 1.0 if n else None
    else:
        stats = {
            "min": float(np.min(xf)),
            "p01": float(np.percentile(xf, 1)),
            "median": float(np.median(xf)),
            "mean": float(np.mean(xf)),
            "std": float(np.std(xf, ddof=0)),
            "p99": float(np.percentile(xf, 99)),
            "max": float(np.max(xf)),
        }
        span = float(np.max(xf) - np.min(xf))
        constant_fraction = 1.0 if span == 0.0 else float(np.mean(xf == xf[0]))
    return {
        "dataset": dataset,
        "sequence_id": sequence_id,
        "source_topic_or_file": source,
        "field_name": field_name,
        "message_type_or_dtype": dtype,
        "unit_if_verified": unit,
        "frame_if_verified": frame,
        "sample_count": n,
        "missing_count": int(n - n_finite),
        "nonfinite_count": int(np.sum(~finite)),
        "constant_fraction": constant_fraction,
        "native_rate_hz": native_rate_hz,
        "role": role,
        "evidence": evidence,
        "notes": notes,
        **stats,
    }
