"""Locked Stage-1 origins and window tensors. All models share the same origins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stage1.windows import count_windows


@dataclass
class SequenceWindows:
    dataset: str
    sequence_id: str
    group: str
    timestamp_s: np.ndarray
    q_f: np.ndarray
    q_w: np.ndarray
    native_index: np.ndarray
    origins: np.ndarray
    hist_qf: np.ndarray
    hist_qw: np.ndarray
    future_qf: np.ndarray
    p95_qf: float

    @property
    def n_windows(self) -> int:
        return int(self.origins.size)


def valid_origins(
    timestamp_s: np.ndarray,
    *,
    lookback_samples: int = 100,
    horizon_samples: int = 20,
    stride_samples: int = 5,
    gap_factor: float = 2.0,
) -> np.ndarray:
    ts = np.asarray(timestamp_s, dtype=np.float64).reshape(-1)
    n = int(ts.size)
    l_win = int(lookback_samples)
    h_win = int(horizon_samples)
    stride = int(stride_samples)
    if n < 2:
        dt_med = None
    else:
        pos = np.diff(ts)
        pos = pos[pos > 0]
        dt_med = float(np.median(pos)) if pos.size else None
    kept: list[int] = []
    for k in range(0, n, stride):
        if k < l_win - 1:
            continue
        if k + h_win >= n:
            continue
        sl = ts[k - l_win + 1 : k + h_win + 1]
        if sl.size != l_win + h_win:
            continue
        dts = np.diff(sl)
        if np.any(dts <= 0):
            continue
        if dt_med is not None and np.any(dts > gap_factor * dt_med):
            continue
        kept.append(k)
    return np.asarray(kept, dtype=np.int64)


def build_sequence_windows(
    rec: dict[str, Any],
    *,
    lookback_samples: int = 100,
    horizon_samples: int = 20,
    stride_samples: int = 5,
    high_percentile: float = 95.0,
) -> SequenceWindows:
    ts = np.asarray(rec["timestamp_s"], dtype=np.float64)
    qf = np.asarray(rec["q_f"], dtype=np.float64)
    qw = np.asarray(rec["q_w"], dtype=np.float64)
    native_index = np.asarray(rec["native_index"], dtype=np.int64)
    origins = valid_origins(
        ts,
        lookback_samples=lookback_samples,
        horizon_samples=horizon_samples,
        stride_samples=stride_samples,
    )
    audit = count_windows(
        ts,
        lookback_samples=lookback_samples,
        horizon_samples=horizon_samples,
        stride_samples=stride_samples,
    )
    if int(origins.size) != int(audit["valid_origins"]):
        raise RuntimeError(
            f"origin count mismatch for {rec['sequence_id']}: {origins.size} vs audit {audit['valid_origins']}"
        )
    hist_idx = origins[:, None] + np.arange(-lookback_samples + 1, 1, dtype=np.int64)
    fut_idx = origins[:, None] + np.arange(1, horizon_samples + 1, dtype=np.int64)
    return SequenceWindows(
        dataset=str(rec["dataset"]),
        sequence_id=str(rec["sequence_id"]),
        group=str(rec["dependency_group"]),
        timestamp_s=ts,
        q_f=qf,
        q_w=qw,
        native_index=native_index,
        origins=origins,
        hist_qf=qf[hist_idx],
        hist_qw=qw[hist_idx],
        future_qf=qf[fut_idx],
        p95_qf=float(np.percentile(qf, high_percentile)),
    )


def lag_slice(hist: np.ndarray, lag: int) -> np.ndarray:
    lag = int(lag)
    if lag < 1 or lag > hist.shape[1]:
        raise ValueError(f"lag {lag} incompatible with history {hist.shape[1]}")
    return hist[:, hist.shape[1] - lag :]
