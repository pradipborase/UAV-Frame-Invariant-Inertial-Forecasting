"""Equal-recording window weights and source-train-only feature scaling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def equal_recording_weights(sequence_ids: list[str] | np.ndarray) -> np.ndarray:
    """Each recording contributes total weight 1, then weights are normalized to sum 1."""
    ids = np.asarray(sequence_ids)
    n = ids.size
    if n == 0:
        return np.asarray([], dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    recs, counts = np.unique(ids, return_counts=True)
    n_rec = float(recs.size)
    count_map = {str(r): int(c) for r, c in zip(recs, counts, strict=True)}
    for i, sid in enumerate(ids):
        w[i] = (1.0 / count_map[str(sid)]) / n_rec
    return w


@dataclass
class SourceStandardScaler:
    """Weighted standard scaler. Fit on source-training windows only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    fit_recording_ids: list[str] = field(default_factory=list)
    fit_dataset: str = ""
    fit_purpose: str = "feature_scaler"
    n_windows: int = 0

    def fit(
        self,
        x: np.ndarray,
        sample_weight: np.ndarray,
        recording_ids: list[str] | np.ndarray,
        *,
        dataset: str,
        purpose: str = "feature_scaler",
    ) -> SourceStandardScaler:
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError("X must be 2-D")
        if w.size != x.shape[0]:
            raise ValueError("sample_weight length mismatch")
        w = w / np.sum(w)
        mean = np.sum(w[:, None] * x, axis=0)
        var = np.sum(w[:, None] * (x - mean) ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.where(std < 1e-12, 1.0, std)
        self.mean_ = mean
        self.std_ = std
        self.fit_recording_ids = sorted({str(s) for s in np.asarray(recording_ids).tolist()})
        self.fit_dataset = dataset
        self.fit_purpose = purpose
        self.n_windows = int(x.shape[0])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("scaler has not been fit")
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_
