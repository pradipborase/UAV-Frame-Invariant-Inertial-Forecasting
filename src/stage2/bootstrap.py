"""Sequence-level bootstrap and exact paired sign-flip test. Exploratory only."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_reps: int = 10000,
    seed: int = 20260912,
    alpha: float = 0.05,
) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n, size=(int(n_reps), n))
    means = v[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean": float(np.mean(v)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def paired_sign_flip_p(delta: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value of the mean difference. n must be small."""
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    n = int(d.size)
    if n == 0:
        return float("nan")
    if n > 20:
        raise ValueError("exact sign-flip is only used for small sequence n")
    obs = float(np.mean(d))
    abs_obs = abs(obs)
    total = 1 << n
    count = 0
    for mask in range(total):
        signed_sum = 0.0
        for i in range(n):
            signed_sum += d[i] if (mask >> i) & 1 else -d[i]
        if abs(signed_sum / n) + 1e-18 >= abs_obs:
            count += 1
    return float(count / total)


def win_loss_tie(delta: np.ndarray, *, atol: float = 1e-12) -> tuple[int, int, int]:
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    wins = int(np.sum(d < -atol))
    losses = int(np.sum(d > atol))
    ties = int(d.size - wins - losses)
    return wins, losses, ties
