"""Sequence-level metrics in physical units. Windows are not inferential units."""

from __future__ import annotations

import numpy as np


def rmse_all(pred: np.ndarray, actual: np.ndarray) -> float:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def mae_all(pred: np.ndarray, actual: np.ndarray) -> float:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.mean(np.abs(err)))


def rmse_at_horizon(pred: np.ndarray, actual: np.ndarray, h: int) -> float:
    e = np.asarray(pred, dtype=np.float64)[:, h - 1] - np.asarray(actual, dtype=np.float64)[:, h - 1]
    return float(np.sqrt(np.mean(e * e)))


def rmse_horizon_curve(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return np.sqrt(np.mean(err * err, axis=0))


def skill_score(model_rmse: float, persistence_rmse: float) -> float:
    if persistence_rmse == 0.0:
        return float("nan") if model_rmse != 0.0 else 1.0
    return float(1.0 - model_rmse / persistence_rmse)


def equal_sequence_mean(values: list[float] | np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(np.mean(v))


def sequence_result_row(
    *,
    evaluation_context: str,
    dataset: str,
    sequence_id: str,
    model: str,
    pred: np.ndarray,
    actual: np.ndarray,
    persistence_rmse: float | None,
    n_origins: int,
    extra: dict[str, str] | None = None,
) -> dict[str, object]:
    curve = rmse_horizon_curve(pred, actual)
    rmse = rmse_all(pred, actual)
    row: dict[str, object] = {
        "evaluation_context": evaluation_context,
        "dataset": dataset,
        "sequence_id": sequence_id,
        "model": model,
        "n_origins": n_origins,
        "rmse_all_horizons": rmse,
        "mae_all_horizons": mae_all(pred, actual),
        "rmse_50ms": float(curve[4]),
        "rmse_100ms": float(curve[9]),
        "rmse_200ms": float(curve[19]),
        "skill": "" if persistence_rmse is None else skill_score(rmse, persistence_rmse),
    }
    for i, val in enumerate(curve, start=1):
        row[f"rmse_h{i:02d}"] = float(val)
    if extra:
        row.update(extra)
    return row
