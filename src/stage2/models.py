"""Four Stage-2 baselines. Direct multi-output only. No recursive rollout."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge

from .weights import SourceStandardScaler, equal_recording_weights
from .windows import SequenceWindows, lag_slice


def persistence_predict(hist_qf: np.ndarray, horizon: int = 20) -> np.ndarray:
    last = np.asarray(hist_qf, dtype=np.float64)[:, -1]
    return np.repeat(last[:, None], int(horizon), axis=1)


def linear_extrapolation_predict(hist_qf_k: np.ndarray, *, dt_s: float = 0.01, horizon: int = 20) -> np.ndarray:
    """OLS line on K past/current samples ending at the origin (t=0). Past only."""
    y = np.asarray(hist_qf_k, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("history must be (n_origins, K)")
    k = y.shape[1]
    x = (np.arange(k, dtype=np.float64) - (k - 1)) * float(dt_s)
    x_c = x - x.mean()
    denom = float(np.dot(x_c, x_c))
    y_mean = y.mean(axis=1)
    slope = (y - y_mean[:, None]) @ x_c / denom if denom > 0 else np.zeros(y.shape[0])
    intercept = y_mean - slope * x.mean()
    t_fut = np.arange(1, int(horizon) + 1, dtype=np.float64) * float(dt_s)
    return intercept[:, None] + slope[:, None] * t_fut[None, :]


def stack_features(
    packs: list[SequenceWindows],
    *,
    lag: int,
    use_qw: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    ids = []
    for pack in packs:
        qf = lag_slice(pack.hist_qf, lag)
        if use_qw:
            qw = lag_slice(pack.hist_qw, lag)
            x = np.concatenate([qf, qw], axis=1)
        else:
            x = qf
        xs.append(x)
        ys.append(pack.future_qf)
        ids.append(np.repeat(pack.sequence_id, pack.n_windows))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(ids, axis=0)


@dataclass
class RidgePack:
    model_name: str
    lag: int
    alpha: float
    use_qw: bool
    ridge: Ridge
    scaler: SourceStandardScaler
    train_sequences: list[str]
    train_dataset: str
    n_windows: int
    weights_sum_by_sequence: dict[str, float] = field(default_factory=dict)
    coef_norm: float = 0.0
    intercept_norm: float = 0.0
    n_features: int = 0
    n_outputs: int = 20


def fit_ridge(
    packs: list[SequenceWindows],
    *,
    lag: int,
    alpha: float,
    use_qw: bool,
    dataset: str,
    model_name: str,
    fit_intercept: bool = True,
) -> RidgePack:
    x, y, ids = stack_features(packs, lag=lag, use_qw=use_qw)
    w = equal_recording_weights(ids)
    scaler = SourceStandardScaler()
    scaler.fit(x, w, ids, dataset=dataset, purpose="feature_scaler")
    xs = scaler.transform(x)
    ridge = Ridge(alpha=float(alpha), fit_intercept=fit_intercept)
    ridge.fit(xs, y, sample_weight=w)
    coef = np.asarray(ridge.coef_, dtype=np.float64)
    intercept = np.asarray(ridge.intercept_, dtype=np.float64)
    recs = sorted({str(s) for s in ids.tolist()})
    w_by = {r: float(np.sum(w[ids == r])) for r in recs}
    return RidgePack(
        model_name=model_name,
        lag=int(lag),
        alpha=float(alpha),
        use_qw=bool(use_qw),
        ridge=ridge,
        scaler=scaler,
        train_sequences=recs,
        train_dataset=dataset,
        n_windows=int(x.shape[0]),
        weights_sum_by_sequence=w_by,
        coef_norm=float(np.linalg.norm(coef)),
        intercept_norm=float(np.linalg.norm(intercept)),
        n_features=int(x.shape[1]),
        n_outputs=int(y.shape[1]),
    )


def ridge_predict(pack: RidgePack, seq: SequenceWindows) -> np.ndarray:
    qf = lag_slice(seq.hist_qf, pack.lag)
    if pack.use_qw:
        qw = lag_slice(seq.hist_qw, pack.lag)
        x = np.concatenate([qf, qw], axis=1)
    else:
        x = qf
    xs = pack.scaler.transform(x)
    pred = np.asarray(pack.ridge.predict(xs), dtype=np.float64)
    if pred.shape != seq.future_qf.shape:
        raise RuntimeError(f"prediction shape {pred.shape} != {seq.future_qf.shape}")
    return pred


def local_predict(seq: SequenceWindows, model: str, *, linear_k: int = 10, dt_s: float = 0.01) -> np.ndarray:
    if model == "B0_PERSISTENCE":
        return persistence_predict(seq.hist_qf, horizon=seq.future_qf.shape[1])
    if model == "B1_LINEAR":
        return linear_extrapolation_predict(seq.hist_qf[:, -int(linear_k) :], dt_s=dt_s, horizon=seq.future_qf.shape[1])
    raise ValueError(model)
