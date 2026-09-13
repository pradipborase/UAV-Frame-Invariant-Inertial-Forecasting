"""Persistence, linear causality, Ridge shapes, weights, metrics, bootstrap."""

from __future__ import annotations

import numpy as np

from stage2.bootstrap import bootstrap_mean_ci, paired_sign_flip_p, win_loss_tie
from stage2.metrics import mae_all, rmse_all, rmse_at_horizon, skill_score
from stage2.models import linear_extrapolation_predict, persistence_predict
from stage2.protocol import verify_protocol_lock
from stage2.weights import SourceStandardScaler, equal_recording_weights
from stage2.windows import lag_slice, valid_origins

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_lock_matches_stage1() -> None:
    lock = verify_protocol_lock(ROOT)
    assert lock["match"] == "PASS", lock


def test_persistence_formula() -> None:
    hist = np.array([[1.0, 2.0, 3.5], [0.0, 0.0, 9.0]])
    pred = persistence_predict(hist, horizon=20)
    assert pred.shape == (2, 20)
    assert np.all(pred[0] == 3.5)
    assert np.all(pred[1] == 9.0)


def test_linear_extrapolation_causality() -> None:
    rng = np.random.default_rng(0)
    n = 40
    a = rng.normal(size=n)
    b = a.copy()
    cut = 20
    b[cut + 1 :] += 5.0
    hist_a = np.stack([a[i - 9 : i + 1] for i in range(9, cut + 1)])
    hist_b = np.stack([b[i - 9 : i + 1] for i in range(9, cut + 1)])
    pa = linear_extrapolation_predict(hist_a, dt_s=0.01, horizon=20)
    pb = linear_extrapolation_predict(hist_b, dt_s=0.01, horizon=20)
    np.testing.assert_allclose(pa, pb, rtol=1e-12, atol=1e-12)


def test_linear_known_slope() -> None:
    t = np.arange(10) - 9
    y = 2.0 + 3.0 * t * 0.01
    pred = linear_extrapolation_predict(y[None, :], dt_s=0.01, horizon=4)
    expected = 2.0 + 3.0 * np.array([0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(pred[0], expected, rtol=1e-12, atol=1e-12)


def test_ridge_direct_output_shapes() -> None:
    from stage2.models import fit_ridge
    from stage2.windows import SequenceWindows

    rng = np.random.default_rng(1)
    n = 300
    qf = rng.normal(size=n) + 9.8
    qw = rng.normal(size=n) * 0.1
    ts = np.arange(n, dtype=np.float64) * 0.01
    origins = valid_origins(ts)
    hist_idx = origins[:, None] + np.arange(-99, 1)
    fut_idx = origins[:, None] + np.arange(1, 21)
    pack = SequenceWindows(
        dataset="EUROC",
        sequence_id="toy_a",
        group="g1",
        timestamp_s=ts,
        q_f=qf,
        q_w=qw,
        native_index=np.arange(n),
        origins=origins,
        hist_qf=qf[hist_idx],
        hist_qw=qw[hist_idx],
        future_qf=qf[fut_idx],
        p95_qf=float(np.percentile(qf, 95)),
    )
    pack2 = SequenceWindows(
        dataset="EUROC",
        sequence_id="toy_b",
        group="g2",
        timestamp_s=ts,
        q_f=qf + 0.1,
        q_w=qw,
        native_index=np.arange(n),
        origins=origins,
        hist_qf=(qf + 0.1)[hist_idx],
        hist_qw=qw[hist_idx],
        future_qf=(qf + 0.1)[fut_idx],
        p95_qf=float(np.percentile(qf + 0.1, 95)),
    )
    fitted = fit_ridge([pack, pack2], lag=25, alpha=1.0, use_qw=False, dataset="EUROC", model_name="B2_RIDGE_QF")
    assert fitted.ridge.coef_.shape == (20, 25)
    from stage2.models import ridge_predict

    pred = ridge_predict(fitted, pack)
    assert pred.shape == pack.future_qf.shape
    x = lag_slice(pack.hist_qf, 25)
    assert x.shape[1] == 25


def test_equal_recording_weights() -> None:
    ids = ["a"] * 2 + ["b"] * 8
    w = equal_recording_weights(ids)
    assert abs(float(w[:2].sum()) - 0.5) < 1e-12
    assert abs(float(w[2:].sum()) - 0.5) < 1e-12
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_scaler_source_train_only() -> None:
    x_train = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float64)
    x_test = np.array([[10.0, 10.0]], dtype=np.float64)
    w = np.array([0.5, 0.5])
    sc = SourceStandardScaler()
    sc.fit(x_train, w, ["s1", "s1"], dataset="EUROC")
    z = sc.transform(x_test)
    assert sc.fit_recording_ids == ["s1"]
    assert sc.fit_dataset == "EUROC"
    assert not np.allclose(z, (x_test - x_test.mean()) / (x_test.std() + 1e-12))


def test_metric_formulas() -> None:
    pred = np.array([[1.0, 1.0], [3.0, 5.0]])
    act = np.array([[1.0, 3.0], [3.0, 5.0]])
    assert abs(rmse_all(pred, act) - np.sqrt(1.0)) < 1e-12
    assert abs(mae_all(pred, act) - 0.5) < 1e-12
    assert abs(rmse_at_horizon(pred, act, 2) - np.sqrt(2.0)) < 1e-12
    assert abs(skill_score(1.0, 2.0) - 0.5) < 1e-12


def test_equal_sequence_mean_not_window_pool() -> None:
    from stage2.metrics import equal_sequence_mean

    assert abs(equal_sequence_mean([1.0, 3.0]) - 2.0) < 1e-12


def test_bootstrap_reproducible() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = bootstrap_mean_ci(v, n_reps=200, seed=20260912)
    b = bootstrap_mean_ci(v, n_reps=200, seed=20260912)
    assert a == b
    d = np.array([-0.2, -0.1, 0.05, -0.3, 0.0, -0.05])
    p = paired_sign_flip_p(d)
    assert 0.0 <= p <= 1.0
    w, l, t = win_loss_tie(d)
    assert w + l + t == 6


def test_valid_origins_match_locked_window_rule() -> None:
    ts = np.arange(1000, dtype=np.float64) * 0.01
    orig = valid_origins(ts)
    from stage1.windows import count_windows

    c = count_windows(ts)
    assert orig.size == c["valid_origins"]
    assert orig[0] >= 99
