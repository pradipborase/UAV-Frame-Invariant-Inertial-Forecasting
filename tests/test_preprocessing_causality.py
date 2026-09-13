"""Causal preprocessing: future samples must not affect past outputs."""

from __future__ import annotations

import numpy as np

from stage1.causal_filter import apply_causal_sos, design_antialias_sos, integer_decimate
from stage1.pipeline import process_invariant_stream


def test_sosfilt_no_future_leakage() -> None:
    sos = design_antialias_sos(200.0)
    n = 800
    cut = 300
    rng = np.random.default_rng(7)
    a = rng.normal(size=n)
    b = a.copy()
    b[cut + 1 :] += 8.0
    ya = apply_causal_sos(sos, a)
    yb = apply_causal_sos(sos, b)
    np.testing.assert_allclose(ya[: cut + 1], yb[: cut + 1], rtol=1e-12, atol=1e-12)
    assert float(np.max(np.abs(ya[cut + 1 :] - yb[cut + 1 :]))) > 1e-3


def test_pipeline_causality_with_decimation() -> None:
    fs = 200.0
    n = 1000
    ts = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(11)
    qf = rng.normal(size=n) + 9.8
    qw = rng.normal(size=n) * 0.1
    cut = 400
    t_cut = ts[cut]
    qf_b = qf.copy()
    qf_b[cut + 1 :] += 5.0
    sos = design_antialias_sos(fs)
    a = process_invariant_stream(ts, qf, qw, native_rate_hz=fs, decimation_factor=2, output_rate_hz=100.0, warmup_s=0.0, sos=sos)
    b = process_invariant_stream(ts, qf_b, qw, native_rate_hz=fs, decimation_factor=2, output_rate_hz=100.0, warmup_s=0.0, sos=sos)
    mask = a["timestamp_s"] <= t_cut + 1e-12
    np.testing.assert_allclose(a["q_f"][mask], b["q_f"][mask], rtol=1e-12, atol=1e-12)


def test_integer_decimation_preserves_timestamps() -> None:
    x = np.arange(10, dtype=np.float64)
    ts = np.linspace(0.0, 0.045, 10)
    xd, tsd, idx = integer_decimate(x, ts, 2)
    np.testing.assert_array_equal(idx, np.array([0, 2, 4, 6, 8]))
    np.testing.assert_allclose(tsd, ts[::2])
    np.testing.assert_allclose(xd, x[::2])
