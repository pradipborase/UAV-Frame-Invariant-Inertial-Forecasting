"""Rotation invariance of Euclidean magnitudes. Synthetic vectors only."""

from __future__ import annotations

import numpy as np

from stage1.invariants import (
    angular_speed_magnitude,
    random_rotation_det_plus_one,
    specific_force_magnitude,
)


def test_qf_formula_3_4_12() -> None:
    q = specific_force_magnitude([3.0], [4.0], [12.0])
    assert abs(float(q[0]) - 13.0) < 1e-15


def test_qw_formula() -> None:
    q = angular_speed_magnitude([0.0], [3.0], [4.0])
    assert abs(float(q[0]) - 5.0) < 1e-15


def test_rotation_invariance_proper_rotations() -> None:
    rng = np.random.default_rng(20260912)
    f = rng.normal(size=(3, 64))
    w = rng.normal(size=(3, 64))
    qf = specific_force_magnitude(f[0], f[1], f[2])
    qw = angular_speed_magnitude(w[0], w[1], w[2])
    for _ in range(20):
        r = random_rotation_det_plus_one(rng)
        assert abs(float(np.linalg.det(r)) - 1.0) < 1e-10
        rf = r @ f
        rw = r @ w
        qf_r = specific_force_magnitude(rf[0], rf[1], rf[2])
        qw_r = angular_speed_magnitude(rw[0], rw[1], rw[2])
        np.testing.assert_allclose(qf_r, qf, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(qw_r, qw, rtol=1e-12, atol=1e-12)
