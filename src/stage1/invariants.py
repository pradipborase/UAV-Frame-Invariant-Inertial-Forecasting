"""Frame-invariant specific-force and angular-speed magnitudes. No modelling."""

from __future__ import annotations

import numpy as np


def specific_force_magnitude(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Accelerometer specific-force magnitude q_f = sqrt(fx^2+fy^2+fz^2) in m/s^2."""
    ax = np.asarray(fx, dtype=np.float64).reshape(-1)
    ay = np.asarray(fy, dtype=np.float64).reshape(-1)
    az = np.asarray(fz, dtype=np.float64).reshape(-1)
    if not (ax.shape == ay.shape == az.shape):
        raise ValueError("specific-force components must share shape")
    return np.sqrt(ax * ax + ay * ay + az * az)


def angular_speed_magnitude(wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    """Angular-speed magnitude q_w = sqrt(wx^2+wy^2+wz^2) in rad/s."""
    gx = np.asarray(wx, dtype=np.float64).reshape(-1)
    gy = np.asarray(wy, dtype=np.float64).reshape(-1)
    gz = np.asarray(wz, dtype=np.float64).reshape(-1)
    if not (gx.shape == gy.shape == gz.shape):
        raise ValueError("angular-velocity components must share shape")
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def random_rotation_det_plus_one(rng: np.random.Generator) -> np.ndarray:
    """Haar-like proper rotation via QR; enforce det(R)=+1."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q
