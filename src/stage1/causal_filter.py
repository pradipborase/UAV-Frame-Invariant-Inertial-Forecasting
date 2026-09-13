"""Deterministic causal anti-alias IIR. No zero-phase filtering. No data-driven fitting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.signal import group_delay, iirdesign, sos2tf, sos2zpk, sosfilt, sosfreqz

# Locked physical edges. Do not retune from prediction error.
PASSBAND_HZ = 30.0
STOPBAND_HZ = 45.0
GPASS_DB = 1.0
GSTOP_DB = 60.0
FTYPE = "ellip"


def design_antialias_sos(fs_hz: float) -> np.ndarray:
    if fs_hz <= 2.0 * STOPBAND_HZ:
        raise ValueError(f"Nyquist must exceed stopband; fs={fs_hz}")
    sos = iirdesign(
        wp=PASSBAND_HZ,
        ws=STOPBAND_HZ,
        gpass=GPASS_DB,
        gstop=GSTOP_DB,
        analog=False,
        ftype=FTYPE,
        output="sos",
        fs=float(fs_hz),
    )
    return np.asarray(sos, dtype=np.float64)


def iir_order_from_sos(sos: np.ndarray, atol: float = 1e-12) -> int:
    """True IIR denominator order from padded SOS.

    SciPy stores some odd-order elliptic designs as SOS with a first-order
    section whose ``a2`` coefficient is zero. Counting ``2 * n_sections``
    therefore mislabels those designs (the historical Stage-1 metadata bug).
    This function does not change coefficients or filtering.
    """
    arr = np.atleast_2d(np.asarray(sos, dtype=np.float64))
    if arr.size == 0:
        return 0
    order = 0
    for section in arr:
        a1 = float(section[4])
        a2 = float(section[5])
        if abs(a2) > atol:
            order += 2
        elif abs(a1) > atol:
            order += 1
    return int(order)


def ellip_sos_filter_name(sos: np.ndarray, fs_hz: float) -> str:
    return f"ellip_sos_order{iir_order_from_sos(sos)}_fs{int(fs_hz)}"


def apply_causal_sos(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward-only SOS filtering. Never uses future samples."""
    return sosfilt(sos, np.asarray(x, dtype=np.float64))


def integer_decimate(x: np.ndarray, ts: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor < 1:
        raise ValueError("decimation factor must be >= 1")
    x = np.asarray(x)
    ts = np.asarray(ts)
    idx = np.arange(0, x.shape[0], factor, dtype=np.int64)
    return x[idx], ts[idx], idx


def pole_radii(sos: np.ndarray) -> np.ndarray:
    _z, p, _k = sos2zpk(sos)
    return np.abs(np.asarray(p, dtype=np.complex128))


def is_stable(sos: np.ndarray, margin: float = 1e-12) -> bool:
    return bool(np.all(pole_radii(sos) < 1.0 - margin))


def frequency_response(sos: np.ndarray, fs_hz: float, n: int = 8192) -> dict[str, np.ndarray]:
    w, h = sosfreqz(sos, worN=n, fs=fs_hz)
    mag = np.abs(h)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-20))
    phase = np.unwrap(np.angle(h))
    b, a = sos2tf(sos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd_w, gd = group_delay((b, a), w=512, fs=fs_hz)
    gd = np.asarray(gd, dtype=np.float64)
    gd = np.where(np.isfinite(gd), gd, np.nan)
    return {
        "frequency_hz": np.asarray(w, dtype=np.float64),
        "magnitude": np.asarray(mag, dtype=np.float64),
        "magnitude_db": np.asarray(mag_db, dtype=np.float64),
        "phase_rad": np.asarray(phase, dtype=np.float64),
        "gd_frequency_hz": np.asarray(gd_w, dtype=np.float64),
        "group_delay_samples": np.asarray(gd, dtype=np.float64),
    }


def spec_metrics(sos: np.ndarray, fs_hz: float) -> dict[str, Any]:
    resp = frequency_response(sos, fs_hz)
    f = resp["frequency_hz"]
    mag_db = resp["magnitude_db"]
    pb = mag_db[f <= PASSBAND_HZ]
    sb = mag_db[f >= STOPBAND_HZ]
    pb_min = float(np.min(pb)) if pb.size else float("nan")
    pb_max = float(np.max(pb)) if pb.size else float("nan")
    sb_max = float(np.max(sb)) if sb.size else float("nan")
    max_passband_loss_db = float(max(0.0, -pb_min))
    min_stopband_atten_db = float(-sb_max)
    gd = resp["group_delay_samples"]
    gd_pb = gd[resp["gd_frequency_hz"] <= PASSBAND_HZ]
    return {
        "fs_hz": float(fs_hz),
        "n_sections": int(sos.shape[0]),
        "order": iir_order_from_sos(sos),
        "max_pole_radius": float(np.max(pole_radii(sos))),
        "stable": is_stable(sos),
        "passband_max_db": pb_max,
        "passband_min_db": pb_min,
        "max_passband_loss_db": max_passband_loss_db,
        "min_stopband_atten_db": min_stopband_atten_db,
        "meets_passband": max_passband_loss_db <= GPASS_DB + 1e-6,
        "meets_stopband": min_stopband_atten_db >= GSTOP_DB - 1e-6,
        "group_delay_passband_median_samples": float(np.median(gd_pb)) if gd_pb.size else float("nan"),
        "group_delay_passband_max_samples": float(np.max(gd_pb)) if gd_pb.size else float("nan"),
        "sos": sos,
        "response": resp,
    }


def impulse_tail_relative(sos: np.ndarray, fs_hz: float, t_s: float) -> float:
    n = int(round(2.0 * fs_hz))
    x = np.zeros(n, dtype=np.float64)
    x[0] = 1.0
    y = np.abs(apply_causal_sos(sos, x))
    peak = float(np.max(y)) if y.size else 1.0
    i = min(int(round(t_s * fs_hz)), y.size - 1)
    tail = float(np.max(y[i:])) if i < y.size else 0.0
    return tail / peak if peak > 0 else 0.0
