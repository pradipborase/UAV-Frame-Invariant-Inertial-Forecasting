"""EuRoC ASL-format CSV/YAML parsers. Audit-only: no resampling or frame rotation."""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .timestamp_audit import audit_timestamps, overlap_seconds

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pgm", ".ppm", ".pnm", ".tif", ".tiff")
SKIP_EXTRACT_SUFFIXES = IMAGE_SUFFIXES + (".bag", ".pcd", ".ply")


def split_euroc_header(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("#"):
        text = text[1:]
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
    else:
        parts = [p for p in text.split() if p]
    return [p for p in parts if p]


def _contains_token(header: str, token: str) -> bool:
    compact = header.lower().replace(" ", "")
    return token.lower().replace(" ", "") in compact


def map_euroc_imu_header(headers: list[str]) -> dict[str, Any]:
    """Map exact EuRoC/ASL IMU header strings. Never match accel via a bare letter 'a'."""

    def find(*tokens: str) -> int | None:
        for i, name in enumerate(headers):
            if all(_contains_token(name, t) for t in tokens):
                return i
        return None

    timestamp_i = find("timestamp") or find("t") or 0
    gx = find("w_rs_s_x") or find("omega_x") or find("gyro_x") or find("w_x [")
    gy = find("w_rs_s_y") or find("omega_y") or find("gyro_y") or find("w_y [")
    gz = find("w_rs_s_z") or find("omega_z") or find("gyro_z") or find("w_z [")
    ax = find("a_rs_s_x") or find("acc_x") or find("lin_acc_x")
    ay = find("a_rs_s_y") or find("acc_y") or find("lin_acc_y")
    az = find("a_rs_s_z") or find("acc_z") or find("lin_acc_z")

    layout = "HEADER_MAPPED"
    if None in (gx, gy, gz, ax, ay, az):
        if len(headers) >= 7:
            gx, gy, gz, ax, ay, az = 1, 2, 3, 4, 5, 6
            layout = "OFFICIAL_MATLAB_7COL_T_WXWYWZ_AXAYAZ"
        else:
            layout = "UNRESOLVED"

    def name(idx: int | None) -> str:
        if idx is None or idx < 0 or idx >= len(headers):
            return "UNRESOLVED"
        return headers[idx]

    accel_unit = "UNRESOLVED"
    gyro_unit = "UNRESOLVED"
    ts_unit = "UNRESOLVED"
    joined = " ".join(headers)
    if "m s^-2" in joined or "m/s^2" in joined or "[m s^-2]" in joined:
        accel_unit = "m/s^2"
    if "rad s^-1" in joined or "rad/s" in joined or "[rad s^-1]" in joined:
        gyro_unit = "rad/s"
    if "[ns]" in joined or "timestamp [ns]" in joined.lower():
        ts_unit = "nanoseconds"

    return {
        "timestamp_index": timestamp_i,
        "gyro_indices": (gx, gy, gz),
        "accel_indices": (ax, ay, az),
        "timestamp_field": name(timestamp_i),
        "gyro_x_raw_field": name(gx),
        "gyro_y_raw_field": name(gy),
        "gyro_z_raw_field": name(gz),
        "accel_x_raw_field": name(ax),
        "accel_y_raw_field": name(ay),
        "accel_z_raw_field": name(az),
        "timestamp_unit_from_header": ts_unit,
        "acceleration_unit_from_header": accel_unit,
        "gyro_unit_from_header": gyro_unit,
        "layout": layout,
        "headers": headers,
    }


def parse_opencv_matrix(node: Any) -> np.ndarray | None:
    if not isinstance(node, dict) or "data" not in node:
        if isinstance(node, list) and len(node) in {9, 16}:
            n = int(math.sqrt(len(node)))
            return np.asarray(node, dtype=np.float64).reshape(n, n)
        return None
    rows = int(node.get("rows", 0))
    cols = int(node.get("cols", 0))
    data = node.get("data", [])
    values = [float(x) for x in data]
    if rows <= 0 or cols <= 0 or len(values) != rows * cols:
        return None
    # ASL YAML data is row-major; official MATLAB reshape(data,rows,cols)' recovers the same.
    return np.asarray(values, dtype=np.float64).reshape(rows, cols)


def load_sensor_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"sensor YAML root is not a mapping: {path}")
    t_bs = parse_opencv_matrix(raw.get("T_BS"))
    rotation = None
    translation = None
    if t_bs is not None and t_bs.shape[0] >= 3 and t_bs.shape[1] >= 4:
        rotation = t_bs[:3, :3]
        translation = t_bs[:3, 3]
    return {
        "raw": raw,
        "sensor_type": raw.get("sensor_type", "UNRESOLVED"),
        "comment": raw.get("comment", ""),
        "rate_hz": raw.get("rate_hz", raw.get("frequency", "UNRESOLVED")),
        "T_BS": t_bs,
        "rotation": rotation,
        "translation": translation,
        "gyroscope_noise_density": raw.get("gyroscope_noise_density", "UNRESOLVED"),
        "gyroscope_random_walk": raw.get("gyroscope_random_walk", "UNRESOLVED"),
        "accelerometer_noise_density": raw.get("accelerometer_noise_density", "UNRESOLVED"),
        "accelerometer_random_walk": raw.get("accelerometer_random_walk", "UNRESOLVED"),
        "source_path": str(path),
    }


def read_euroc_csv(path: Path) -> tuple[list[str], np.ndarray, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], np.zeros((0, 0), dtype=np.float64), ""
    header_line = lines[0]
    headers = split_euroc_header(header_line)
    start = 1
    if not header_line.lstrip().startswith("#"):
        try:
            [float(x) for x in split_euroc_header(header_line)[: min(3, len(headers))]]
            headers = [f"col{i}" for i in range(len(headers))]
            start = 0
        except ValueError:
            start = 1
    rows: list[list[object]] = []
    for line in lines[start:]:
        if line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        vals: list[object] = []
        ok = True
        for item in parts:
            if item == "" or item.lower() in {"nan", "na"}:
                vals.append(float("nan"))
                continue
            try:
                if "." not in item and "e" not in item.lower():
                    vals.append(int(item))
                else:
                    vals.append(float(item))
            except ValueError:
                ok = False
                break
        if ok and vals:
            rows.append(vals)
    if not rows:
        return headers, np.zeros((0, len(headers)), dtype=np.float64), header_line
    width = max(len(r) for r in rows)
    padded = [r + [float("nan")] * (width - len(r)) for r in rows]
    if len(headers) != width:
        if len(headers) < width:
            headers = headers + [f"col{i}" for i in range(len(headers), width)]
        else:
            headers = headers[:width]
    return headers, np.asarray(padded, dtype=object), header_line


def as_uint64(ts: np.ndarray) -> np.ndarray:
    col = np.asarray(ts).reshape(-1)
    return np.asarray([int(x) for x in col], dtype=np.uint64)


def relative_seconds_from_ns(ts: np.ndarray, origin_ns: int | None = None) -> tuple[np.ndarray, int | None]:
    if ts is None or np.asarray(ts).size == 0:
        return np.asarray([], dtype=np.float64), origin_ns
    ts_u = as_uint64(ts)
    origin = int(ts_u[0] if origin_ns is None else origin_ns)
    delta = ts_u.astype(np.int64) - np.int64(origin)
    return delta.astype(np.float64) / 1e9, origin


def timestamps_to_seconds(ts: np.ndarray, declared_unit: str) -> tuple[np.ndarray, str]:
    """Convert timestamps to seconds without destroying nanosecond spacing.

    Absolute Unix-like nanosecond counters exceed float64 mantissa. Differences
    are computed in integer nanoseconds relative to the first sample, then scaled.
    Returned seconds therefore start at 0.0 for a nanosecond series.
    """
    col = np.asarray(ts).reshape(-1)
    if col.size == 0:
        return np.asarray([], dtype=np.float64), declared_unit or "UNRESOLVED"
    if declared_unit == "nanoseconds":
        rel, _ = relative_seconds_from_ns(col)
        return rel, "nanoseconds"
    try:
        as_int = as_uint64(col)
        if col.size >= 2:
            med = int(np.median(np.abs(np.diff(as_int.astype(np.int64)))))
            if med > 1000:
                rel, _ = relative_seconds_from_ns(as_int)
                return rel, "nanoseconds"
    except (ValueError, OverflowError, TypeError):
        pass
    ts_f = np.asarray(col, dtype=np.float64)
    if ts_f.size < 2:
        return ts_f, declared_unit or "UNRESOLVED"
    dt = float(np.nanmedian(np.abs(np.diff(ts_f))))
    if dt > 1e5:
        rel, _ = relative_seconds_from_ns(col)
        return rel, "nanoseconds"
    if dt > 50:
        return ts_f / 1e6, "microseconds"
    return ts_f, "seconds"


def extra_dt_frequency_stats(ts_seconds: np.ndarray) -> dict[str, Any]:
    ts = np.asarray(ts_seconds, dtype=np.float64).reshape(-1)
    ts = ts[np.isfinite(ts)]
    empty = {
        "min_dt": None,
        "max_dt": None,
        "measured_p05_rate_hz": None,
        "measured_p95_rate_hz": None,
        "measured_median_rate_hz": None,
    }
    if ts.size < 2:
        return empty
    diffs = np.diff(ts)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return empty
    rates = 1.0 / positive
    return {
        "min_dt": float(np.min(positive)),
        "max_dt": float(np.max(positive)),
        "measured_p05_rate_hz": float(np.percentile(rates, 5)),
        "measured_p95_rate_hz": float(np.percentile(rates, 95)),
        "measured_median_rate_hz": float(np.median(rates)),
    }


def imu_channel_integrity(accel: np.ndarray, gyro: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accel_nan": 0,
        "accel_inf": 0,
        "gyro_nan": 0,
        "gyro_inf": 0,
        "accel_constant": False,
        "gyro_constant": False,
        "finite_fraction_required": 1.0,
    }
    if accel is None or accel.size == 0:
        return out
    out["accel_nan"] = int(np.sum(~np.isfinite(accel) & np.isnan(accel)))
    out["accel_inf"] = int(np.sum(np.isinf(accel)))
    out["gyro_nan"] = int(np.sum(~np.isfinite(gyro) & np.isnan(gyro)))
    out["gyro_inf"] = int(np.sum(np.isinf(gyro)))
    finite_a = np.isfinite(accel)
    finite_g = np.isfinite(gyro)
    req = int(accel.shape[0] * accel.shape[1] + gyro.shape[0] * gyro.shape[1])
    finite = int(np.sum(finite_a) + np.sum(finite_g))
    out["finite_fraction_required"] = (finite / req) if req else 0.0
    if accel.shape[0] > 1:
        out["accel_constant"] = bool(np.all(np.nanstd(accel, axis=0) == 0))
        out["gyro_constant"] = bool(np.all(np.nanstd(gyro, axis=0) == 0))
    return out


def rest_norm_median(accel: np.ndarray, n_head: int = 400) -> float | None:
    if accel is None or accel.size == 0:
        return None
    n = min(int(n_head), int(accel.shape[0]))
    if n <= 0:
        return None
    norms = np.linalg.norm(accel[:n], axis=1)
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return None
    return float(np.median(norms))


def should_extract_member(name: str) -> bool:
    n = name.replace("\\", "/").lower()
    if n.endswith(SKIP_EXTRACT_SUFFIXES):
        return False
    base = n.split("/")[-1]
    if base.startswith("._"):
        return False
    return n.endswith((".csv", ".yaml", ".yml"))


def extract_text_members(zip_path: Path, dest_dir: Path) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not should_extract_member(info.filename):
                continue
            rel = Path(info.filename.replace("\\", "/"))
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                extracted[str(rel).replace("\\", "/")] = target
                continue
            with zf.open(info) as src, target.open("wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            extracted[str(rel).replace("\\", "/")] = target
    return extracted


def extract_inner_zips(outer_zip: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(outer_zip) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not name.lower().endswith(".zip"):
                continue
            if Path(name).name.startswith("._"):
                continue
            target = dest_dir / Path(name).name
            if target.exists() and target.stat().st_size == info.file_size:
                out.append(target)
                continue
            with zf.open(info) as src, target.open("wb") as handle:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            out.append(target)
    return out


def list_zip_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return [info.filename.replace("\\", "/") for info in zf.infolist() if not info.is_dir()]


def find_named(extracted: dict[str, Path], *needles: str) -> Path | None:
    for key, path in extracted.items():
        k = key.replace("\\", "/").lower()
        if all(n.lower() in k for n in needles):
            return path
    return None


def rotation_to_quaternion_wxyz(R: np.ndarray) -> list[float]:
    """wxyz from a rotation matrix. Reporting only — streams are not rotated."""
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def map_euroc_gt_header(headers: list[str]) -> dict[str, Any]:
    def find(*tokens: str) -> int | None:
        for i, name in enumerate(headers):
            if all(_contains_token(name, t) for t in tokens):
                return i
        return None

    def names(*idxs: int | None) -> str:
        parts = []
        for idx in idxs:
            if idx is None or idx >= len(headers):
                parts.append("UNRESOLVED")
            else:
                parts.append(headers[idx])
        return ",".join(parts)

    ts = find("timestamp") or 0
    px, py, pz = find("p_rs_r_x") or find("p_x"), find("p_rs_r_y") or find("p_y"), find("p_rs_r_z") or find("p_z")
    qw = find("q_rs_w") or find("q_w")
    qx = find("q_rs_x") or find("q_x")
    qy = find("q_rs_y") or find("q_y")
    qz = find("q_rs_z") or find("q_z")
    vx, vy, vz = find("v_rs_r_x") or find("v_x"), find("v_rs_r_y") or find("v_y"), find("v_rs_r_z") or find("v_z")
    bwx = find("b_w_rs_s_x") or find("bw_x")
    bax = find("b_a_rs_s_x") or find("ba_x")
    # Direct specific-force / linear-acceleration GT columns (not accelerometer bias).
    acc_gt = [
        h
        for h in headers
        if _contains_token(h, "a_rs") and "b_a" not in h.lower().replace(" ", "")
    ]
    quat_order = "UNRESOLVED"
    if None not in (qw, qx, qy, qz):
        order = sorted([(qw, "w"), (qx, "x"), (qy, "y"), (qz, "z")], key=lambda t: t[0])
        quat_order = "".join(p[1] for p in order)

    return {
        "timestamp_field": headers[ts] if ts < len(headers) else "UNRESOLVED",
        "position_fields": names(px, py, pz),
        "orientation_fields": names(qw, qx, qy, qz),
        "quaternion_order_if_applicable": quat_order,
        "velocity_fields": names(vx, vy, vz) if vx is not None else "NOT_PRESENT",
        "angular_velocity_fields": "NOT_PRESENT",
        "acceleration_fields": ",".join(acc_gt) if acc_gt else "NO_DIRECT_GT_ACCELERATION",
        "bias_gyro_present": bwx is not None,
        "bias_accel_present": bax is not None,
        "headers": headers,
        "timestamp_index": ts,
    }


def compute_gt_overlap(imu_ts_s: np.ndarray, gt_ts_s: np.ndarray) -> dict[str, Any]:
    if imu_ts_s.size == 0 or gt_ts_s.size == 0:
        return {
            "imu_gt_start_overlap": None,
            "imu_gt_end_overlap": None,
            "overlap_duration_s": 0.0,
            "overlap_fraction": 0.0,
        }
    overlap, frac = overlap_seconds(
        float(imu_ts_s[0]),
        float(imu_ts_s[-1]),
        float(gt_ts_s[0]),
        float(gt_ts_s[-1]),
    )
    start = max(float(imu_ts_s[0]), float(gt_ts_s[0]))
    end = min(float(imu_ts_s[-1]), float(gt_ts_s[-1]))
    return {
        "imu_gt_start_overlap": start if overlap and overlap > 0 else None,
        "imu_gt_end_overlap": end if overlap and overlap > 0 else None,
        "overlap_duration_s": float(overlap or 0.0),
        "overlap_fraction": float(frac or 0.0),
    }


def audit_native_timestamps(ts_native: np.ndarray, native_unit: str) -> dict[str, Any]:
    stats = audit_timestamps(ts_native, native_unit)
    return stats


def BytesIO_text(text: str) -> io.StringIO:
    return io.StringIO(text)
