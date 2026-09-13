"""Readers for CSV / IMU text / ground-truth text. Raw archives stay untouched."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

IMU_NAME_HINTS = ("imu.txt", "imu.csv", "imu0", "snappy_imu", "blackbird_slash_imu")
GT_NAME_HINTS = ("groundtruth", "ground_truth", "gt.txt", "groundTruthPoses", "blackbird_slash_state")
LEICA_HINTS = ("leica", "prism", "ms60")


def _norm(name: str) -> str:
    return name.replace("\\", "/").lower()


def zip_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return [info.filename for info in zf.infolist() if not info.is_dir()]


def classify_zip_member(name: str) -> str | None:
    n = _norm(name)
    base = n.split("/")[-1]
    if any(base.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".pgm", ".ppm", ".pnm")):
        return None
    if "imu" in base and base.endswith((".txt", ".csv")):
        return "imu"
    if "groundtruth" in base or base.startswith("gt.") or "ground_truth" in base:
        return "gt"
    if "leica" in n:
        return "leica"
    if base.endswith((".yaml", ".yml")):
        return "yaml"
    if "camchain" in base or "imu.yaml" in base or "kalibr" in n:
        return "calib"
    if base.endswith(".txt") and "image" not in base and "event" not in base:
        return "text"
    return None


def extract_audit_members(zip_path: Path, dest_dir: Path) -> dict[str, Path]:
    """Extract only IMU/GT/calib text from a zip into cache. Original zip is not modified."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            role = classify_zip_member(info.filename)
            if role is None:
                continue
            target_name = Path(info.filename.replace("\\", "/")).name
            target = dest_dir / target_name
            if target.exists() and target.stat().st_size == info.file_size:
                extracted.setdefault(role, target)
                extracted[f"file:{target_name}"] = target
                continue
            with zf.open(info) as src, target.open("wb") as out:
                out.write(src.read())
            extracted.setdefault(role, target)
            extracted[f"file:{target_name}"] = target
    return extracted


def read_text_table(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (column_names, numeric_array, header_comment_lines)."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    comments: list[str] = []
    data_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("%"):
            comments.append(stripped)
            continue
        data_lines.append(stripped)
    header_cols: list[str] = []
    if comments:
        last = comments[-1].lstrip("#% ").replace(",", " ")
        tokens = [t for t in last.split() if t]
        if len(tokens) >= 4:
            header_cols = tokens
    sample = data_lines[0] if data_lines else ""
    delimiter = "," if (sample.count(",") >= 2 and sample.count(",") >= sample.count(" ")) else None
    rows: list[list[str]] = []
    for line in data_lines:
        if delimiter == ",":
            rows.append(next(csv.reader([line])))
        else:
            rows.append(line.split())
    if not rows:
        return header_cols, np.zeros((0, 0), dtype=np.float64), comments
    # Drop a header row if the first row is non-numeric.
    start = 0
    try:
        [float(x) for x in rows[0][: min(3, len(rows[0]))]]
    except ValueError:
        header_cols = rows[0]
        start = 1
    numeric: list[list[float]] = []
    width = max(len(r) for r in rows[start:]) if rows[start:] else 0
    for row in rows[start:]:
        vals: list[float] = []
        ok = True
        for item in row[:width]:
            item = item.strip()
            if item == "" or item.lower() in {"nan", "na"}:
                vals.append(float("nan"))
                continue
            try:
                vals.append(float(item))
            except ValueError:
                ok = False
                break
        if ok:
            if len(vals) < width:
                vals.extend([float("nan")] * (width - len(vals)))
            numeric.append(vals)
    array = np.asarray(numeric, dtype=np.float64) if numeric else np.zeros((0, width), dtype=np.float64)
    if not header_cols:
        header_cols = [f"col{i}" for i in range(array.shape[1])]
    elif array.shape[1] == len(header_cols) + 1:
        header_cols = ["index"] + header_cols
    elif len(header_cols) < array.shape[1]:
        header_cols = header_cols + [f"col{i}" for i in range(len(header_cols), array.shape[1])]
    return header_cols[: array.shape[1]], array, comments


def _to_seconds(ts: np.ndarray) -> tuple[np.ndarray, str]:
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0:
        return ts, "UNRESOLVED"
    if ts.size == 1:
        mag = abs(float(ts[0]))
        if mag > 1e16:
            return ts / 1e9, "nanoseconds"
        return ts, "seconds_or_single_sample"
    dt = float(np.nanmedian(np.abs(np.diff(ts))))
    if not np.isfinite(dt) or dt <= 0:
        return ts, "UNRESOLVED"
    if dt > 1e5:
        return ts / 1e9, "nanoseconds"
    if dt > 50:
        return ts / 1e6, "microseconds"
    return ts, "seconds"


def interpret_imu_table(columns: list[str], array: np.ndarray, comments: list[str]) -> dict[str, Any]:
    cols_l = [c.lower() for c in columns]
    mapping = {
        "timestamp": None,
        "gx": None,
        "gy": None,
        "gz": None,
        "ax": None,
        "ay": None,
        "az": None,
    }
    joined_comments = "\n".join(comments)

    def find(*needles: str) -> int | None:
        for i, name in enumerate(cols_l):
            if all(n in name for n in needles):
                return i
        return None

    mapping["timestamp"] = find("time") or find("stamp") or 0
    mapping["gx"] = find("ang_vel", "x") or find("gyro", "x") or find("omega", "x")
    mapping["gy"] = find("ang_vel", "y") or find("gyro", "y") or find("omega", "y")
    mapping["gz"] = find("ang_vel", "z") or find("gyro", "z") or find("omega", "z")
    mapping["ax"] = find("lin_acc", "x") or find("linacc", "x") or find("lin_acc") or find("acc", "x")
    mapping["ay"] = find("lin_acc", "y") or find("linacc", "y") or find("acc", "y")
    mapping["az"] = find("lin_acc", "z") or find("linacc", "z") or find("acc", "z")

    # EuRoC / common 7-column layout: t, wx, wy, wz, ax, ay, az
    if array.shape[1] >= 7 and all(mapping[k] is None for k in ("gx", "gy", "gz", "ax", "ay", "az")):
        mapping.update({"timestamp": 0, "gx": 1, "gy": 2, "gz": 3, "ax": 4, "ay": 5, "az": 6})
        layout = "ASSUMED_7COL_T_WXWYWZ_AXAYAZ_UNCONFIRMED_BY_HEADER"
    elif all(v is not None for v in mapping.values()):
        layout = "HEADER_MAPPED"
    elif array.shape[1] >= 7:
        mapping.update({"timestamp": mapping["timestamp"] or 0, "gx": 1, "gy": 2, "gz": 3, "ax": 4, "ay": 5, "az": 6})
        layout = "FALLBACK_7COL"
    else:
        layout = "UNRESOLVED"

    def col(key: str) -> np.ndarray | None:
        idx = mapping.get(key)
        if idx is None or array.size == 0 or idx >= array.shape[1]:
            return None
        return array[:, int(idx)]

    ts = col("timestamp")
    if ts is not None:
        ts_s, timestamp_unit = _to_seconds(ts)
    else:
        ts_s = np.array([])
        timestamp_unit = "UNRESOLVED"

    unit_from_header = "UNRESOLVED"
    if "m s^-2" in joined_comments or "m/s^2" in joined_comments or "[m s^-2]" in joined_comments:
        unit_from_header = "m/s^2"
    gyro_unit_header = "UNRESOLVED"
    if "rad s^-1" in joined_comments or "rad/s" in joined_comments:
        gyro_unit_header = "rad/s"

    return {
        "columns": columns,
        "comments": comments,
        "layout": layout,
        "mapping": mapping,
        "timestamps_s": ts_s,
        "timestamp_unit_native": timestamp_unit,
        "gyro": None if col("gx") is None else np.column_stack([col("gx"), col("gy"), col("gz")]),
        "accel": None if col("ax") is None else np.column_stack([col("ax"), col("ay"), col("az")]),
        "unit_acceleration_from_header": unit_from_header,
        "unit_gyro_from_header": gyro_unit_header,
        "accel_fields": {
            "x": columns[mapping["ax"]] if mapping["ax"] is not None and mapping["ax"] < len(columns) else "UNRESOLVED",
            "y": columns[mapping["ay"]] if mapping["ay"] is not None and mapping["ay"] < len(columns) else "UNRESOLVED",
            "z": columns[mapping["az"]] if mapping["az"] is not None and mapping["az"] < len(columns) else "UNRESOLVED",
        },
        "gyro_fields": {
            "x": columns[mapping["gx"]] if mapping["gx"] is not None and mapping["gx"] < len(columns) else "UNRESOLVED",
            "y": columns[mapping["gy"]] if mapping["gy"] is not None and mapping["gy"] < len(columns) else "UNRESOLVED",
            "z": columns[mapping["gz"]] if mapping["gz"] is not None and mapping["gz"] < len(columns) else "UNRESOLVED",
        },
        "timestamp_field": columns[mapping["timestamp"]] if mapping["timestamp"] is not None and mapping["timestamp"] < len(columns) else "UNRESOLVED",
        "n": int(array.shape[0]),
    }


def interpret_gt_table(columns: list[str], array: np.ndarray, comments: list[str]) -> dict[str, Any]:
    cols_l = [c.lower() for c in columns]
    joined = "\n".join(comments)

    def find(*needles: str) -> int | None:
        for i, name in enumerate(cols_l):
            if all(n in name for n in needles):
                return i
        return None

    ts_idx = find("time") or find("t") or 0
    # Common 8-col: t x y z qx qy qz qw   or t x y z qw qx qy qz
    if array.shape[1] >= 8:
        pos_idx = [1, 2, 3]
        quat_idx = [4, 5, 6, 7]
        quat_order = "UNRESOLVED_4TUPLE"
        if find("qx") is not None:
            quat_idx = [find("qx"), find("qy"), find("qz"), find("qw") or find("q", "w")]
            quat_order = "xyzw_from_header" if (find("qw") or 0) > (find("qx") or 0) else "UNRESOLVED"
            if find("qw") is not None and find("qw") < find("qx"):
                quat_order = "wxyz_from_header"
        elif "qx qy qz qw" in joined.lower() or "qx,qy,qz,qw" in joined.lower().replace(" ", ""):
            quat_order = "xyzw_from_comment"
        elif "qw qx qy qz" in joined.lower():
            quat_order = "wxyz_from_comment"
            quat_idx = [4, 5, 6, 7]
    else:
        pos_idx = [i for i in (find("x"), find("y"), find("z")) if i is not None]
        quat_idx = [i for i in (find("qx"), find("qy"), find("qz"), find("qw")) if i is not None]
        quat_order = "HEADER_MAPPED" if len(quat_idx) == 4 else "UNRESOLVED"

    ts = array[:, ts_idx] if array.size else np.array([])
    ts_s, unit = _to_seconds(ts)

    vel_fields = "NOT_PRESENT"
    acc_fields = "NOT_PRESENT"
    if any("vel" in c or c in {"vx", "vy", "vz"} for c in cols_l):
        vel_fields = ",".join(c for c in columns if "vel" in c.lower() or c.lower() in {"vx", "vy", "vz"})
    if any("acc" in c for c in cols_l):
        acc_fields = ",".join(c for c in columns if "acc" in c.lower())

    return {
        "columns": columns,
        "comments": comments,
        "timestamps_s": ts_s,
        "timestamp_unit_native": unit,
        "timestamp_field": columns[ts_idx] if ts_idx < len(columns) else "UNRESOLVED",
        "position": array[:, pos_idx] if array.size and pos_idx else None,
        "position_fields": ",".join(columns[i] if i < len(columns) else str(i) for i in pos_idx),
        "orientation": array[:, quat_idx] if array.size and len(quat_idx) == 4 and max(quat_idx) < array.shape[1] else None,
        "orientation_fields": ",".join(str(columns[i] if i < len(columns) else i) for i in quat_idx),
        "orientation_convention": quat_order,
        "velocity_fields": vel_fields,
        "acceleration_fields": acc_fields,
        "gt_type": "GROUND_TRUTH_POSE_ONLY" if acc_fields == "NOT_PRESENT" else "UNRESOLVED_MAY_INCLUDE_MORE",
        "n": int(array.shape[0]),
    }


def read_csv_flexible(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
