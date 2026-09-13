"""Load Stage-1 canonical processed streams. Do not modify raw archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stage1.folds import EUROC_ROOM, UZH_GROUP
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY


def processed_path(root: Path, dataset: str, sequence_id: str) -> Path:
    return root / "data" / "processed" / "stage1" / f"{dataset}_{sequence_id}.csv"


def load_processed_csv(path: Path) -> dict[str, Any]:
    dataset = ""
    sequence_id = ""
    rows_ts: list[float] = []
    rows_qf: list[float] = []
    rows_qw: list[float] = []
    rows_idx: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped.startswith("# dataset="):
                    dataset = stripped.split("=", 1)[1]
                elif stripped.startswith("# sequence_id="):
                    sequence_id = stripped.split("=", 1)[1]
                continue
            if stripped.startswith("timestamp_s"):
                continue
            parts = stripped.split(",")
            rows_ts.append(float(parts[0]))
            rows_qf.append(float(parts[1]))
            rows_qw.append(float(parts[2]))
            rows_idx.append(int(float(parts[3]))) if len(parts) > 3 else rows_idx.append(-1)
    ts = np.asarray(rows_ts, dtype=np.float64)
    qf = np.asarray(rows_qf, dtype=np.float64)
    qw = np.asarray(rows_qw, dtype=np.float64)
    native_index = np.asarray(rows_idx, dtype=np.int64)
    return {
        "dataset": dataset,
        "sequence_id": sequence_id,
        "timestamp_s": ts,
        "q_f": qf,
        "q_w": qw,
        "native_index": native_index,
        "path": str(path),
    }


def load_all_processed(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for seq in EUROC_PRIMARY:
        rec = load_processed_csv(processed_path(root, "EUROC", seq))
        rec["dependency_group"] = EUROC_ROOM[seq]
        out[("EUROC", seq)] = rec
    for seq in UZH_PRIMARY:
        rec = load_processed_csv(processed_path(root, "UZH_FPV", seq))
        rec["dependency_group"] = UZH_GROUP[seq]
        out[("UZH_FPV", seq)] = rec
    return out
