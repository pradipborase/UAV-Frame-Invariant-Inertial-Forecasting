"""Load frozen Stage-2/3 CSV tables. No model fitting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

CONTEXTS = ("WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC")
PUB_MODELS = ("Persistence", "Best frozen Ridge", "TCN", "GRU", "Transformer")
ADV_MODELS = ("TCN", "GRU", "Transformer")
BOOT_REPS = 10000
BOOT_SEED = 20260912
TOL_CSV = 1e-6
TOL_REPORT = 5e-4

STAGE3_SEQ_FILES = {
    "WITHIN_EUROC": "WITHIN_EUROC_ADVANCED.csv",
    "WITHIN_UZH": "WITHIN_UZH_ADVANCED.csv",
    "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_ADVANCED.csv",
    "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_ADVANCED.csv",
}
STAGE2_SEQ_FILES = {
    "WITHIN_EUROC": "WITHIN_EUROC_SEQUENCE_RESULTS.csv",
    "WITHIN_UZH": "WITHIN_UZH_SEQUENCE_RESULTS.csv",
    "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv",
    "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv",
}
BEST_RIDGE_ID = {
    "WITHIN_EUROC": "B3_RIDGE_QF_QW",
    "WITHIN_UZH": "B3_RIDGE_QF_QW",
    "EUROC_TO_UZH": "B3_RIDGE_QF_QW",
    "UZH_TO_EUROC": "B2_RIDGE_QF",
}
MODEL_FROM_FILE = {
    "B0_PERSISTENCE": "Persistence",
    "BEST_RIDGE": "Best frozen Ridge",
    "TCN": "TCN",
    "GRU": "GRU",
    "TRANSFORMER": "Transformer",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def load_stage3_sequences(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ctx, name in STAGE3_SEQ_FILES.items():
        for raw in read_csv(root / "results" / "stage3" / name):
            model = MODEL_FROM_FILE.get(raw["model"])
            if model is None:
                continue
            rec = dict(raw)
            rec["pub_model"] = model
            rec["evaluation_context"] = ctx
            rows.append(rec)
    return rows


def load_stage2_sequences(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ctx, name in STAGE2_SEQ_FILES.items():
        for raw in read_csv(root / "results" / "stage2" / name):
            rec = dict(raw)
            rec["evaluation_context"] = ctx
            rows.append(rec)
    return rows


def sequence_values(rows: list[dict[str, Any]], context: str, pub_model: str, field: str) -> np.ndarray:
    vals = [
        fnum(r[field])
        for r in rows
        if r["evaluation_context"] == context and r.get("pub_model") == pub_model
    ]
    return np.asarray(vals, dtype=np.float64)


def sequence_ids(rows: list[dict[str, Any]], context: str, pub_model: str) -> list[str]:
    ids: list[str] = []
    for r in rows:
        if r["evaluation_context"] == context and r.get("pub_model") == pub_model:
            sid = str(r["sequence_id"])
            if sid not in ids:
                ids.append(sid)
    return ids
