"""Stage-1 artefact checks. Processed IMU files are optional in the public package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
import yaml

from stage1.folds import fold_is_sequence_disjoint

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "results" / "stage1"
NEED = STAGE1 / "PROCESSED_DATA_MANIFEST.csv"
STAGE1_REPORT = STAGE1 / "STAGE1_REPORT.md"


def _rows(name: str) -> list[dict[str, str]]:
    with (STAGE1 / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _processed_path(row: dict[str, str]) -> Path:
    raw = Path(row["local_path"])
    if raw.exists():
        return raw
    rel = ROOT / "data" / "processed" / "stage1" / f"{row['dataset']}_{row['sequence_id']}.csv"
    if rel.exists():
        return rel
    if not raw.is_absolute():
        candidate = ROOT / raw
        if candidate.exists():
            return candidate
    return rel


def _processed_streams_available() -> bool:
    if not NEED.exists():
        return False
    rows = _rows("PROCESSED_DATA_MANIFEST.csv")
    return bool(rows) and all(_processed_path(row).exists() for row in rows)


@pytest.mark.skipif(not _processed_streams_available(), reason="processed Stage-1 streams are intentionally not shipped")
def test_fourteen_sequences_and_hashes() -> None:
    rows = _rows("PROCESSED_DATA_MANIFEST.csv")
    assert len(rows) == 14
    for row in rows:
        path = _processed_path(row)
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"]
        assert row["q_f_unit"] == "m/s^2"
        assert row["q_w_unit"] == "rad/s"


@pytest.mark.skipif(not NEED.exists(), reason="stage1 artefacts not generated yet")
def test_fold_csvs_disjoint_and_transfer_separated() -> None:
    for name in (
        "FOLD_MANIFEST_WITHIN_EUROC.csv",
        "FOLD_MANIFEST_WITHIN_UZH.csv",
        "FOLD_MANIFEST_SOURCE_SELECTION.csv",
        "FOLD_MANIFEST_TRANSFER.csv",
    ):
        rows = _rows(name)
        ok, issues = fold_is_sequence_disjoint(rows)
        assert ok, issues
    transfer = _rows("FOLD_MANIFEST_TRANSFER.csv")
    e2u = [r for r in transfer if r["fold_id"] == "TRANSFER_EUROC_TO_UZH"]
    assert {r["dataset"] for r in e2u if r["split_role"] == "source_train"} == {"EUROC"}
    assert {r["dataset"] for r in e2u if r["split_role"] == "target_eval"} == {"UZH_FPV"}
    u2e = [r for r in transfer if r["fold_id"] == "TRANSFER_UZH_TO_EUROC"]
    assert {r["dataset"] for r in u2e if r["split_role"] == "source_train"} == {"UZH_FPV"}
    assert {r["dataset"] for r in u2e if r["split_role"] == "target_eval"} == {"EUROC"}


@pytest.mark.skipif(not (ROOT / "PROTOCOL_LOCK.md").exists(), reason="protocol lock not written")
def test_protocol_hash_file_matches() -> None:
    hashes = (STAGE1 / "PROTOCOL_LOCK_HASHES.txt").read_text(encoding="utf-8")
    md = hashlib.sha256((ROOT / "PROTOCOL_LOCK.md").read_bytes()).hexdigest()
    yml = hashlib.sha256((ROOT / "configs" / "protocol_lock.yaml").read_bytes()).hexdigest()
    assert f"PROTOCOL_LOCK.md sha256={md}" in hashes
    assert f"configs/protocol_lock.yaml sha256={yml}" in hashes
    data = yaml.safe_load((ROOT / "configs" / "protocol_lock.yaml").read_text(encoding="utf-8"))
    assert data["FIT_SCOPE_FITTED_PREPROCESSORS"] == "SOURCE_TRAIN_ONLY"
    assert data["filter"]["fit_scope"] == "NONE_DETERMINISTIC_PHYSICAL"
    assert "PRE-REGISTERED" not in (ROOT / "PROTOCOL_LOCK.md").read_text(encoding="utf-8")
    assert STAGE1_REPORT.exists(), STAGE1_REPORT
    assert "PRE-REGISTERED" not in STAGE1_REPORT.read_text(encoding="utf-8")
