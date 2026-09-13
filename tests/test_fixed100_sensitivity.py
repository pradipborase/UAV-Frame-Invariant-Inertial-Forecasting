"""Fixed 100-lag Ridge B3 sensitivity: four contexts, lag=100, source-only, frozen aggregates."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT / "results" / "sensitivity"
SUMMARY = SENS / "RIDGE_B3_FIXED_LAG100_SUMMARY.csv"

EXPECTED_MEAN = {
    "WITHIN_EUROC": 0.7482092167,
    "WITHIN_UZH": 1.956741482,
    "EUROC_TO_UZH": 2.125631626,
    "UZH_TO_EUROC": 0.8423342158,
}
EXPECTED_ROUNDED = {
    "WITHIN_EUROC": 0.748209,
    "WITHIN_UZH": 1.956741,
    "EUROC_TO_UZH": 2.125632,
    "UZH_TO_EUROC": 0.842334,
}

NEED = pytest.mark.skipif(not SUMMARY.exists(), reason="fixed-lag100 sensitivity CSVs not shipped")


def _rows(name: str) -> list[dict[str, str]]:
    with (SENS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@NEED
def test_all_four_contexts_present_with_lag_100() -> None:
    summary = _rows("RIDGE_B3_FIXED_LAG100_SUMMARY.csv")
    seq = _rows("RIDGE_B3_FIXED_LAG100_SEQUENCE_RESULTS.csv")
    sel = _rows("RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv")
    ctxs = {r["evaluation_context"] for r in summary}
    assert ctxs == {"WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"}
    assert all(int(r["lag"]) == 100 for r in summary)
    assert all(int(r["lag"]) == 100 for r in seq)
    assert all(int(r["lag"]) == 100 for r in sel)


@NEED
def test_no_target_domain_tuning_metadata() -> None:
    sel = _rows("RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv")
    for row in sel:
        notes = row.get("notes", "").lower()
        assert "target" not in notes or "no target" in notes
        if row["dataset"] == "EUROC":
            assert "indoor_" not in row["train_sequences"]
            assert "outdoor_" not in row["train_sequences"]
        if row["dataset"] == "UZH_FPV":
            assert "V1_" not in row["train_sequences"]
            assert "V2_" not in row["train_sequences"]
    text = (SENS / "RIDGE_B3_FIXED_LAG100_COMPARISON.md").read_text(encoding="utf-8")
    assert "Target data were not used for tuning" in text
    src = (ROOT / "src" / "stage2" / "fixed_lag100_sensitivity.py").read_text(encoding="utf-8")
    assert "Target-domain recordings never enter alpha selection" in src


@NEED
def test_aggregate_results_match_frozen_csvs() -> None:
    summary = {r["evaluation_context"]: r for r in _rows("RIDGE_B3_FIXED_LAG100_SUMMARY.csv")}
    for ctx, expected in EXPECTED_MEAN.items():
        got = float(summary[ctx]["mean_sequence_rmse"])
        assert abs(got - expected) < 1e-8, (ctx, got, expected)
        assert abs(got - EXPECTED_ROUNDED[ctx]) < 5e-7, (ctx, got, EXPECTED_ROUNDED[ctx])
