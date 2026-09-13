"""Primary Ridge comparator is source-selected B3 in both transfer directions."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from stage4.publication_current import (
    PRIMARY_RIDGE_BY_CONTEXT,
    PRIMARY_RIDGE_ID,
    RIDGE_NAME,
    load_publication_sequences,
    primary_ridge_id,
    summarize_model,
)

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication_current"
B3_UZH_TO_EUROC = 0.83080377215
B2_UZH_TO_EUROC = 0.8153206077

NEED = pytest.mark.skipif(
    not (PUB / "RIDGE_COMPARATOR_MAP.csv").exists(),
    reason="publication_current not built yet",
)


def _rows(name: str) -> list[dict[str, str]]:
    with (PUB / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_b3_is_primary_in_both_transfer_directions() -> None:
    assert PRIMARY_RIDGE_ID == "B3_RIDGE_QF_QW"
    assert PRIMARY_RIDGE_BY_CONTEXT["EUROC_TO_UZH"] == "B3_RIDGE_QF_QW"
    assert PRIMARY_RIDGE_BY_CONTEXT["UZH_TO_EUROC"] == "B3_RIDGE_QF_QW"
    assert primary_ridge_id("EUROC_TO_UZH") == "B3_RIDGE_QF_QW"
    assert primary_ridge_id("UZH_TO_EUROC") == "B3_RIDGE_QF_QW"
    assert primary_ridge_id("WITHIN_EUROC") == "B3_RIDGE_QF_QW"
    assert primary_ridge_id("WITHIN_UZH") == "B3_RIDGE_QF_QW"


@NEED
def test_uzh_to_euroc_primary_ridge_is_approximately_0_830804() -> None:
    seq = load_publication_sequences(ROOT)
    rec = summarize_model(seq, "UZH_TO_EUROC", RIDGE_NAME)
    assert abs(rec["mean_seq_rmse"] - B3_UZH_TO_EUROC) < 1e-8
    assert abs(rec["mean_seq_rmse"] - 0.830804) < 5e-7
    assert abs(rec["mean_seq_rmse"] - B2_UZH_TO_EUROC) > 1e-4
    for name in ("TABLE_PRIMARY_RESULTS.csv", "PRIMARY_RESULTS.csv"):
        path = PUB / name
        if not path.exists():
            continue
        table = [r for r in _rows(name) if r["context"] == "UZH_TO_EUROC" and r["model"] == RIDGE_NAME][0]
        assert table["ridge_id"] == "B3_RIDGE_QF_QW"
        assert abs(float(table["mean_seq_rmse"]) - B3_UZH_TO_EUROC) < 1e-8


@NEED
def test_comparator_map_is_b3_for_all_four_contexts() -> None:
    rows = _rows("RIDGE_COMPARATOR_MAP.csv")
    assert {r["context"] for r in rows} == {"WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"}
    for row in rows:
        assert row["primary_ridge_id"] == "B3_RIDGE_QF_QW"
        assert row["role"] == "primary Ridge comparator"
