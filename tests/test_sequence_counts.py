"""Inferential sequence counts remain EuRoC n=6 and UZH n=8."""

from __future__ import annotations

from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY
from stage4.io import load_stage3_sequences
from stage4_common import ROOT, csv_rows, require_stage4


def test_locked_sequence_ids() -> None:
    assert len(EUROC_PRIMARY) == 6
    assert len(UZH_PRIMARY) == 8


def test_stage3_sequence_files_have_expected_n() -> None:
    rows = load_stage3_sequences(ROOT)
    euroc = {r["sequence_id"] for r in rows if r["evaluation_context"] == "WITHIN_EUROC"}
    uzh = {r["sequence_id"] for r in rows if r["evaluation_context"] == "WITHIN_UZH"}
    assert euroc == set(EUROC_PRIMARY)
    assert uzh == set(UZH_PRIMARY)


def test_primary_table_n() -> None:
    require_stage4()
    for row in csv_rows("TABLE_PRIMARY_RESULTS.csv"):
        if row["context"] in {"WITHIN_EUROC", "UZH_TO_EUROC"}:
            assert int(row["n_sequences"]) == 6
        else:
            assert int(row["n_sequences"]) == 8
