"""Manuscript-facing publication_current uses source-selected B3 Ridge only."""

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

NEED_PUB = pytest.mark.skipif(
    not (PUB / "TABLE_PRIMARY_RESULTS.csv").exists(),
    reason="publication_current not built yet",
)


def _rows(name: str) -> list[dict[str, str]]:
    with (PUB / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_primary_ridge_is_hardcoded_b3_for_every_context() -> None:
    assert PRIMARY_RIDGE_ID == "B3_RIDGE_QF_QW"
    assert set(PRIMARY_RIDGE_BY_CONTEXT) == {"WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"}
    assert set(PRIMARY_RIDGE_BY_CONTEXT.values()) == {"B3_RIDGE_QF_QW"}
    for ctx in PRIMARY_RIDGE_BY_CONTEXT:
        assert primary_ridge_id(ctx) == "B3_RIDGE_QF_QW"


def test_source_does_not_select_ridge_from_target_rmse() -> None:
    text = (ROOT / "src" / "stage4" / "publication_current.py").read_text(encoding="utf-8")
    assert "np.argmin" not in text
    assert "PRIMARY_RIDGE_ID = \"B3_RIDGE_QF_QW\"" in text
    assert "never inspects target-domain RMSE to choose a Ridge variant" in text


@NEED_PUB
def test_ridge_comparator_map_is_b3_everywhere() -> None:
    rows = _rows("RIDGE_COMPARATOR_MAP.csv")
    assert {r["context"] for r in rows} == {"WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"}
    for row in rows:
        assert row["primary_ridge_id"] == "B3_RIDGE_QF_QW"
        assert row["role"] == "primary Ridge comparator"
        assert "supplementary sensitivity only" in row["b2_status"]


@NEED_PUB
def test_uzh_to_euroc_primary_ridge_is_b3_not_b2() -> None:
    seq = load_publication_sequences(ROOT)
    rec = summarize_model(seq, "UZH_TO_EUROC", RIDGE_NAME)
    assert abs(rec["mean_seq_rmse"] - B3_UZH_TO_EUROC) < 1e-8
    assert abs(rec["mean_seq_rmse"] - B2_UZH_TO_EUROC) > 1e-4
    assert abs(rec["rmse_50ms"] - 0.69723093717) < 1e-8
    assert abs(rec["rmse_100ms"] - 0.83473824805) < 1e-8
    assert abs(rec["rmse_200ms"] - 0.92015072882) < 1e-8
    table = [r for r in _rows("TABLE_PRIMARY_RESULTS.csv") if r["context"] == "UZH_TO_EUROC" and r["model"] == RIDGE_NAME][0]
    assert table["ridge_id"] == "B3_RIDGE_QF_QW"
    assert abs(float(table["mean_seq_rmse"]) - B3_UZH_TO_EUROC) < 1e-8


def test_readme_uses_source_selected_b3_language() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "context-specific best frozen Ridge" not in readme
    assert "source-selected B3 Ridge comparator" in readme
    assert "B3 is the primary Ridge comparator in both zero-shot transfer" in readme
    posthoc = (ROOT / "docs" / "POSTHOC_REPORTING_CORRECTION.md").read_text(encoding="utf-8")
    assert "No model was retrained" in posthoc
    assert "B3" in posthoc
    combined = (ROOT / "docs" / "POSTHOC_REPORTING_CORRECTIONS.md").read_text(encoding="utf-8")
    assert "B3_RIDGE_QF_QW" in combined
    assert "order 5" in combined
    assert "order 6" in combined
