"""Transfer gap = zero-shot target RMSE - corresponding within-target RMSE."""

from __future__ import annotations

from stage4.io import fnum, load_stage3_sequences
from stage4.stats import summarize_model
from stage4_common import ROOT, csv_rows, require_stage4


def test_transfer_gap_formula() -> None:
    rows = load_stage3_sequences(ROOT)
    zs = summarize_model(rows, "EUROC_TO_UZH", "Best frozen Ridge")["mean_seq_rmse"]
    wi = summarize_model(rows, "WITHIN_UZH", "Best frozen Ridge")["mean_seq_rmse"]
    gap = zs - wi
    require_stage4()
    pub = [r for r in csv_rows("TRANSFER_GAP_FINAL.csv") if r["direction"] == "EUROC_TO_UZH" and r["model"] == "Best frozen Ridge"][0]
    assert abs(gap - fnum(pub["transfer_gap"])) < 1e-12
    assert gap > 0


def test_persistence_omitted_from_transfer_gap_table() -> None:
    require_stage4()
    models = {r["model"] for r in csv_rows("TRANSFER_GAP_FINAL.csv")}
    assert "Persistence" not in models
    assert "Best frozen Ridge" in models
