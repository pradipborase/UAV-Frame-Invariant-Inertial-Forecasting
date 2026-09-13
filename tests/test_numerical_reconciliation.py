"""Recomputed publication numbers must match frozen Stage-2/3 summaries."""

from __future__ import annotations

import numpy as np

from stage4.io import fnum, load_stage3_sequences
from stage4.stats import summarize_model
from stage4_common import ROOT, csv_rows, require_stage4


def test_reconciliation_csv_all_pass() -> None:
    require_stage4()
    rows = csv_rows("NUMERICAL_RECONCILIATION.csv")
    fails = [r for r in rows if r["status"] != "PASS"]
    assert not fails, fails[:8]


def test_independent_euroc_ridge_mean() -> None:
    seq = load_stage3_sequences(ROOT)
    rec = summarize_model(seq, "WITHIN_EUROC", "Best frozen Ridge")
    reported = [r for r in csv_rows("TABLE_PRIMARY_RESULTS.csv") if r["context"] == "WITHIN_EUROC" and r["model"] == "Best frozen Ridge"][0]
    assert abs(rec["mean_seq_rmse"] - fnum(reported["mean_seq_rmse"])) < 1e-10
    assert rec["n_sequences"] == 6


def test_equal_sequence_mean_definition() -> None:
    vals = np.array([0.4, 0.6, 0.8])
    assert abs(float(np.mean(vals)) - 0.6) < 1e-12
