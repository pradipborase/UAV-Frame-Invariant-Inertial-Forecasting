"""Primary table values must match recomputed sequence-level summaries."""

from __future__ import annotations

from stage4.io import PUB_MODELS, CONTEXTS, fnum, load_stage3_sequences
from stage4.run import fmt3, fmt_ci
from stage4.stats import all_summaries
from stage4_common import ROOT, csv_rows, require_stage4


def test_primary_table_matches_recompute() -> None:
    require_stage4()
    summaries = all_summaries(load_stage3_sequences(ROOT))
    table = csv_rows("TABLE_PRIMARY_RESULTS.csv")
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            rec = [s for s in summaries if s["context"] == ctx and s["model"] == model][0]
            row = [r for r in table if r["context"] == ctx and r["model"] == model][0]
            assert abs(rec["mean_seq_rmse"] - fnum(row["mean_seq_rmse"])) < 1e-8
            assert abs(rec["rmse_50ms"] - fnum(row["rmse_50ms"])) < 1e-8
            assert abs(rec["rmse_100ms"] - fnum(row["rmse_100ms"])) < 1e-8
            assert abs(rec["rmse_200ms"] - fnum(row["rmse_200ms"])) < 1e-8
            assert row["formatted_rmse_ci"] == fmt_ci(rec["mean_seq_rmse"], rec["ci_low"], rec["ci_high"])
            assert fmt3(rec["mean_seq_rmse"]) == f"{rec['mean_seq_rmse']:.3f}"
