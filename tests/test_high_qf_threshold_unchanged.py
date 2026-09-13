"""High-q_f threshold remains sequence-specific p95 of processed q_f."""

from __future__ import annotations

from stage4.io import fnum, read_csv
from stage4_common import ROOT, csv_rows, require_stage4


def test_p95_matches_stage2_predictability() -> None:
    pred = {r["sequence_id"]: fnum(r["p95_qf"]) for r in read_csv(ROOT / "results" / "stage2" / "PREDICTABILITY_CHARACTERIZATION.csv")}
    high = read_csv(ROOT / "results" / "stage3" / "HIGH_QF_ADVANCED.csv")
    for row in high:
        assert abs(fnum(row["p95_threshold"]) - pred[row["sequence_id"]]) < 1e-6


def test_difficulty_ratio_definition() -> None:
    require_stage4()
    for row in csv_rows("HIGH_QF_PUBLICATION_DATA.csv"):
        all_rmse = fnum(row["rmse_all"])
        high = fnum(row["rmse_high_qf"])
        assert abs(fnum(row["difficulty_ratio"]) - high / all_rmse) < 1e-8
        assert high > all_rmse
        assert "sequence-specific p95" in row["p95_source"]
