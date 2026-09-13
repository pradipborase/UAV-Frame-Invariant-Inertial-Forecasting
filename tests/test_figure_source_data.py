"""Figure source CSVs must reproduce displayed primary RMSE values."""

from __future__ import annotations

from pathlib import Path

import pytest

from stage4.io import CONTEXTS, PUB_MODELS, fnum
from stage4.publication_current import RIDGE_NAME
from stage4_common import csv_rows, require_stage4

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication_current"
NEED_PUB = pytest.mark.skipif(
    not (PUB / "TABLE_PRIMARY_RESULTS.csv").exists(),
    reason="publication_current not built yet",
)


def test_fig2_matches_primary_table() -> None:
    require_stage4()
    fig = csv_rows("figure_data/FIG2_primary_rmse.csv")
    table = csv_rows("TABLE_PRIMARY_RESULTS.csv")
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            a = [r for r in fig if r["context"] == ctx and r["model"] == model][0]
            b = [r for r in table if r["context"] == ctx and r["model"] == model][0]
            assert abs(fnum(a["mean_seq_rmse"]) - fnum(b["mean_seq_rmse"])) < 1e-12


def test_horizon_index_mapping() -> None:
    require_stage4()
    rows = csv_rows("HORIZON_PUBLICATION_DATA.csv")
    steps = {(int(r["horizon_step"]), int(r["horizon_ms"])) for r in rows}
    assert (1, 10) in steps
    assert (5, 50) in steps
    assert (10, 100) in steps
    assert (20, 200) in steps
    h5 = [r for r in rows if r["context"] == "WITHIN_EUROC" and r["model"] == "Best frozen Ridge" and r["horizon_step"] == "5"][0]
    table = [r for r in csv_rows("TABLE_PRIMARY_RESULTS.csv") if r["context"] == "WITHIN_EUROC" and r["model"] == "Best frozen Ridge"][0]
    assert abs(fnum(h5["mean_sequence_rmse"]) - fnum(table["rmse_50ms"])) < 1e-12


@NEED_PUB
def test_publication_current_figures_exist() -> None:
    assert PUB.exists(), PUB
    for stem in (
        "Fig01_Protocol_Schematic",
        "Fig02_Primary_RMSE",
        "Fig03_Transfer_Gap",
        "Fig04_Horizon_RMSE",
        "Fig05_Advanced_vs_Ridge",
    ):
        path = PUB / "figures" / f"{stem}.pdf"
        assert path.exists() and path.stat().st_size > 0, path
    assert not (PUB / "figures" / "Fig06_High_qf.pdf").exists()
    fig = list((PUB / "figure_data" / "FIG2_primary_rmse.csv").open(encoding="utf-8"))[0]
    assert "model" in fig


@NEED_PUB
def test_publication_current_fig2_matches_primary_table() -> None:
    import csv

    fig_path = PUB / "figure_data" / "FIG2_primary_rmse.csv"
    table_path = PUB / "TABLE_PRIMARY_RESULTS.csv"
    assert fig_path.exists() and table_path.exists()
    with fig_path.open(encoding="utf-8", newline="") as handle:
        fig = list(csv.DictReader(handle))
    with table_path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    for ctx in CONTEXTS:
        a = [r for r in fig if r["context"] == ctx and r["model"] == RIDGE_NAME][0]
        b = [r for r in table if r["context"] == ctx and r["model"] == RIDGE_NAME][0]
        assert abs(fnum(a["mean_seq_rmse"]) - fnum(b["mean_seq_rmse"])) < 1e-12


@NEED_PUB
def test_publication_current_horizon_has_all_20_steps() -> None:
    import csv

    path = PUB / "HORIZON_PUBLICATION_DATA.csv"
    assert path.exists()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = {(int(r["horizon_step"]), int(r["horizon_ms"])) for r in rows}
    assert steps == {(h, h * 10) for h in range(1, 21)}
    for ctx in CONTEXTS:
        n = sum(1 for r in rows if r["context"] == ctx and r["model"] == RIDGE_NAME)
        assert n == 20
