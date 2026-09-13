"""Delta = RMSE_advanced - RMSE_Ridge; negative means the advanced model is better."""

from __future__ import annotations

import numpy as np

from stage4.io import load_stage3_sequences
from stage4.stats import paired_vs_ridge
from stage4_common import ROOT, csv_rows, require_stage4


def test_delta_formula_and_sign_convention() -> None:
    rows = load_stage3_sequences(ROOT)
    rec = paired_vs_ridge(rows, "WITHIN_UZH", "Transformer")
    assert rec["mean_paired_delta"] < 0
    assert rec["absolute_improvement"] > 0
    assert abs(rec["mean_paired_delta"] - (rec["advanced_mean"] - rec["ridge_mean"])) < 1e-12
    assert rec["wins"] > rec["losses"]


def test_euroc_to_uzh_transformer_degrades() -> None:
    rows = load_stage3_sequences(ROOT)
    rec = paired_vs_ridge(rows, "EUROC_TO_UZH", "Transformer")
    assert rec["mean_paired_delta"] > 0
    assert rec["losses"] > rec["wins"]


def test_published_delta_definition_string() -> None:
    require_stage4()
    rows = csv_rows("TABLE_ADVANCED_VS_RIDGE.csv")
    assert all("RMSE_advanced - RMSE_Ridge" in r["delta_definition"] for r in rows)


def test_relative_improvement_formula() -> None:
    ridge = 2.0
    adv = 1.8
    abs_imp = ridge - adv
    rel = 100.0 * abs_imp / ridge
    assert abs(rel - 10.0) < 1e-12
    delta = adv - ridge
    assert abs(delta + 0.2) < 1e-12
    rng = np.array([delta])
    assert rng[0] < 0
