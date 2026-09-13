"""Fixed 100-lag Ridge B3 sensitivity: protocol, leakage, and primary-file integrity."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from audit.hashing import sha256_file
from stage2.fixed_lag100_sensitivity import (
    CONTEXTS,
    FIXED_LAG,
    MODEL_ID,
    PRIMARY_RIDGE_ID,
    PROTECTED_PRIMARY,
)
from stage2.protocol import verify_protocol_lock
from stage2.windows import lag_slice

ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT / "results" / "sensitivity"
NEED = SENS / "RIDGE_B3_FIXED_LAG100_SUMMARY.csv"


def test_lag_slice_100_uses_full_lookback() -> None:
    hist = np.arange(300, dtype=np.float64).reshape(3, 100)
    sl = lag_slice(hist, 100)
    assert sl.shape == (3, 100)
    np.testing.assert_array_equal(sl, hist)


def test_sensitivity_module_has_no_torch() -> None:
    text = (ROOT / "src" / "stage2" / "fixed_lag100_sensitivity.py").read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "from torch" not in text
    script = (ROOT / "scripts" / "run_ridge_fixed_lag100_sensitivity.py").read_text(encoding="utf-8")
    assert "import torch" not in script
    assert "Does not train neural models" in script


def test_alpha_grid_matches_locked_yaml() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "stage2_baselines.yaml").read_text(encoding="utf-8"))
    assert cfg["lags"] == [10, 25, 50, 100]
    assert [float(a) for a in cfg["alphas"]] == [1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0]
    assert verify_protocol_lock(ROOT)["match"] == "PASS"


@pytest.mark.skipif(not NEED.exists(), reason="fixed-lag100 sensitivity not run yet")
def test_outputs_exist_and_lag_is_100() -> None:
    for name in (
        "RIDGE_B3_FIXED_LAG100_SEQUENCE_RESULTS.csv",
        "RIDGE_B3_FIXED_LAG100_SUMMARY.csv",
        "RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv",
        "RIDGE_B3_FIXED_LAG100_COMPARISON.md",
    ):
        path = SENS / name
        assert path.exists() and path.stat().st_size > 0, path
    with (SENS / "RIDGE_B3_FIXED_LAG100_SEQUENCE_RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        seq = list(csv.DictReader(handle))
    with (SENS / "RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv").open(encoding="utf-8", newline="") as handle:
        sel = list(csv.DictReader(handle))
    with (SENS / "RIDGE_B3_FIXED_LAG100_SUMMARY.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert seq and all(int(r["lag"]) == FIXED_LAG for r in seq)
    assert sel and all(int(r["lag"]) == FIXED_LAG for r in sel)
    assert {r["model"] for r in seq} == {MODEL_ID}
    alphas = {1e-6, 1e-4, 1e-2, 1.0, 100.0}
    for r in sel:
        assert float(r["selected_alpha"]) in alphas
    ctxs = {r["evaluation_context"] for r in summary}
    assert ctxs == set(CONTEXTS)
    for col in (
        "n_sequences",
        "mean_sequence_rmse",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "mean_rmse_50ms",
        "mean_rmse_100ms",
        "mean_rmse_200ms",
    ):
        assert all(r[col] != "" for r in summary)


@pytest.mark.skipif(not NEED.exists(), reason="fixed-lag100 sensitivity not run yet")
def test_source_only_selection_and_transfer_counts() -> None:
    with (SENS / "RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv").open(encoding="utf-8", newline="") as handle:
        sel = list(csv.DictReader(handle))
    with (SENS / "RIDGE_B3_FIXED_LAG100_SEQUENCE_RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        seq = list(csv.DictReader(handle))
    for r in sel:
        if r["dataset"] == "EUROC":
            assert "indoor_" not in r["train_sequences"]
            assert "outdoor_" not in r["train_sequences"]
        if r["dataset"] == "UZH_FPV":
            assert "V1_" not in r["train_sequences"]
            assert "V2_" not in r["train_sequences"]
        held = r["held_out_sequence"]
        if held:
            assert held not in r["train_sequences"].split(",")
    euroc = [r for r in seq if r["evaluation_context"] == "WITHIN_EUROC"]
    uzh = [r for r in seq if r["evaluation_context"] == "WITHIN_UZH"]
    e2u = [r for r in seq if r["evaluation_context"] == "EUROC_TO_UZH"]
    u2e = [r for r in seq if r["evaluation_context"] == "UZH_TO_EUROC"]
    assert len(euroc) == 6
    assert len(uzh) == 8
    assert len(e2u) == 8
    assert len(u2e) == 6


@pytest.mark.skipif(not NEED.exists(), reason="fixed-lag100 sensitivity not run yet")
def test_comparison_mentions_required_statements() -> None:
    text = (SENS / "RIDGE_B3_FIXED_LAG100_COMPARISON.md").read_text(encoding="utf-8")
    assert "No neural model was retrained" in text
    assert "Target data were not used for tuning" in text
    assert "Primary manuscript results were not changed" in text
    assert "RMSE_fixed100" in text and "RMSE_primary_B3" in text


def test_primary_stage2_stage3_files_still_match_on_disk() -> None:
    freeze = ROOT / "results" / "stage2" / "SOURCE_MODELS_FROZEN.md"
    sidecar = ROOT / "results" / "stage2" / "SOURCE_MODELS_FROZEN.sha256"
    assert sha256_file(freeze) == sidecar.read_text(encoding="utf-8").strip()
    assert sha256_file(ROOT / "results" / "stage3" / "HYPERPARAMETER_GRID.csv") == (
        ROOT / "results" / "stage3" / "HYPERPARAMETER_GRID.sha256"
    ).read_text(encoding="utf-8").strip()
    with (ROOT / "results" / "stage2" / "BOOTSTRAP_SUMMARY.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    b3_u2e = [r for r in rows if r["evaluation_context"] == "UZH_TO_EUROC" and r["model"] == PRIMARY_RIDGE_ID][0]
    assert abs(float(b3_u2e["mean"]) - 0.8308037721) < 1e-8
    for rel in PROTECTED_PRIMARY:
        assert (ROOT / rel).exists(), rel
