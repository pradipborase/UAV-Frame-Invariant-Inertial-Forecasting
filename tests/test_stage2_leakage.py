"""Leakage, freeze-hash, and transfer separation tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from stage2.protocol import verify_protocol_lock

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "results" / "stage2"


def test_stage2_config_has_frozen_grid() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "stage2_baselines.yaml").read_text(encoding="utf-8"))
    assert cfg["lookback_samples"] == 100
    assert cfg["horizon_samples"] == 20
    assert cfg["linear_k"] == 10
    assert cfg["lags"] == [10, 25, 50, 100]
    assert cfg["scaler"]["fit_scope"] == "SOURCE_TRAIN_ONLY"
    assert cfg["ridge"]["recursive"] is False


@pytest.mark.skipif(not (STAGE2 / "FITTED_OBJECT_LOG.csv").exists(), reason="stage2 not run yet")
def test_fitted_objects_source_only() -> None:
    with (STAGE2 / "FITTED_OBJECT_LOG.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for r in rows:
        if r["evaluation_context"] in {"WITHIN_EUROC", "EUROC_TO_UZH", "SOURCE_SELECTION_EUROC"}:
            assert r["fit_dataset"] == "EUROC"
            assert "indoor_" not in r["fit_sequences"]
            assert "outdoor_" not in r["fit_sequences"]
        if r["evaluation_context"] in {"WITHIN_UZH", "UZH_TO_EUROC", "SOURCE_SELECTION_UZH"}:
            assert r["fit_dataset"] == "UZH_FPV"
            assert "V1_" not in r["fit_sequences"]
            assert "V2_" not in r["fit_sequences"]


@pytest.mark.skipif(not (STAGE2 / "SOURCE_MODELS_FROZEN.md").exists(), reason="stage2 not run yet")
def test_frozen_file_hash_preserved() -> None:
    from audit.hashing import sha256_file

    expected = (STAGE2 / "SOURCE_MODELS_FROZEN.sha256").read_text(encoding="utf-8").strip()
    observed = sha256_file(STAGE2 / "SOURCE_MODELS_FROZEN.md")
    assert observed == expected
    assert verify_protocol_lock(ROOT)["match"] == "PASS"


@pytest.mark.skipif(not (STAGE2 / "SCALER_MANIFEST.csv").exists(), reason="stage2 not run yet")
def test_no_target_scaler() -> None:
    with (STAGE2 / "SCALER_MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(r["target_scaler"] == "NONE" for r in rows)
    assert all(r["fit_scope"] == "SOURCE_TRAIN_ONLY" for r in rows)


@pytest.mark.skipif(not (STAGE2 / "WITHIN_EUROC_SEQUENCE_RESULTS.csv").exists(), reason="stage2 not run yet")
def test_identical_origin_counts_across_models() -> None:
    with (STAGE2 / "WITHIN_EUROC_SEQUENCE_RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_seq: dict[str, set[str]] = {}
    for r in rows:
        by_seq.setdefault(r["sequence_id"], set()).add(r["n_origins"])
    for seq, vals in by_seq.items():
        assert len(vals) == 1, (seq, vals)
