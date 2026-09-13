"""Protocol schema, FIT_SCOPE, and hash reproducibility."""

from __future__ import annotations

from pathlib import Path

import yaml

from audit.hashing import sha256_bytes
from stage1.run import _protocol_lock_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_sha256_reproducible() -> None:
    assert sha256_bytes(b"stage1-lock") == sha256_bytes(b"stage1-lock")
    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_protocol_yaml_schema() -> None:
    data = yaml.safe_load(_protocol_lock_yaml())
    assert data["primary_target"] == "q_f"
    assert data["common_sampling_rate_hz"] == 100
    assert data["inferential_unit"] == "SEQUENCE"
    assert data["future_normalization_rule"] == "SOURCE_TRAIN_ONLY"
    assert data["FIT_SCOPE_FITTED_PREPROCESSORS"] == "SOURCE_TRAIN_ONLY"
    assert data["filter"]["fit_scope"] == "NONE_DETERMINISTIC_PHYSICAL"
    assert data["filter"]["causal"] is True or data["filter"]["requirements"]["causal"] is True
    assert "EUROC_TO_UZH" in data["transfer_directions"]
    assert data["bootstrap_repetitions"] == 10000
    assert data["terminology"] == "PRE_SPECIFIED_LOCKED_PROTOCOL"
    assert "PRE-REGISTERED" not in _protocol_lock_yaml()


def test_stage1_config_exists() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "stage1.yaml").read_text(encoding="utf-8"))
    assert cfg["common_rate_hz"] == 100
    assert len(cfg["euroc_primary"]) == 6
    assert len(cfg["uzh_primary"]) == 8
