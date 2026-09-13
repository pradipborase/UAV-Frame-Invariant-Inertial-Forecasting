"""Candidate history is 100 samples; neural models use all 100; Ridge lag is selected from the locked grid."""

from __future__ import annotations

from pathlib import Path

import yaml

from stage2.windows import lag_slice
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_history_is_100_samples() -> None:
    proto = yaml.safe_load((ROOT / "configs" / "protocol_lock.yaml").read_text(encoding="utf-8"))
    s2 = yaml.safe_load((ROOT / "configs" / "stage2_baselines.yaml").read_text(encoding="utf-8"))
    s3 = yaml.safe_load((ROOT / "configs" / "stage3_advanced.yaml").read_text(encoding="utf-8"))
    assert proto["lookback_samples"] == 100
    assert abs(float(proto["lookback_s"]) - 1.0) < 1e-12
    assert s2["lookback_samples"] == 100
    assert s3["lookback_samples"] == 100
    assert s2["horizon_samples"] == 20
    assert s3["horizon_samples"] == 20
    assert s2["stride_samples"] == 5
    assert s3["stride_samples"] == 5


def test_neural_input_history_is_100() -> None:
    s3 = yaml.safe_load((ROOT / "configs" / "stage3_advanced.yaml").read_text(encoding="utf-8"))
    assert s3["lookback_samples"] == 100
    models = (ROOT / "src" / "stage3" / "models.py").read_text(encoding="utf-8")
    assert "lookback: int = 100" in models


def test_ridge_lag_grid_includes_10_25_50_100() -> None:
    s2 = yaml.safe_load((ROOT / "configs" / "stage2_baselines.yaml").read_text(encoding="utf-8"))
    assert s2["lags"] == [10, 25, 50, 100]
    hist = np.arange(400, dtype=np.float64).reshape(4, 100)
    for lag in (10, 25, 50, 100):
        sl = lag_slice(hist, lag)
        assert sl.shape == (4, lag)
        np.testing.assert_array_equal(sl, hist[:, -lag:])


def test_docs_qualify_that_ridge_does_not_always_use_100() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "every model uses 100 samples" not in readme
    note = (ROOT / "docs" / "INPUT_HISTORY_CLARIFICATION.md").read_text(encoding="utf-8")
    assert "candidate history" in note.lower()
    assert "[10, 25, 50, 100]" in note or "[10, 25, 50, 100]" in note.replace(" ", "")
    assert "TCN" in note and "GRU" in note and "Transformer" in note
    assert "most recent" in note.lower() or "lag" in note.lower()
