"""Bootstrap and CIs use sequences, not windows, as resampling units."""

from __future__ import annotations

import inspect

import numpy as np

from stage2.bootstrap import bootstrap_mean_ci
from stage2.metrics import equal_sequence_mean
from stage4 import stats as stage4_stats
from stage4.io import BOOT_REPS, BOOT_SEED
from stage4.io import sequence_values
from stage4.stats import summarize_model
from stage4.io import load_stage3_sequences
from stage4_common import ROOT


def test_bootstrap_resamples_sequence_vector_length() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    boot = bootstrap_mean_ci(v, n_reps=BOOT_REPS, seed=BOOT_SEED)
    assert boot["mean"] == float(np.mean(v))
    assert boot["ci_low"] < boot["mean"] < boot["ci_high"]
    src = inspect.getsource(bootstrap_mean_ci)
    assert "integers(0, n" in src
    assert "n_reps" in src


def test_summarize_uses_one_rmse_per_sequence() -> None:
    rows = load_stage3_sequences(ROOT)
    rec = summarize_model(rows, "WITHIN_EUROC", "Persistence")
    assert rec["n_sequences"] == 6
    vals = sequence_values(rows, "WITHIN_EUROC", "Persistence", "rmse_all_horizons")
    assert vals.size == 6
    assert abs(rec["mean_seq_rmse"] - equal_sequence_mean(vals)) < 1e-12


def test_locked_bootstrap_constants() -> None:
    assert BOOT_REPS == 10000
    assert BOOT_SEED == 20260912
    src = inspect.getsource(stage4_stats.summarize_model)
    assert "BOOT_REPS" in src
    assert "sequence_values" in src
