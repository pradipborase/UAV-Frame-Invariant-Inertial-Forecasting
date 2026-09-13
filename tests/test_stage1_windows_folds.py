"""Window counts, boundary safety, and fold disjointness."""

from __future__ import annotations

import numpy as np

from stage1.folds import (
    EUROC_ROOM,
    UZH_GROUP,
    fold_is_sequence_disjoint,
    leave_group_out,
    leave_one_recording_out,
    transfer_manifests,
)
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY
from stage1.windows import count_windows


def test_window_count_formula() -> None:
    n = 1000
    ts = np.arange(n, dtype=np.float64) * 0.01
    c = count_windows(ts, lookback_samples=100, horizon_samples=20, stride_samples=5)
    assert c["candidate_origins"] == 200
    assert c["excluded_due_to_start"] == 20  # 0,5,...,95
    # valid k on {0,5,...,995} with k>=99 and k+20<1000 -> 100..975 step 5
    assert c["valid_origins"] == 176
    assert c["excluded_due_to_gap"] == 0


def test_gap_excludes_window() -> None:
    ts = np.arange(400, dtype=np.float64) * 0.01
    ts[200:] += 1.0  # 1 s jump
    c = count_windows(ts, lookback_samples=100, horizon_samples=20, stride_samples=5)
    assert c["excluded_due_to_gap"] > 0


def test_loro_and_transfer_disjoint() -> None:
    e = leave_one_recording_out("EUROC", EUROC_PRIMARY, EUROC_ROOM)
    u = leave_one_recording_out("UZH_FPV", UZH_PRIMARY, UZH_GROUP)
    s = leave_group_out("EUROC", EUROC_PRIMARY, EUROC_ROOM, evaluation_class="SOURCE_SELECTION")
    t = transfer_manifests(EUROC_PRIMARY, UZH_PRIMARY)
    for rows in (e, u, s, t):
        ok, issues = fold_is_sequence_disjoint(rows)
        assert ok, issues
    euroc_to_uzh = [r for r in t if r["fold_id"] == "TRANSFER_EUROC_TO_UZH"]
    train_ds = {r["dataset"] for r in euroc_to_uzh if r["split_role"] == "source_train"}
    tgt_ds = {r["dataset"] for r in euroc_to_uzh if r["split_role"] == "target_eval"}
    assert train_ds == {"EUROC"}
    assert tgt_ds == {"UZH_FPV"}
    uzh_to_euroc = [r for r in t if r["fold_id"] == "TRANSFER_UZH_TO_EUROC"]
    assert {r["dataset"] for r in uzh_to_euroc if r["split_role"] == "source_train"} == {"UZH_FPV"}
    assert {r["dataset"] for r in uzh_to_euroc if r["split_role"] == "target_eval"} == {"EUROC"}


def test_euroc_loro_has_six_folds() -> None:
    e = leave_one_recording_out("EUROC", EUROC_PRIMARY, EUROC_ROOM)
    folds = {r["fold_id"] for r in e}
    assert len(folds) == 6
    u = leave_one_recording_out("UZH_FPV", UZH_PRIMARY, UZH_GROUP)
    assert len({r["fold_id"] for r in u}) == 8
