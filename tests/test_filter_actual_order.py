"""Executed elliptic SOS order is 5 at 200 Hz and 6 at 500 Hz; coefficients match the freeze."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stage1.causal_filter import design_antialias_sos, iir_order_from_sos, spec_metrics

ROOT = Path(__file__).resolve().parents[1]

# Frozen executed SOS copied from results/stage1/FILTER_SPECIFICATION.md (coefficients only).
FROZEN_SOS_200 = np.array(
    [
        [0.0091318170074144, 0.005350949480373, 0.0091318170074144, 1.0, -0.7154723854883437, 0.0],
        [1.0, 1.0, 0.0, 1.0, -1.2780591166411965, 0.65515191910096],
        [1.0, -0.2390030141857213, 1.0, 1.0, -1.1149960257436253, 0.890164733124431],
    ],
    dtype=np.float64,
)
FROZEN_SOS_500 = np.array(
    [
        [
            1.5855758865265046e-03,
            -5.8721120498604075e-04,
            1.5855758865265042e-03,
            1.0,
            -1.7896232979513227e00,
            8.1342355958889545e-01,
        ],
        [
            1.0,
            -1.6050513520512639e00,
            9.9999999999999967e-01,
            1.0,
            -1.8055888662122941e00,
            8.9544045541022332e-01,
        ],
        [
            1.0,
            -1.7425815108590852e00,
            9.9999999999999989e-01,
            1.0,
            -1.8324197734320145e00,
            9.7025346715314031e-01,
        ],
    ],
    dtype=np.float64,
)


def test_order_200_hz_is_five() -> None:
    sos = design_antialias_sos(200.0)
    assert iir_order_from_sos(sos) == 5
    assert spec_metrics(sos, 200.0)["order"] == 5
    assert iir_order_from_sos(FROZEN_SOS_200) == 5
    assert FROZEN_SOS_200.shape[0] == 3
    assert abs(FROZEN_SOS_200[0, 5]) < 1e-12


def test_order_500_hz_is_six() -> None:
    sos = design_antialias_sos(500.0)
    assert iir_order_from_sos(sos) == 6
    assert spec_metrics(sos, 500.0)["order"] == 6
    assert iir_order_from_sos(FROZEN_SOS_500) == 6
    assert FROZEN_SOS_500.shape[0] == 3


def test_live_coefficients_match_frozen_executed_sos() -> None:
    live200 = design_antialias_sos(200.0)
    live500 = design_antialias_sos(500.0)
    np.testing.assert_allclose(live200, FROZEN_SOS_200, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(live500, FROZEN_SOS_500, rtol=1e-12, atol=1e-12)


def test_historical_stage1_files_were_not_rewritten() -> None:
    hist = (ROOT / "results" / "stage1" / "FILTER_SPECIFICATION.md").read_text(encoding="utf-8")
    euroc = hist.split("## EuRoC design", 1)[1].split("## UZH design", 1)[0]
    assert "- order: 6 (3 SOS sections)" in euroc
    snap = (ROOT / "audit_original" / "results" / "stage1" / "FILTER_SPECIFICATION.md").read_text(encoding="utf-8")
    assert snap == hist
