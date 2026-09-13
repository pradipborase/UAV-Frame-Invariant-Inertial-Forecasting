"""Anti-alias specification, stability, and 100 Hz integer decimation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stage1.causal_filter import GPASS_DB, GSTOP_DB, design_antialias_sos, is_stable, spec_metrics
from stage1.pipeline import process_invariant_stream, rate_from_timestamps

ROOT = Path(__file__).resolve().parents[1]


def test_filter_meets_spec_both_rates() -> None:
    for fs in (200.0, 500.0):
        sos = design_antialias_sos(fs)
        assert is_stable(sos)
        m = spec_metrics(sos, fs)
        assert m["meets_passband"], m
        assert m["meets_stopband"], m
        assert m["max_passband_loss_db"] <= GPASS_DB + 1e-6
        assert m["min_stopband_atten_db"] >= GSTOP_DB - 1e-6
        assert m["order"] >= 1


def test_output_rate_near_100hz() -> None:
    fs = 200.0
    n = 4000
    ts = np.arange(n, dtype=np.float64) / fs
    qf = np.ones(n) * 9.81
    qw = np.zeros(n)
    out = process_invariant_stream(
        ts, qf, qw, native_rate_hz=fs, decimation_factor=2, output_rate_hz=100.0, warmup_s=1.0
    )
    rates = rate_from_timestamps(out["timestamp_s"])
    assert rates["strictly_monotonic"]
    assert rates["median_hz"] is not None
    assert abs(rates["median_hz"] - 100.0) < 1e-9
    assert float(out["timestamp_s"][0]) >= 1.0 - 1e-12


def test_source_has_no_zero_phase_filter() -> None:
    hits = []
    for path in (ROOT / "src").rglob("*.py"):
        if "stage4" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            for tok in ("filtfilt", "sosfiltfilt"):
                if tok in line and "forbidden" not in line.lower():
                    hits.append(f"{path}:{i}:{tok}")
    assert not hits, hits
