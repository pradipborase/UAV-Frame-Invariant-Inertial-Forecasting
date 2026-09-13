"""Final results lock file must exist and match its SHA-256 sidecar."""

from __future__ import annotations

from audit.hashing import sha256_file
from stage4_common import ROOT, require_stage4


def test_final_lock_hash_matches_file() -> None:
    out = require_stage4()
    lock = out / "FINAL_RESULTS_LOCK.md"
    sidecar = out / "FINAL_RESULTS_LOCK.sha256"
    assert lock.exists()
    assert sidecar.exists()
    observed = sha256_file(lock)
    recorded = sidecar.read_text(encoding="utf-8").strip()
    assert observed == recorded
    text = lock.read_text(encoding="utf-8")
    assert "primary table sha256:" in text
    assert "n_target_fitted_objects: 0" in text
    assert (ROOT / "results" / "stage4" / "FINAL_RESULTS_LOCK.md").exists()
