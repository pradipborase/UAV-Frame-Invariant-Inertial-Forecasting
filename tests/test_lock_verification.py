"""Frozen protocol and model-lock hashes must match Stage-4 expected values."""

from __future__ import annotations

from audit.hashing import sha256_file
from stage4.io import read_csv
from stage4.locks import EXPECTED, locks_ok, verify_locks
from stage4_common import ROOT, require_stage4


def test_live_lock_hashes_match_expected() -> None:
    rows = verify_locks(ROOT)
    assert locks_ok(rows), rows
    for rel, expected in EXPECTED.items():
        assert sha256_file(ROOT / rel) == expected


def test_lock_verification_csv_all_pass() -> None:
    require_stage4()
    rows = read_csv(ROOT / "results" / "stage4" / "LOCK_VERIFICATION.csv")
    assert len(rows) == len(EXPECTED)
    assert all(r["status"] == "PASS" for r in rows)
