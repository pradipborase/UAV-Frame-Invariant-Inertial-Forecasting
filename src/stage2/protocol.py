"""Verify the Stage-1 protocol lock has not been modified."""

from __future__ import annotations

from pathlib import Path

from audit.hashing import sha256_file

EXPECTED_PROTOCOL_MD = "b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed"
EXPECTED_PROTOCOL_YAML = "8fd75b59e35aac291f597fb856ff1debfc9609cf7a52adf6fc0e3502bf3c0d7d"


def verify_protocol_lock(root: Path) -> dict[str, str]:
    md = sha256_file(root / "PROTOCOL_LOCK.md")
    yml = sha256_file(root / "configs" / "protocol_lock.yaml")
    ok = md == EXPECTED_PROTOCOL_MD and yml == EXPECTED_PROTOCOL_YAML
    return {
        "protocol_md": md,
        "protocol_yaml": yml,
        "expected_md": EXPECTED_PROTOCOL_MD,
        "expected_yaml": EXPECTED_PROTOCOL_YAML,
        "match": "PASS" if ok else "PROTOCOL_LOCK_MISMATCH",
    }
