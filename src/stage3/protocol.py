"""Verify Stage-1 protocol lock and Stage-2 source freeze have not been modified."""

from __future__ import annotations

from pathlib import Path

from audit.hashing import sha256_file

EXPECTED_PROTOCOL_MD = "b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed"
EXPECTED_PROTOCOL_YAML = "8fd75b59e35aac291f597fb856ff1debfc9609cf7a52adf6fc0e3502bf3c0d7d"
EXPECTED_STAGE2_FREEZE = "0aba0b20ddc30136de783b9f98a1a475915ee3ef20577e03efec8be1c15bb892"
EXPECTED_STAGE2_YAML = "0073b683fa0420810c8225be00fb649f11c3616949f07b4b97bae710700a4561"


def verify_stage3_prerequisites(root: Path) -> dict[str, str]:
    md = sha256_file(root / "PROTOCOL_LOCK.md")
    yml = sha256_file(root / "configs" / "protocol_lock.yaml")
    freeze = sha256_file(root / "results" / "stage2" / "SOURCE_MODELS_FROZEN.md")
    s2 = sha256_file(root / "configs" / "stage2_baselines.yaml")
    ok = (
        md == EXPECTED_PROTOCOL_MD
        and yml == EXPECTED_PROTOCOL_YAML
        and freeze == EXPECTED_STAGE2_FREEZE
        and s2 == EXPECTED_STAGE2_YAML
    )
    return {
        "protocol_md": md,
        "protocol_yaml": yml,
        "stage2_freeze": freeze,
        "stage2_yaml": s2,
        "expected_md": EXPECTED_PROTOCOL_MD,
        "expected_yaml": EXPECTED_PROTOCOL_YAML,
        "expected_stage2_freeze": EXPECTED_STAGE2_FREEZE,
        "expected_stage2_yaml": EXPECTED_STAGE2_YAML,
        "match": "PASS" if ok else "PROTOCOL_LOCK_MISMATCH",
    }
