"""Frozen artifact hashes for Stage-4 lock verification. No model fitting."""

from __future__ import annotations

from pathlib import Path

from audit.hashing import sha256_file

EXPECTED = {
    "PROTOCOL_LOCK.md": "b90e67caf7e44264ce02a3a2b6a9b9cd1925719b5201d0e77f1aafd4501defed",
    "configs/protocol_lock.yaml": "8fd75b59e35aac291f597fb856ff1debfc9609cf7a52adf6fc0e3502bf3c0d7d",
    "configs/stage2_baselines.yaml": "0073b683fa0420810c8225be00fb649f11c3616949f07b4b97bae710700a4561",
    "results/stage2/SOURCE_MODELS_FROZEN.md": "0aba0b20ddc30136de783b9f98a1a475915ee3ef20577e03efec8be1c15bb892",
    "configs/stage3_advanced.yaml": "a8453474ea26a552f8414408e91456da78ebf63b4b88b90dbd950540f72e9700",
    "results/stage3/HYPERPARAMETER_GRID.csv": "c5105e181542ad6afa8457acb598709ce82603c15aae75800756098da2eb123f",
    "results/stage3/STAGE3_SOURCE_MODELS_FROZEN.md": "f4be16be714f2e6ad6d9ec4628fe7346dbf2294713604b4c158f4771ac072bb0",
}


def verify_locks(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel, expected in EXPECTED.items():
        path = root / rel
        if not path.exists():
            rows.append(
                {
                    "artifact": rel,
                    "expected_hash": expected,
                    "observed_hash": "",
                    "status": "MISSING",
                    "notes": "critical frozen artifact not found",
                }
            )
            continue
        observed = sha256_file(path)
        ok = observed == expected
        rows.append(
            {
                "artifact": rel,
                "expected_hash": expected,
                "observed_hash": observed,
                "status": "PASS" if ok else "FAIL",
                "notes": "hash match" if ok else "STAGE4_FAIL_LOCK_MISMATCH",
            }
        )
    return rows


def locks_ok(rows: list[dict[str, str]]) -> bool:
    return all(r["status"] == "PASS" for r in rows)
