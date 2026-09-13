"""SHA-256 helpers. Raw files are hash-locked after download."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
