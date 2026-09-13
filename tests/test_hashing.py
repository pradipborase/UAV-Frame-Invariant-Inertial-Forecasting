from __future__ import annotations

from pathlib import Path

import pytest

from audit.hashing import sha256_bytes, sha256_file
from audit.downloader import MANIFEST_COLUMNS, DownloadRecord


def test_sha256_known_vector(tmp_path: Path) -> None:
    payload = b"stage0-hash-fixture"
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_distinguishes_files(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    assert sha256_file(a) != sha256_file(b)


def test_download_manifest_schema_fields() -> None:
    rec = DownloadRecord(
        dataset="UZH_FPV",
        sequence_id="fixture",
        source_name="imu.txt",
        source_url="https://example.invalid/imu.txt",
        retrieval_utc="2026-01-01T00:00:00Z",
        local_path="data/raw/uzh_fpv/fixture/imu.txt",
        file_type="txt",
        content_role="IMU",
        remote_size_bytes="10",
        downloaded_size_bytes="10",
        sha256="abc",
        http_etag_if_available="",
        http_last_modified_if_available="",
        download_status="DOWNLOADED",
        official_source="YES",
        license="CC BY-NC-SA 3.0",
        notes="fixture",
    )
    assert list(rec.__dataclass_fields__.keys()) == MANIFEST_COLUMNS
    allowed = {
        "DOWNLOADED",
        "ALREADY_PRESENT_HASH_VERIFIED",
        "FAILED",
        "SKIPPED_BUDGET",
        "SKIPPED_NOT_REQUIRED",
        "SOURCE_UNAVAILABLE",
    }
    assert rec.download_status in allowed
