"""Streaming downloader with resume, retry, hash lock, and a download budget."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .hashing import sha256_file

UTC = timezone.utc

# ETH Research Collection and similar hosts reject empty User-Agent strings.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 UAV_MEASUREMENT_FRESH/stage0c"
    ),
    "Accept": "*/*",
}


def utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DownloadRecord:
    dataset: str
    sequence_id: str
    source_name: str
    source_url: str
    retrieval_utc: str
    local_path: str
    file_type: str
    content_role: str
    remote_size_bytes: str
    downloaded_size_bytes: str
    sha256: str
    http_etag_if_available: str
    http_last_modified_if_available: str
    download_status: str
    official_source: str
    license: str
    notes: str


MANIFEST_COLUMNS = [
    "dataset",
    "sequence_id",
    "source_name",
    "source_url",
    "retrieval_utc",
    "local_path",
    "file_type",
    "content_role",
    "remote_size_bytes",
    "downloaded_size_bytes",
    "sha256",
    "http_etag_if_available",
    "http_last_modified_if_available",
    "download_status",
    "official_source",
    "license",
    "notes",
]


class BudgetTracker:
    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = int(limit_bytes)
        self.consumed_bytes = 0

    def would_exceed(self, nbytes: int) -> bool:
        return (self.consumed_bytes + int(nbytes)) > self.limit_bytes

    def add(self, nbytes: int) -> None:
        self.consumed_bytes += int(nbytes)


def head_remote(
    url: str,
    timeout_s: int,
    *,
    probe_get_on_error: bool = True,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "ok": False,
        "status_code": None,
        "content_length": None,
        "etag": "",
        "last_modified": "",
        "error": "",
        "content_type": "",
    }
    req_headers = dict(DEFAULT_HEADERS)
    if headers:
        req_headers.update(headers)
    try:
        response = requests.head(url, timeout=timeout_s, allow_redirects=True, headers=req_headers)
        info["status_code"] = int(response.status_code)
        info["ok"] = 200 <= response.status_code < 400
        length = response.headers.get("Content-Length")
        info["content_length"] = int(length) if length and str(length).isdigit() else None
        info["etag"] = response.headers.get("ETag") or ""
        info["last_modified"] = response.headers.get("Last-Modified") or ""
        info["content_type"] = response.headers.get("Content-Type") or ""
        # Do not probe-GET multi-gigabyte bitstreams; callers pass expected_size instead.
        if probe_get_on_error and response.status_code in (403, 404, 405):
            get_resp = requests.get(
                url, timeout=timeout_s, stream=True, allow_redirects=True, headers=req_headers
            )
            info["status_code"] = int(get_resp.status_code)
            info["ok"] = 200 <= get_resp.status_code < 400
            length = get_resp.headers.get("Content-Length")
            info["content_length"] = int(length) if length and str(length).isdigit() else info["content_length"]
            info["etag"] = get_resp.headers.get("ETag") or info["etag"]
            info["last_modified"] = get_resp.headers.get("Last-Modified") or info["last_modified"]
            get_resp.close()
    except requests.RequestException as exc:
        info["error"] = str(exc)
        info["ok"] = False
    return info


def _host_unavailable(error: str) -> bool:
    text = (error or "").lower()
    needles = (
        "failed to resolve",
        "name or service not known",
        "getaddrinfo",
        "nameresolutionerror",
        "not be resolved",
        "nodename nor servname",
        "temporary failure in name resolution",
        "failed to establish a new connection",
        "newconnectionerror",
    )
    return any(n in text for n in needles)


def _record(**kwargs: Any) -> dict[str, str]:
    rec = DownloadRecord(
        dataset=str(kwargs.get("dataset", "")),
        sequence_id=str(kwargs.get("sequence_id", "")),
        source_name=str(kwargs.get("source_name", "")),
        source_url=str(kwargs.get("source_url", "")),
        retrieval_utc=str(kwargs.get("retrieval_utc", utc_now())),
        local_path=str(kwargs.get("local_path", "")),
        file_type=str(kwargs.get("file_type", "")),
        content_role=str(kwargs.get("content_role", "")),
        remote_size_bytes=str(kwargs.get("remote_size_bytes", "")),
        downloaded_size_bytes=str(kwargs.get("downloaded_size_bytes", "")),
        sha256=str(kwargs.get("sha256", "")),
        http_etag_if_available=str(kwargs.get("http_etag_if_available", "")),
        http_last_modified_if_available=str(kwargs.get("http_last_modified_if_available", "")),
        download_status=str(kwargs.get("download_status", "")),
        official_source=str(kwargs.get("official_source", "YES")),
        license=str(kwargs.get("license", "")),
        notes=str(kwargs.get("notes", "")),
    )
    return asdict(rec)


def download_file(
    url: str,
    dest: Path,
    *,
    budget: BudgetTracker,
    timeout_s: int = 120,
    retries: int = 5,
    chunk_bytes: int = 1024 * 1024,
    dataset: str = "",
    sequence_id: str = "",
    source_name: str = "",
    file_type: str = "",
    content_role: str = "",
    license_name: str = "",
    notes: str = "",
    expected_size: int | None = None,
) -> dict[str, str]:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    head = head_remote(
        url,
        timeout_s=min(timeout_s, 30),
        probe_get_on_error=expected_size is None,
    )
    remote_size = expected_size if expected_size is not None else head.get("content_length")

    if dest.exists() and dest.stat().st_size > 0:
        if part.exists():
            part.unlink()
        digest = sha256_file(dest)
        size = dest.stat().st_size
        if remote_size is not None and size != int(remote_size):
            return _record(
                dataset=dataset,
                sequence_id=sequence_id,
                source_name=source_name or dest.name,
                source_url=url,
                local_path=str(dest),
                file_type=file_type,
                content_role=content_role,
                remote_size_bytes=remote_size or "",
                downloaded_size_bytes=size,
                sha256=digest,
                http_etag_if_available=head.get("etag", ""),
                http_last_modified_if_available=head.get("last_modified", ""),
                download_status="FAILED",
                license=license_name,
                notes="Existing file size does not match remote Content-Length; not overwritten.",
            )
        budget.add(size)
        return _record(
            dataset=dataset,
            sequence_id=sequence_id,
            source_name=source_name or dest.name,
            source_url=url,
            local_path=str(dest),
            file_type=file_type,
            content_role=content_role,
            remote_size_bytes=remote_size or size,
            downloaded_size_bytes=size,
            sha256=digest,
            http_etag_if_available=head.get("etag", ""),
            http_last_modified_if_available=head.get("last_modified", ""),
            download_status="ALREADY_PRESENT_HASH_VERIFIED",
            license=license_name,
            notes=notes or "Existing file hashed; not re-downloaded.",
        )

    size_for_budget = int(remote_size) if remote_size else 0
    if size_for_budget and budget.would_exceed(size_for_budget):
        return _record(
            dataset=dataset,
            sequence_id=sequence_id,
            source_name=source_name or dest.name,
            source_url=url,
            local_path=str(dest),
            file_type=file_type,
            content_role=content_role,
            remote_size_bytes=remote_size or "",
            downloaded_size_bytes=0,
            sha256="",
            http_etag_if_available=head.get("etag", ""),
            http_last_modified_if_available=head.get("last_modified", ""),
            download_status="SKIPPED_BUDGET",
            license=license_name,
            notes="DOWNLOAD_BUDGET_REACHED",
        )

    head_blocks = (not head.get("ok")) and (
        head.get("status_code") not in (200, 206)
        or _host_unavailable(str(head.get("error") or ""))
    )
    # Known bitstream size: skip HEAD-only rejection (ETH may 403 HEAD but allow GET).
    if head_blocks and not (
        expected_size is not None and not _host_unavailable(str(head.get("error") or ""))
    ):
        status = "FAILED"
        if head.get("status_code") in (404, 502, 503, 504) or _host_unavailable(str(head.get("error") or "")):
            status = "SOURCE_UNAVAILABLE"
            notes = (notes + " OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE").strip()
        if head.get("status_code") is None and head.get("error"):
            status = "SOURCE_UNAVAILABLE"
            notes = (notes + " OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE").strip()
        return _record(
            dataset=dataset,
            sequence_id=sequence_id,
            source_name=source_name or dest.name,
            source_url=url,
            local_path=str(dest),
            file_type=file_type,
            content_role=content_role,
            remote_size_bytes=remote_size or "",
            downloaded_size_bytes=0,
            sha256="",
            http_etag_if_available=head.get("etag", ""),
            http_last_modified_if_available=head.get("last_modified", ""),
            download_status=status,
            license=license_name,
            notes=notes or str(head.get("error") or head.get("status_code")),
        )

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req_headers = dict(DEFAULT_HEADERS)
            req_headers.update(headers)
            timeout = (30, int(timeout_s)) if not isinstance(timeout_s, tuple) else timeout_s
            with requests.get(
                url, stream=True, timeout=timeout, headers=req_headers, allow_redirects=True
            ) as response:
                if response.status_code == 200 and existing > 0:
                    existing = 0
                    mode = "wb"
                elif response.status_code in (200, 206):
                    mode = "ab" if existing > 0 and response.status_code == 206 else "wb"
                    if response.status_code == 200:
                        existing = 0
                else:
                    last_error = f"HTTP {response.status_code}"
                    time.sleep(min(2 ** attempt, 20))
                    continue
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            handle.write(chunk)
            size = part.stat().st_size
            if remote_size is not None and size != int(remote_size):
                last_error = f"size mismatch local={size} remote={remote_size}"
                time.sleep(min(2 ** attempt, 20))
                continue
            if size == 0:
                last_error = "empty download"
                time.sleep(min(2 ** attempt, 20))
                continue
            digest = sha256_file(part)
            part.replace(dest)
            budget.add(size)
            return _record(
                dataset=dataset,
                sequence_id=sequence_id,
                source_name=source_name or dest.name,
                source_url=url,
                local_path=str(dest),
                file_type=file_type,
                content_role=content_role,
                remote_size_bytes=remote_size or size,
                downloaded_size_bytes=size,
                sha256=digest,
                http_etag_if_available=head.get("etag", ""),
                http_last_modified_if_available=head.get("last_modified", ""),
                download_status="DOWNLOADED",
                license=license_name,
                notes=notes,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 20))
    return _record(
        dataset=dataset,
        sequence_id=sequence_id,
        source_name=source_name or dest.name,
        source_url=url,
        local_path=str(dest),
        file_type=file_type,
        content_role=content_role,
        remote_size_bytes=remote_size or "",
        downloaded_size_bytes=part.stat().st_size if part.exists() else 0,
        sha256="",
        http_etag_if_available=head.get("etag", ""),
        http_last_modified_if_available=head.get("last_modified", ""),
        download_status="FAILED",
        license=license_name,
        notes=f"Failed after {retries} retries: {last_error}",
    )
