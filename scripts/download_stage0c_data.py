"""Download official EuRoC Vicon archives (Phase A) within the 15 GiB Stage-0C budget."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit.config import load_yaml  # noqa: E402
from audit.downloader import BudgetTracker, download_file, utc_now  # noqa: E402

MANIFEST_COLUMNS = [
    "sequence_id",
    "official_source",
    "source_url",
    "archive_name",
    "retrieval_utc",
    "local_path",
    "remote_size_bytes",
    "downloaded_size_bytes",
    "sha256",
    "etag",
    "last_modified",
    "download_status",
    "extraction_status",
    "notes",
]


def _status_map(raw: str) -> str:
    if raw in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}:
        return "HASH_VERIFIED" if raw != "DOWNLOADED" else "DOWNLOADED"
    if raw == "DOWNLOADED":
        return "DOWNLOADED"
    return raw


def main() -> int:
    cfg = load_yaml(ROOT / "configs" / "stage0c.yaml")
    euroc = cfg["euroc"]
    budget = BudgetTracker(int(cfg["download_budget_bytes"]))
    dest_dir = ROOT / "data" / "raw" / "euroc"
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = ROOT / "results" / "stage0c"
    results.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    base = euroc["bitstream_base"].rstrip("/")
    timeout_s = int(cfg.get("download_timeout_s", 600))
    retries = int(cfg.get("download_retries", 6))
    chunk = int(cfg.get("download_chunk_bytes", 1048576))

    vicon_bytes = sum(int(a["size_bytes"]) for a in euroc["phase_a_archives"])
    print(f"Phase A official Vicon archives: {vicon_bytes} bytes ({vicon_bytes / (1024 ** 3):.3f} GiB)")
    print(f"Stage 0C additional budget: {budget.limit_bytes} bytes ({budget.limit_bytes / (1024 ** 3):.3f} GiB)")
    if vicon_bytes > budget.limit_bytes:
        print("STOP: Phase A exceeds 15 GiB additional budget. Not downloading.")
        (results / "EUROC_BUDGET_STOP.txt").write_text(
            f"required_additional_bytes={vicon_bytes}\nbudget_bytes={budget.limit_bytes}\n",
            encoding="utf-8",
        )
        return 2

    for archive in euroc["phase_a_archives"]:
        url = f"{base}/{archive['uuid']}/content"
        dest = dest_dir / archive["name"]
        print(f"Downloading {archive['name']} ({archive['size_bytes']} bytes) from official ETH bitstream...")
        rec = download_file(
            url,
            dest,
            budget=budget,
            timeout_s=timeout_s,
            retries=retries,
            chunk_bytes=chunk,
            dataset="EUROC",
            sequence_id=",".join(archive["sequences"]),
            source_name=archive["name"],
            file_type="zip",
            content_role="EUROC_VICON_BUNDLE",
            license_name=euroc["license"],
            notes="Official ETH Research Collection ORIGINAL bitstream. Nested sequence zip+bag.",
            expected_size=int(archive["size_bytes"]),
        )
        status = rec["download_status"]
        if status == "DOWNLOADED" and rec.get("sha256"):
            mapped = "HASH_VERIFIED"
        elif status == "ALREADY_PRESENT_HASH_VERIFIED":
            mapped = "HASH_VERIFIED"
        else:
            mapped = status
        rows.append(
            {
                "sequence_id": rec["sequence_id"],
                "official_source": "YES",
                "source_url": rec["source_url"],
                "archive_name": rec["source_name"],
                "retrieval_utc": rec["retrieval_utc"],
                "local_path": rec["local_path"],
                "remote_size_bytes": rec["remote_size_bytes"],
                "downloaded_size_bytes": rec["downloaded_size_bytes"],
                "sha256": rec["sha256"],
                "etag": rec["http_etag_if_available"],
                "last_modified": rec["http_last_modified_if_available"],
                "download_status": mapped,
                "extraction_status": "NOT_EXTRACTED",
                "notes": rec["notes"],
            }
        )
        print(f"  status={mapped} size={rec['downloaded_size_bytes']} sha256={rec['sha256'][:16] if rec['sha256'] else ''}...")
        if mapped in {"FAILED", "SOURCE_UNAVAILABLE", "SKIPPED_BUDGET", "PARTIAL"}:
            break

    manifest = results / "EUROC_DOWNLOAD_MANIFEST.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "utc": utc_now(),
        "consumed_bytes": budget.consumed_bytes,
        "budget_bytes": budget.limit_bytes,
        "rows": rows,
    }
    (results / "EUROC_DOWNLOAD_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Consumed {budget.consumed_bytes} / {budget.limit_bytes} bytes")
    return 0 if rows and all(r["download_status"] in {"DOWNLOADED", "HASH_VERIFIED"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
