"""Stage-0 acquisition: official files only, budget-capped, IMU/GT/calib preferred."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .catalogs import (
    BLACKBIRD_CSV_FILES,
    BLACKBIRD_FLIGHT_FILES,
    UZH_CALIB_FILES,
    blackbird_sequences,
    uzh_sequences,
)
from .config import Stage0Config
from .downloader import BudgetTracker, download_file
from .source_verification import verify_sources, write_source_verification


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def blackbird_file_url(flight: str, rel: str) -> str:
    root = "http://blackbird-dataset.mit.edu/BlackbirdDatasetData"
    flight = flight.strip("/")
    return f"{root}/{flight}/{rel}"


def acquire_blackbird(cfg: Stage0Config, budget: BudgetTracker, rows: list[dict[str, str]]) -> None:
    print("Probing official Blackbird data host (no unofficial mirror).", flush=True)
    selection = cfg.stage0.get("blackbird_stage0_selection", [])
    timeout = int(cfg.stage0.get("download_timeout_s", 120))
    retries = int(cfg.stage0.get("download_retries", 5))
    dest_root = cfg.blackbird_raw
    for item in selection:
        seq = item["sequence_id"]
        wanted = [
            ("csv/blackbird_slash_imu.csv", "csv", "IMU"),
            ("csv/blackbird_slash_state.csv", "csv", "GROUND_TRUTH"),
            ("groundTruthPoses.csv", "csv", "GROUND_TRUTH"),
            ("flightNormalizationOffset.csv", "csv", "CALIBRATION"),
        ]
        for rel, ftype, role in wanted:
            url = blackbird_file_url(seq, rel)
            dest = dest_root / seq.replace("/", "__") / Path(rel).name
            rec = download_file(
                url,
                dest,
                budget=budget,
                timeout_s=timeout,
                retries=retries,
                dataset="BLACKBIRD",
                sequence_id=seq,
                source_name=Path(rel).name,
                file_type=ftype,
                content_role=role,
                license_name="UNRESOLVED_DATA_LICENSE",
                notes="Official sequenceDownloader.py path. Images/videos not requested.",
            )
            if rec["download_status"] == "SOURCE_UNAVAILABLE":
                rec["notes"] = (
                    rec.get("notes", "")
                    + " OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE; no unofficial mirror used."
                ).strip()
            rows.append(rec)
            if rec["download_status"] == "SOURCE_UNAVAILABLE":
                # Do not hammer a down host with the rest of the file list.
                break
        if rows and rows[-1]["download_status"] == "SOURCE_UNAVAILABLE":
            # Mark remaining selected sequences as unavailable without extra HTTP.
            remaining = selection[selection.index(item) + 1 :]
            for later in remaining:
                rows.append(
                    {
                        "dataset": "BLACKBIRD",
                        "sequence_id": later["sequence_id"],
                        "source_name": "blackbird_slash_imu.csv",
                        "source_url": blackbird_file_url(later["sequence_id"], "csv/blackbird_slash_imu.csv"),
                        "retrieval_utc": rec.get("retrieval_utc", ""),
                        "local_path": "",
                        "file_type": "csv",
                        "content_role": "IMU",
                        "remote_size_bytes": "",
                        "downloaded_size_bytes": "0",
                        "sha256": "",
                        "http_etag_if_available": "",
                        "http_last_modified_if_available": "",
                        "download_status": "SOURCE_UNAVAILABLE",
                        "official_source": "YES",
                        "license": "UNRESOLVED_DATA_LICENSE",
                        "notes": "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE; skipped after host failure.",
                    }
                )
            break


def acquire_uzh(cfg: Stage0Config, budget: BudgetTracker, rows: list[dict[str, str]]) -> None:
    timeout = int(cfg.stage0.get("download_timeout_s", 120))
    retries = int(cfg.stage0.get("download_retries", 5))
    dest_root = cfg.uzh_raw
    license_name = "CC BY-NC-SA 3.0"
    for calib in UZH_CALIB_FILES:
        dest = dest_root / "calib" / Path(calib["url"]).name
        print("DOWNLOAD", calib["url"], flush=True)
        rows.append(
            download_file(
                calib["url"],
                dest,
                budget=budget,
                timeout_s=timeout,
                retries=retries,
                dataset="UZH_FPV",
                sequence_id=calib["calib_id"],
                source_name=Path(calib["url"]).name,
                file_type="zip",
                content_role="CALIBRATION",
                license_name=license_name,
                notes="Official Kalibr YAML archive. Raw calib bags not downloaded (images).",
            )
        )
    catalog = {row["sequence_id"]: row for row in uzh_sequences(sensor_platforms=("snapdragon",))}
    for item in cfg.stage0.get("uzh_fpv_stage0_selection", []):
        seq = item["sequence_id"]
        meta = catalog.get(seq)
        if meta is None:
            rows.append(
                {
                    "dataset": "UZH_FPV",
                    "sequence_id": seq,
                    "source_name": seq,
                    "source_url": "",
                    "retrieval_utc": "",
                    "local_path": "",
                    "file_type": "zip",
                    "content_role": "IMU_GT",
                    "remote_size_bytes": "",
                    "downloaded_size_bytes": "0",
                    "sha256": "",
                    "http_etag_if_available": "",
                    "http_last_modified_if_available": "",
                    "download_status": "FAILED",
                    "official_source": "YES",
                    "license": license_name,
                    "notes": "sequence_id not in official catalog",
                }
            )
            continue
        if meta.get("leica_raw_url"):
            dest = dest_root / "leica" / Path(meta["leica_raw_url"]).name
            rows.append(
                download_file(
                    meta["leica_raw_url"],
                    dest,
                    budget=budget,
                    timeout_s=timeout,
                    retries=retries,
                    dataset="UZH_FPV",
                    sequence_id=seq,
                    source_name=Path(meta["leica_raw_url"]).name,
                    file_type="zip",
                    content_role="LEICA_POSITION",
                    license_name=license_name,
                    notes="Official raw Leica measurements. Not withheld-GT recovery.",
                )
            )
        dest = dest_root / "sequences" / meta["v3_zip_name"]
        print("DOWNLOAD", meta["v3_url"], flush=True)
        rows.append(
            download_file(
                meta["v3_url"],
                dest,
                budget=budget,
                timeout_s=timeout,
                retries=retries,
                dataset="UZH_FPV",
                sequence_id=seq,
                source_name=meta["v3_zip_name"],
                file_type="zip",
                content_role="IMU_GT_TEXT_ARCHIVE",
                license_name=license_name,
                notes=(
                    "Official v3 zip (changelog 2020-03-30 GT time-offset fix). "
                    "Images inside archive are not extracted. Rosbag duplicate not downloaded."
                ),
            )
        )


def run_acquisition(cfg: Stage0Config) -> list[dict[str, str]]:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    report = verify_sources(cfg)
    write_source_verification(cfg, report)
    (cfg.metadata_dir / "source_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    budget = BudgetTracker(cfg.download_budget_bytes)
    rows: list[dict[str, str]] = []
    acquire_blackbird(cfg, budget, rows)
    acquire_uzh(cfg, budget, rows)
    _write_manifest(cfg.results_dir / "DOWNLOAD_MANIFEST.csv", rows)
    (cfg.metadata_dir / "download_budget.json").write_text(
        json.dumps(
            {
                "limit_bytes": budget.limit_bytes,
                "consumed_bytes": budget.consumed_bytes,
                "consumed_gib": budget.consumed_bytes / (1024**3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows
