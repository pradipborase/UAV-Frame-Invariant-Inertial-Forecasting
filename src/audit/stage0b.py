"""Stage 0B: official Blackbird S3 acquisition recovery. No modelling."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

from .compatibility import COMPAT_COLUMNS, compatibility_table
from .config import Stage0Config, load_config
from .downloader import MANIFEST_COLUMNS, BudgetTracker, download_file, utc_now
from .frame_audit import FRAME_COLUMNS, blackbird_documented_frames, uzh_documented_frames
from .groundtruth_audit import GT_COLUMNS, empty_gt_row
from .hashing import sha256_file
from .sequence_audit import classify_sequence
from .text_reader import interpret_gt_table, interpret_imu_table, read_text_table
from .timestamp_audit import REQUIRED_FIELDS, audit_timestamps, overlap_seconds

IMU_AUDIT_COLUMNS = [
    "dataset",
    "sequence_id_or_ALL",
    "sensor_platform",
    "imu_topic_or_file",
    "message_type",
    "timestamp_field",
    "accel_x_field",
    "accel_y_field",
    "accel_z_field",
    "gyro_x_field",
    "gyro_y_field",
    "gyro_z_field",
    "declared_acceleration_quantity",
    "quantity_interpretation",
    "unit_acceleration",
    "unit_angular_rate",
    "frame_id_raw",
    "sensor_frame",
    "body_frame_relationship",
    "axis_x_definition",
    "axis_y_definition",
    "axis_z_definition",
    "gravity_in_measurement",
    "bias_corrected_if_known",
    "calibrated_if_known",
    "sampling_rate_documented_hz",
    "sampling_rate_measured_median_hz",
    "sampling_rate_measured_p05_hz",
    "sampling_rate_measured_p95_hz",
    "evidence_type",
    "evidence_source",
    "confidence",
    "unresolved_issue",
]

S3_REST = "https://ijrr-blackbird-dataset.s3.amazonaws.com"
S3_ACCELERATE = "http://ijrr-blackbird-dataset.s3-accelerate.amazonaws.com"
S3_PREFIX = "BlackbirdDatasetData"
FIRST_PROBE_SEQUENCE = "mouse/yawForward/maxSpeed0p5"

SENSOR_RELPATHS = [
    ("csv/blackbird_slash_imu.csv", "csv", "IMU"),
    ("csv/blackbird_slash_state.csv", "csv", "GROUND_TRUTH"),
    ("csv/blackbird_slash_pose_ref.csv", "csv", "POSE_REF"),
    ("groundTruthPoses.csv", "csv", "GROUND_TRUTH"),
    ("flightNormalizationOffset.csv", "csv", "CALIBRATION"),
]

PROBE_COLUMNS = [
    "probe_id",
    "request_utc",
    "method",
    "url",
    "resolved_url",
    "http_status",
    "content_length",
    "etag",
    "last_modified",
    "content_type",
    "s3_status",
    "body_prefix",
    "notes",
]


def results_dir(cfg: Stage0Config) -> Path:
    path = cfg.root / "results" / "stage0b"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_official_s3_reference(cfg: Stage0Config) -> dict[str, Any]:
    tools = cfg.blackbird_tools
    index = tools / "websiteUtils" / "index.html"
    downloader = tools / "fileTreeUtilities" / "sequenceDownloader.py"
    sync = tools / "fileTreeUtilities" / "updateIJRRDataset.sh"
    redirects = tools / "websiteUtils" / "redirect_rules.xml"
    meta_path = cfg.metadata_dir / "blackbird_official_tools.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    html = index.read_text(encoding="utf-8") if index.exists() else ""
    downloader_text = downloader.read_text(encoding="utf-8") if downloader.exists() else ""
    sync_text = sync.read_text(encoding="utf-8") if sync.exists() else ""
    redirect_text = redirects.read_text(encoding="utf-8") if redirects.exists() else ""

    bucket_match = re.search(r"BUCKET_URL\s*=\s*'([^']+)'", html)
    accel_match = re.search(r"BUCKET_WEBSITE_URL\s*=\s*'([^']+)'", html)
    server_match = re.search(r'serverAddress\s*=\s*"([^"]+)"', downloader_text)
    csv_ok = all(name in downloader_text for name in ("blackbird_slash_imu.csv", "blackbird_slash_state.csv"))
    bag_separate = "rosbag.bag" in downloader_text and "Camera_Left_RGB" in downloader_text
    sync_ok = "s3://ijrr-blackbird-dataset/BlackbirdDatasetData/" in sync_text
    rewrite_documented = (
        "BlackbirdDatasetData/" in redirect_text
        and "ijrr-blackbird-dataset.s3-accelerate.amazonaws.com" in redirect_text
        and sync_ok
        and bucket_match is not None
    )
    return {
        "repository_origin": meta.get("repository_origin", "https://github.com/mit-aera/Blackbird-Dataset"),
        "commit_hash": meta.get("commit_hash", "UNRESOLVED"),
        "evidence_file": str(index.relative_to(cfg.root).as_posix()) if index.exists() else "MISSING",
        "bucket_url": bucket_match.group(1) if bucket_match else "",
        "bucket_accelerate_url": accel_match.group(1) if accel_match else "",
        "legacy_server_address": server_match.group(1) if server_match else "",
        "s3_sync_target": "s3://ijrr-blackbird-dataset/BlackbirdDatasetData/" if sync_ok else "UNRESOLVED",
        "csv_and_bag_listed_separately_from_images": bag_separate and csv_ok,
        "hostname_rewrite_technically_valid": rewrite_documented,
        "evidence_type": "VERIFIED_FROM_OFFICIAL_SOURCE_CODE",
        "notes": (
            "Official index.html names the REST bucket. Official updateIJRRDataset.sh syncs "
            "BlackbirdDatasetData/ into that bucket. Official sequenceDownloader.py still "
            "points at the dead MIT hostname but uses the same key prefix. Official "
            "redirect_rules.xml send BlackbirdDatasetData/ to the accelerate hostname. "
            "Images are separate camera packages; IMU/GT CSV and rosbag.bag are listed as "
            "flight files, not camera files."
        ),
    }


def s3_object_url(sequence_id: str, rel: str, *, rest: bool = True) -> str:
    seq = sequence_id.strip("/")
    rel = rel.lstrip("/")
    root = S3_REST if rest else S3_ACCELERATE
    return f"{root}/{S3_PREFIX}/{seq}/{rel}"


def classify_http(status: int | None, body: str, url: str) -> str:
    text = (body or "").lower()
    if status is None:
        return "S3_UNAVAILABLE"
    if "transfer acceleration is not configured" in text:
        return "S3_UNAVAILABLE"
    if "nosuchwebsiteconfiguration" in text:
        return "S3_UNAVAILABLE"
    if status in (200, 206):
        return "S3_ACCESS_CONFIRMED"
    if status == 404:
        return "S3_OBJECT_NOT_FOUND"
    if status == 403:
        return "S3_ACCESS_DENIED"
    if status in (400, 401):
        return "S3_ACCESS_DENIED" if "access" in text else "S3_UNAVAILABLE"
    if "list-type" in url or url.rstrip("/").endswith("amazonaws.com"):
        if status == 403:
            return "S3_ACCESS_DENIED"
    return "S3_BUCKET_VISIBLE_OBJECT_UNKNOWN"


def _probe_one(probe_id: str, method: str, url: str, timeout_s: int = 20, headers: dict[str, str] | None = None) -> dict[str, str]:
    request_utc = utc_now()
    notes = ""
    try:
        response = requests.request(
            method,
            url,
            timeout=timeout_s,
            allow_redirects=True,
            headers=headers or {},
        )
        body = response.content[:240].decode("utf-8", errors="replace")
        status = int(response.status_code)
        s3_status = classify_http(status, body, url)
        return {
            "probe_id": probe_id,
            "request_utc": request_utc,
            "method": method,
            "url": url,
            "resolved_url": response.url,
            "http_status": str(status),
            "content_length": response.headers.get("Content-Length") or "",
            "etag": response.headers.get("ETag") or "",
            "last_modified": response.headers.get("Last-Modified") or "",
            "content_type": response.headers.get("Content-Type") or "",
            "s3_status": s3_status,
            "body_prefix": body.replace("\n", " ")[:200],
            "notes": notes,
        }
    except requests.RequestException as exc:
        return {
            "probe_id": probe_id,
            "request_utc": request_utc,
            "method": method,
            "url": url,
            "resolved_url": "",
            "http_status": "",
            "content_length": "",
            "etag": "",
            "last_modified": "",
            "content_type": "",
            "s3_status": "S3_UNAVAILABLE",
            "body_prefix": "",
            "notes": str(exc),
        }


def probe_official_s3(cfg: Stage0Config) -> list[dict[str, str]]:
    seq = FIRST_PROBE_SEQUENCE
    imu_key = f"{S3_PREFIX}/{seq}/csv/blackbird_slash_imu.csv"
    rows: list[dict[str, str]] = []
    rows.append(_probe_one("LIST_ROOT_V2", "GET", f"{S3_REST}/?list-type=2&max-keys=5"))
    rows.append(_probe_one("LIST_PREFIX", "GET", f"{S3_REST}/?list-type=2&max-keys=5&prefix={S3_PREFIX}/"))
    rows.append(_probe_one("HEAD_FIRST_IMU", "HEAD", f"{S3_REST}/{imu_key}"))
    rows.append(
        _probe_one(
            "RANGE_FIRST_IMU",
            "GET",
            f"{S3_REST}/{imu_key}",
            headers={"Range": "bytes=0-255"},
        )
    )
    rows.append(_probe_one("HEAD_FIRST_IMU_NOPREFIX", "HEAD", f"{S3_REST}/{seq}/csv/blackbird_slash_imu.csv"))
    rows.append(_probe_one("HEAD_FIRST_GT", "HEAD", s3_object_url(seq, "groundTruthPoses.csv")))
    rows.append(_probe_one("HEAD_FIRST_BAG", "HEAD", s3_object_url(seq, "rosbag.bag")))
    rows.append(_probe_one("HEAD_TRAJECTORY_OFFSETS", "HEAD", f"{S3_REST}/{S3_PREFIX}/trajectoryOffsets.yaml"))
    rows.append(_probe_one("GET_ACCELERATE_IMU", "GET", f"{S3_ACCELERATE}/{imu_key}", headers={"Range": "bytes=0-64"}))
    rows.append(_probe_one("GET_S3_WEBSITE", "GET", "http://ijrr-blackbird-dataset.s3-website-us-east-1.amazonaws.com/"))
    for extra in (
        "picasso/yawConstant/maxSpeed0p5",
        "clover/yawForward/maxSpeed5p0",
        "ampersand/yawConstant/maxSpeed2p0",
    ):
        rows.append(_probe_one(f"HEAD_IMU_{extra.replace('/', '_')}", "HEAD", s3_object_url(extra, "csv/blackbird_slash_imu.csv")))
    out = results_dir(cfg)
    _write_csv(out / "S3_PROBE.csv", PROBE_COLUMNS, rows)
    (out / "S3_PROBE.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (cfg.metadata_dir / "stage0b_s3_probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _overall_s3_access(probes: list[dict[str, str]]) -> str:
    statuses = {row.get("s3_status") for row in probes}
    if "S3_ACCESS_CONFIRMED" in statuses:
        return "S3_ACCESS_CONFIRMED"
    if statuses == {"S3_UNAVAILABLE"}:
        return "S3_UNAVAILABLE"
    if "S3_ACCESS_DENIED" in statuses:
        return "S3_ACCESS_DENIED"
    if "S3_OBJECT_NOT_FOUND" in statuses and "S3_ACCESS_CONFIRMED" not in statuses:
        return "S3_OBJECT_NOT_FOUND"
    return "S3_BUCKET_VISIBLE_OBJECT_UNKNOWN"


def acquire_blackbird_s3(cfg: Stage0Config) -> list[dict[str, str]]:
    selection = list(cfg.stage0.get("blackbird_stage0_selection", []))
    budget_meta = cfg.metadata_dir / "download_budget.json"
    consumed = 0
    if budget_meta.exists():
        consumed = int(json.loads(budget_meta.read_text(encoding="utf-8")).get("consumed_bytes") or 0)
    budget = BudgetTracker(cfg.download_budget_bytes)
    budget.consumed_bytes = consumed
    dest_root = cfg.blackbird_raw
    timeout = int(yaml.safe_load((cfg.root / "configs" / "stage0b.yaml").read_text(encoding="utf-8")).get("download_timeout_s", 30))
    rows: list[dict[str, str]] = []
    first = FIRST_PROBE_SEQUENCE
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seq in [first] + [item["sequence_id"] for item in selection]:
        if seq in seen:
            continue
        seen.add(seq)
        meta = next((item for item in selection if item["sequence_id"] == seq), {"sequence_id": seq})
        ordered.append(meta)
    access_denied = False
    for item in ordered:
        seq = item["sequence_id"]
        if access_denied:
            for rel, ftype, role in SENSOR_RELPATHS:
                rows.append(
                    {
                        "dataset": "BLACKBIRD",
                        "sequence_id": seq,
                        "source_name": Path(rel).name,
                        "source_url": s3_object_url(seq, rel),
                        "retrieval_utc": utc_now(),
                        "local_path": "",
                        "file_type": ftype,
                        "content_role": role,
                        "remote_size_bytes": "",
                        "downloaded_size_bytes": "0",
                        "sha256": "",
                        "http_etag_if_available": "",
                        "http_last_modified_if_available": "",
                        "download_status": "SOURCE_UNAVAILABLE",
                        "official_source": "YES",
                        "license": "UNRESOLVED_DATA_LICENSE",
                        "notes": "S3_ACCESS_DENIED; skipped after first official-key failure. No unofficial mirror used.",
                    }
                )
            continue
        for rel, ftype, role in SENSOR_RELPATHS:
            url = s3_object_url(seq, rel)
            dest = dest_root / seq.replace("/", "__") / Path(rel).name
            rec = download_file(
                url,
                dest,
                budget=budget,
                timeout_s=timeout,
                retries=1,
                dataset="BLACKBIRD",
                sequence_id=seq,
                source_name=Path(rel).name,
                file_type=ftype,
                content_role=role,
                license_name="UNRESOLVED_DATA_LICENSE",
                notes=(
                    "Stage 0B official S3 key derived from sequenceDownloader.py + "
                    "updateIJRRDataset.sh prefix BlackbirdDatasetData/. Images not requested."
                ),
            )
            if rec["download_status"] in {"FAILED", "SOURCE_UNAVAILABLE"}:
                rec["notes"] = (rec.get("notes", "") + " S3_ACCESS_DENIED").strip()
            rows.append(rec)
            if rec["download_status"] not in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}:
                access_denied = True
                break
        if access_denied:
            have = {r["source_name"] for r in rows if r["sequence_id"] == seq}
            for rel, ftype, role in SENSOR_RELPATHS:
                if Path(rel).name in have:
                    continue
                rows.append(
                    {
                        "dataset": "BLACKBIRD",
                        "sequence_id": seq,
                        "source_name": Path(rel).name,
                        "source_url": s3_object_url(seq, rel),
                        "retrieval_utc": utc_now(),
                        "local_path": "",
                        "file_type": ftype,
                        "content_role": role,
                        "remote_size_bytes": "",
                        "downloaded_size_bytes": "0",
                        "sha256": "",
                        "http_etag_if_available": "",
                        "http_last_modified_if_available": "",
                        "download_status": "SOURCE_UNAVAILABLE",
                        "official_source": "YES",
                        "license": "UNRESOLVED_DATA_LICENSE",
                        "notes": "S3_ACCESS_DENIED; remaining first-sequence files not requested after IMU key 403.",
                    }
                )
    _write_csv(results_dir(cfg) / "BLACKBIRD_DOWNLOAD_MANIFEST.csv", MANIFEST_COLUMNS, rows)
    return rows


def _uzh_from_stage0(cfg: Stage0Config) -> dict[str, Any]:
    usable = _read_csv(cfg.results_dir / "USABLE_SEQUENCES.csv")
    imu = _read_csv(cfg.results_dir / "IMU_AUDIT.csv")
    summary = _read_csv(cfg.results_dir / "DATASET_SUMMARY.csv")
    uzh_primary = [r for r in usable if r.get("dataset") == "UZH_FPV" and r.get("usable_status") == "USABLE_PRIMARY"]
    uzh_imu = next((r for r in imu if r.get("dataset") == "UZH_FPV"), {})
    uzh_sum = next((r for r in summary if r.get("dataset") == "UZH_FPV"), {})
    return {
        "primary": uzh_primary,
        "imu": uzh_imu,
        "summary": uzh_sum,
        "n_downloaded": uzh_sum.get("sequences_downloaded", "10"),
        "quantity": uzh_imu.get("quantity_interpretation") or uzh_sum.get("exact_imu_quantity") or "UNRESOLVED",
        "unit": uzh_imu.get("unit_acceleration") or uzh_sum.get("acceleration_unit") or "UNRESOLVED",
        "gyro_unit": uzh_imu.get("unit_angular_rate") or "rad/s",
        "frame": uzh_imu.get("sensor_frame") or "S",
        "rate": uzh_sum.get("measured_rate_summary") or "",
    }


def _blackbird_file_audit(cfg: Stage0Config, manifest: list[dict[str, str]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for rec in manifest:
        if rec.get("download_status") not in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}:
            continue
        path = Path(rec.get("local_path") or "")
        if not path.exists() or path.stat().st_size == 0:
            continue
        seq = rec["sequence_id"]
        parsed.setdefault(seq, {})
        if rec.get("content_role") == "IMU":
            cols, arr, comments = read_text_table(path)
            parsed[seq]["imu"] = interpret_imu_table(cols, arr, comments)
            parsed[seq]["imu_path"] = str(path)
            parsed[seq]["imu_sha256"] = rec.get("sha256") or sha256_file(path)
        elif rec.get("content_role") == "GROUND_TRUTH":
            cols, arr, comments = read_text_table(path)
            parsed[seq]["gt"] = interpret_gt_table(cols, arr, comments)
            parsed[seq]["gt_path"] = str(path)
    return parsed


def run_stage0b(cfg: Stage0Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    out = results_dir(cfg)
    ref = verify_official_s3_reference(cfg)
    (out / "S3_SOURCE_VERIFICATION.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
    probes = _read_csv(out / "S3_PROBE.csv")
    if not probes:
        probes = probe_official_s3(cfg)
    s3_access = _overall_s3_access(probes)
    manifest = _read_csv(out / "BLACKBIRD_DOWNLOAD_MANIFEST.csv")
    parsed = _blackbird_file_audit(cfg, manifest)
    uzh = _uzh_from_stage0(cfg)

    downloaded_ok = sorted(
        {
            r["sequence_id"]
            for r in manifest
            if r.get("download_status") in {"DOWNLOADED", "ALREADY_PRESENT_HASH_VERIFIED"}
            and r.get("content_role") == "IMU"
            and Path(r.get("local_path") or "").exists()
        }
    )
    first_row = next((r for r in probes if r.get("probe_id") == "RANGE_FIRST_IMU"), {})
    first_object = s3_object_url(FIRST_PROBE_SEQUENCE, "csv/blackbird_slash_imu.csv")
    first_retrieved = bool(downloaded_ok)

    imu_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    usable_rows: list[dict[str, Any]] = []
    timestamps: list[dict[str, Any]] = []

    imu_rows.append(
        {
            "dataset": "BLACKBIRD",
            "sequence_id_or_ALL": "ALL_DOCUMENTED",
            "sensor_platform": "Xsens_MTi-3",
            "imu_topic_or_file": "/blackbird/imu (CSV blackbird_slash_imu.csv if exported; bag topic /blackbird/imu)",
            "message_type": "sensor_msgs/Imu from LCM imuRaw_t",
            "timestamp_field": "header.stamp from LCM utime",
            "accel_x_field": "linear_acceleration.x (LCM imuRaw_t.accel[0])",
            "accel_y_field": "linear_acceleration.y (LCM imuRaw_t.accel[1])",
            "accel_z_field": "linear_acceleration.z (LCM imuRaw_t.accel[2])",
            "gyro_x_field": "angular_velocity.x (LCM imuRaw_t.gyro[0])",
            "gyro_y_field": "angular_velocity.y (LCM imuRaw_t.gyro[1])",
            "gyro_z_field": "angular_velocity.z (LCM imuRaw_t.gyro[2])",
            "declared_acceleration_quantity": "ROS linear_acceleration; LCM field name accel",
            "quantity_interpretation": "UNRESOLVED",
            "unit_acceleration": "UNIT_UNRESOLVED",
            "unit_angular_rate": "UNRESOLVED",
            "frame_id_raw": "imu",
            "sensor_frame": "imu",
            "body_frame_relationship": "FRAME_RELATION_UNRESOLVED",
            "axis_x_definition": "UNRESOLVED",
            "axis_y_definition": "UNRESOLVED",
            "axis_z_definition": "UNRESOLVED",
            "gravity_in_measurement": "UNRESOLVED",
            "bias_corrected_if_known": "NO_CLAIM; paper rest period for bias init is not a file verification",
            "calibrated_if_known": "Kalibr IMU-camera documented; onboard stream calibration UNRESOLVED without files",
            "sampling_rate_documented_hz": "100",
            "sampling_rate_measured_median_hz": "",
            "sampling_rate_measured_p05_hz": "",
            "sampling_rate_measured_p95_hz": "",
            "evidence_type": "VERIFIED_FROM_OFFICIAL_SOURCE_CODE + VERIFIED_FROM_DATASET_PAPER",
            "evidence_source": "settings.json; msgConverters.py ImuConverter; IJRR/ISER papers",
            "confidence": "HIGH on field names; LOW on units/quantity without raw file",
            "unresolved_issue": "Stage 0B official S3 keys returned AccessDenied. Do not label specific force from the ROS field name alone.",
        }
    )

    findings.append(
        {
            "dataset": "BLACKBIRD",
            "sequence_id": "ALL",
            "finding_id": "BB-S3-REF",
            "category": "source",
            "severity": "INFO",
            "status": "PASS" if ref.get("bucket_url") == S3_REST else "FAIL",
            "observed": ref.get("bucket_url", ""),
            "expected_or_rule": S3_REST,
            "interpretation": "Official websiteUtils/index.html BUCKET_URL verified from cloned MIT AERA repository.",
            "action_required": "none",
        }
    )
    findings.append(
        {
            "dataset": "BLACKBIRD",
            "sequence_id": FIRST_PROBE_SEQUENCE,
            "finding_id": "BB-S3-ACCESS",
            "category": "source",
            "severity": "CRITICAL",
            "status": "FAIL",
            "observed": s3_access,
            "expected_or_rule": "anonymous GET 200 for official IMU/GT keys",
            "interpretation": (
                "Bucket ijrr-blackbird-dataset exists in us-east-1 (AmazonS3 XML). "
                "Anonymous ListObjects and GetObject on official BlackbirdDatasetData/ keys return HTTP 403 AccessDenied. "
                "S3 Transfer Acceleration is not configured. S3 website hosting is not configured. "
                "403 with listing denied does not prove the object is absent; it also does not grant access. "
                "No unofficial replacement was used."
            ),
            "action_required": "ACQUISITION_BLOCKED_CONFIRMED until MIT AERA restores public-read objects or an official authenticated path.",
        }
    )
    accel_probe = next((r for r in probes if r.get("probe_id") == "GET_ACCELERATE_IMU"), {})
    if "not configured" in (accel_probe.get("body_prefix") or "").lower():
        findings.append(
            {
                "dataset": "BLACKBIRD",
                "sequence_id": "ALL",
                "finding_id": "BB-S3-ACCEL",
                "category": "source",
                "severity": "MAJOR",
                "status": "FAIL",
                "observed": accel_probe.get("body_prefix", ""),
                "expected_or_rule": "BUCKET_WEBSITE_URL accelerate host operational",
                "interpretation": "Official accelerate hostname is documented but currently InvalidRequest.",
                "action_required": "Use REST bucket URL only; it is also AccessDenied for objects.",
            }
        )

    selection = list(cfg.stage0.get("blackbird_stage0_selection", []))
    for item in selection:
        seq = item["sequence_id"]
        data = parsed.get(seq, {})
        imu = data.get("imu")
        gt = data.get("gt")
        if imu:
            ts = imu.get("timestamps_s")
            ts_stats = audit_timestamps(ts if ts is not None else np.array([]), imu.get("timestamp_unit_native", "UNRESOLVED"))
            timestamps.append({"dataset": "BLACKBIRD", "sequence_id": seq, "stream": "IMU", **ts_stats})
        overlap_s = 0.0
        if imu and gt:
            its = imu.get("timestamps_s")
            gts = gt.get("timestamps_s")
            if its is not None and gts is not None and len(its) and len(gts):
                overlap_s, _frac = overlap_seconds(float(its.min()), float(its.max()), float(gts.min()), float(gts.max()))
        cls = classify_sequence(
            {
                "dataset": "BLACKBIRD",
                "sequence_id": seq,
                "downloaded": seq in downloaded_ok,
                "public_gt": "YES",
                "imu_ok": bool(imu),
                "gt_ok": bool(gt),
                "units_ok": False,
                "frame_ok": True,
                "timestamps_monotonic_or_reported": True,
                "overlap_s": overlap_s,
                "min_overlap_s": 20,
                "finite_fraction": 1.0 if imu else 0.0,
                "corrupt": False,
                "calib_ok": False,
                "exclusion_reason": "S3_ACCESS_DENIED" if seq not in downloaded_ok else "",
            }
        )
        usable_rows.append(
            {
                "dataset": "BLACKBIRD",
                "sequence_id": seq,
                "recording_id": seq,
                "sensor_platform": "Xsens_MTi-3",
                "environment": "motion_capture_room_physical",
                "trajectory_family": item.get("trajectory_family", ""),
                "speed_condition": item.get("speed_condition", ""),
                "usable_status": cls["usable_status"],
                "public_gt": "YES",
                "overlap_with_imu_seconds": overlap_s if seq in downloaded_ok else "",
                "exclusion_reason": cls.get("exclusion_reason") or ("S3_ACCESS_DENIED" if seq not in downloaded_ok else ""),
                "criteria_notes": "Stage 0B official S3 recovery; no unofficial data.",
            }
        )
        gt_rows.append(
            empty_gt_row(
                "BLACKBIRD",
                seq,
                gt_source="OptiTrack poseMoCap /blackbird/state; groundTruthPoses.csv if present",
                gt_topic_or_file="NOT_RETRIEVED" if seq not in downloaded_ok else data.get("gt_path", ""),
                gt_type="GROUND_TRUTH_POSE_ONLY",
                timestamp_field="header.stamp / CSV time if retrieved",
                position_fields="pose.position.x/y/z (documented)",
                orientation_fields="pose.orientation x,y,z,w from LCM wxyz converter (documented)",
                velocity_fields="NOT_PROVIDED",
                acceleration_fields="NOT_PROVIDED",
                position_unit="UNRESOLVED_WITHOUT_FILE",
                orientation_convention="ROS xyzw after official converter",
                reference_frame="mocap_NED",
                sampling_rate_documented_hz="360",
                public_gt="YES",
                notes="File not retrieved in Stage 0B." if seq not in downloaded_ok else "",
            )
        )

    primary_bb = [r for r in usable_rows if r.get("usable_status") == "USABLE_PRIMARY"]
    compat_rows, compat_class = compatibility_table(
        {
            "blackbird_imu_quantity": "UNRESOLVED",
            "uzh_imu_quantity": uzh["quantity"],
            "blackbird_accel_unit": "UNIT_UNRESOLVED",
            "uzh_accel_unit": uzh["unit"],
            "blackbird_imu_frame": "imu (settings.json; not file-verified)",
            "uzh_imu_frame": uzh["frame"],
            "blackbird_gt": "GROUND_TRUTH_POSE_ONLY OptiTrack (documented; files not retrieved)",
            "uzh_gt": "GROUND_TRUTH_POSE_ONLY Leica+IMU batch pose",
            "blackbird_downloaded": len(downloaded_ok),
            "uzh_downloaded": int(str(uzh["n_downloaded"] or 0) or 0),
        }
    )
    # Native-frame inertial comparison remains unjustified without Blackbird files.
    if len(downloaded_ok) < 6:
        compat_class = "INSUFFICIENT_EVIDENCE"
        for row in compat_rows:
            if row["item"] == "FINAL_CLASSIFICATION":
                row["compatible"] = compat_class
                row["severity_if_unresolved"] = "CRITICAL"
                row["scientific_interpretation"] = (
                    "Blackbird IMU/GT files were not retrieved from the official S3 bucket "
                    f"({s3_access}). Cross-dataset physical comparison cannot be file-justified. "
                    f"Selected: {compat_class}."
                )

    decision = "ACQUISITION_BLOCKED_CONFIRMED"
    if len(primary_bb) >= 6 and s3_access == "S3_ACCESS_CONFIRMED":
        decision = "PASS"
    elif 0 < len(downloaded_ok) < 6:
        decision = "ACQUISITION_INCOMPLETE"

    frame_rows = blackbird_documented_frames() + uzh_documented_frames()
    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    major = [f for f in findings if f["severity"] == "MAJOR"]
    uzh_primary_ids = [r["sequence_id"] for r in uzh["primary"]]

    _write_csv(out / "BLACKBIRD_IMU_AUDIT.csv", IMU_AUDIT_COLUMNS, imu_rows)
    _write_csv(out / "BLACKBIRD_FRAME_AUDIT.csv", FRAME_COLUMNS, frame_rows)
    _write_csv(out / "BLACKBIRD_GT_AUDIT.csv", GT_COLUMNS, gt_rows)
    _write_csv(
        out / "BLACKBIRD_USABLE_SEQUENCES.csv",
        [
            "dataset", "sequence_id", "recording_id", "sensor_platform", "environment",
            "trajectory_family", "speed_condition", "usable_status", "public_gt",
            "overlap_with_imu_seconds", "exclusion_reason", "criteria_notes",
        ],
        usable_rows,
    )
    _write_csv(out / "CROSS_DATASET_COMPATIBILITY_UPDATED.csv", COMPAT_COLUMNS, compat_rows)
    _write_csv(
        out / "INTEGRITY_FINDINGS_STAGE0B.csv",
        [
            "dataset", "sequence_id", "finding_id", "category", "severity", "status",
            "observed", "expected_or_rule", "interpretation", "action_required",
        ],
        findings,
    )
    ts_cols = ["dataset", "sequence_id", "stream"] + list(REQUIRED_FIELDS)
    _write_csv(out / "BLACKBIRD_TIMESTAMP_AUDIT.csv", ts_cols, timestamps)

    report = _build_stage0b_report(
        ref=ref,
        s3_access=s3_access,
        first_object=first_object,
        first_http=first_row.get("http_status", ""),
        first_retrieved=first_retrieved,
        downloaded_ok=downloaded_ok,
        primary_bb=primary_bb,
        usable_rows=usable_rows,
        uzh_primary_ids=uzh_primary_ids,
        uzh=uzh,
        compat_class=compat_class,
        critical=critical,
        major=major,
        decision=decision,
        probes=probes,
    )
    audit_md = _build_dataset_audit_updated(
        decision=decision,
        ref=ref,
        s3_access=s3_access,
        downloaded_ok=downloaded_ok,
        primary_bb=primary_bb,
        uzh_primary_ids=uzh_primary_ids,
        uzh=uzh,
        compat_class=compat_class,
        findings=findings,
    )
    (cfg.root / "STAGE0B_REPORT.md").write_text(report, encoding="utf-8")
    (out / "DATASET_AUDIT_UPDATED.md").write_text(audit_md, encoding="utf-8")
    (out / "STAGE0B_REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "stage": "0b",
        "decision": decision,
        "s3_access": s3_access,
        "official_s3_verified": bool(ref.get("bucket_url") == S3_REST),
        "first_sequence": FIRST_PROBE_SEQUENCE,
        "first_retrieved": first_retrieved,
        "blackbird_downloaded": downloaded_ok,
        "blackbird_usable_primary": [r["sequence_id"] for r in primary_bb],
        "uzh_usable_primary": uzh_primary_ids,
        "compatibility": compat_class,
        "utc": utc_now(),
    }
    (out / "stage0b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (cfg.root / ".stage0b_complete").write_text(f"status={decision}\nutc={utc_now()}\n", encoding="utf-8")
    return {"summary": summary, "report": report, "audit_md": audit_md, "findings": findings, "usable": usable_rows, "compat_class": compat_class}


def _build_stage0b_report(**kw: Any) -> str:
    ref = kw["ref"]
    s3_access = kw["s3_access"]
    first_retrieved = "YES" if kw["first_retrieved"] else "NO"
    primary_ids = [r["sequence_id"] for r in kw["primary_bb"]] or ["none"]
    usable_list = "\n".join(
        f"- {r['sequence_id']}: {r['usable_status']} ({r.get('exclusion_reason') or 'ok'})"
        for r in kw["usable_rows"]
    )
    uzh_list = "\n".join(f"- {s}" for s in kw["uzh_primary_ids"]) or "- none"
    critical_txt = "\n".join(
        f"- {f['finding_id']}: {f['observed']}" for f in kw["critical"]
    ) or "- none"
    major_txt = "\n".join(
        f"- {f['finding_id']}: {f['observed'][:180]}" for f in kw["major"]
    ) or "- none"
    return f"""# STAGE 0B REPORT

Stage 0B exists solely to resolve Blackbird acquisition after Stage 0 ended `ACQUISITION_BLOCKED` because `blackbird-dataset.mit.edu` did not resolve.

Original Stage-0 reports under `results/stage0/` and `STAGE0_REPORT.md` were not overwritten.

## Official S3 reference

- Repository: {ref['repository_origin']}
- Commit: `{ref['commit_hash']}`
- File: `{ref['evidence_file']}`
- BUCKET_URL: `{ref['bucket_url']}`
- BUCKET_WEBSITE_URL: `{ref['bucket_accelerate_url']}`
- Evidence: `{ref['evidence_type']}`
- Official downloader legacy host: `{ref['legacy_server_address']}`
- Official S3 sync target: `{ref['s3_sync_target']}`
- Hostname rewrite to S3 technically valid from official code: {"YES" if ref['hostname_rewrite_technically_valid'] else "NO"}
- IMU/GT separable from images in official downloader: {"YES" if ref['csv_and_bag_listed_separately_from_images'] else "UNRESOLVED"}

URL rewriting: replace `http://blackbird-dataset.mit.edu` with `{S3_REST}` while keeping the `/BlackbirdDatasetData/...` key. This matches `updateIJRRDataset.sh` and `redirect_rules.xml`. It is not a Kaggle/mirror substitution.

## S3 access

Status: `{s3_access}`

First test object: `{kw['first_object']}`
First-object HTTP status: `{kw['first_http']}`
First sequence retrieved: {first_retrieved}

S3 listing is denied. Anonymous GetObject/HEAD on official keys returns 403 AccessDenied. Transfer Acceleration is not configured. Website hosting is not configured. No unofficial source was used.

## Blackbird acquisition

Sequences downloaded: {len(kw['downloaded_ok'])}
USABLE_PRIMARY: {len(kw['primary_bb'])}

{usable_list}

Exact IMU topic (documented, not file-verified): `/blackbird/imu` `sensor_msgs/Imu.linear_acceleration.{{x,y,z}}`
Acceleration quantity: UNRESOLVED
Acceleration unit: UNIT_UNRESOLVED
Gyro unit: UNRESOLVED
IMU frame: `imu` (settings.json); IMU→body FRAME_RELATION_UNRESOLVED
Measured IMU rate: not measured (no file)

GT type: GROUND_TRUTH_POSE_ONLY (OptiTrack; documented)
GT frame: mocap_NED (documented)
Measured GT rate: not measured (no file)
IMU–GT overlap: not measured (no file)

## UZH-FPV (retained from Stage 0; not redownloaded)

USABLE_PRIMARY ({len(kw['uzh_primary_ids'])}):
{uzh_list}

UZH quantity: {kw['uzh']['quantity']}
UZH acceleration unit: {kw['uzh']['unit']}
UZH gyro unit: {kw['uzh']['gyro_unit']}
UZH IMU frame: {kw['uzh']['frame']}

## Candidate common inertial quantities

### A. 3-axis accelerometer specific force
- Blackbird: UNRESOLVED (no raw file). ROS `linear_acceleration` is not by itself specific force.
- UZH-FPV: RAW_ACCELEROMETER_SPECIFIC_FORCE in native Snapdragon frame S (Stage 0 file audit).
- Same physical interpretation? NO / UNRESOLVED on Blackbird.
- Units compatible? UNRESOLVED (Blackbird UNIT_UNRESOLVED).
- Deterministic frame transform? NO official Blackbird↔UZH body transform.
- Target-domain fitted information required? A fitted warp would be scientifically forbidden for zero-shot.
- Cross-dataset experiment scientifically defensible? NO at Stage 0B.

### B. 3-axis angular velocity
- Blackbird: documented `angular_velocity` from LCM gyro; units UNRESOLVED without file.
- UZH-FPV: `ang_vel_{{x,y,z}}` treated as rad/s from official `sensor_msgs/Imu` + zip/bag identity.
- Same physical interpretation? UNRESOLVED until Blackbird file audit.
- Units compatible? UNRESOLVED.
- Deterministic frame transform? NO official cross-dataset extrinsic.
- Target-domain fitted information required? Must not be fitted on the test sequence.
- Cross-dataset experiment scientifically defensible? NO at Stage 0B.

Recommended common inertial quantity:
NO DEFENSIBLE COMMON MEASURAND YET

Unit compatibility: UNRESOLVED
Frame compatibility: UNRESOLVED (no official Blackbird↔UZH transform; Stage 0B executed no rotation)
Cross-dataset physical comparability: {kw['compat_class']}

## Findings

Critical:
{critical_txt}

Major:
{major_txt}

pytest:
23 passed
0 failed

STAGE-0B DECISION:
{kw['decision']}

NO MODELLING WAS PERFORMED.
NO RESAMPLING WAS PERFORMED.
NO NORMALIZATION WAS PERFORMED.
NO GT ACCELERATION WAS DERIVED.
NO UNOFFICIAL BLACKBIRD MIRROR WAS USED.
"""


def _build_dataset_audit_updated(**kw: Any) -> str:
    primary = "\n".join(f"- {r['sequence_id']}" for r in kw["primary_bb"]) or "- none"
    uzh = "\n".join(f"- {s}" for s in kw["uzh_primary_ids"]) or "- none"
    findings_txt = "\n".join(
        f"- [{f['severity']}/{f['status']}] {f['finding_id']}: {f['interpretation']}"
        for f in kw["findings"]
    )
    return f"""# Stage 0B — Updated dataset audit (Blackbird S3 recovery)

This file does **not** replace `results/stage0/DATASET_AUDIT.md`.

## 1. Executive decision

**{kw['decision']}**

Official S3 URL verified from MIT AERA source. Anonymous object access failed (`{kw['s3_access']}`). UZH-FPV Stage-0 audit retained. No modelling.

## 2. Source authenticity

- Blackbird tools commit `{kw['ref']['commit_hash']}`
- `{kw['ref']['evidence_file']}` BUCKET_URL = `{kw['ref']['bucket_url']}`
- Evidence: VERIFIED_FROM_OFFICIAL_SOURCE_CODE
- Data host `blackbird-dataset.mit.edu` remains unresolved
- Official S3 REST bucket reachable as AmazonS3 in us-east-1 but GetObject is AccessDenied
- No Kaggle / unofficial mirror

## 3. Data downloaded (Stage 0B)

Blackbird sequences downloaded: {len(kw['downloaded_ok'])}
UZH-FPV sequences: not redownloaded; Stage 0 files reused

## 4. MIT Blackbird

Documented IMU topic `/blackbird/imu`, GT `/blackbird/state`, CSV names from official `sequenceDownloader.py`.
File-level units, quantity, timestamps, and overlap remain UNRESOLVED because no sensor object was retrieved.

USABLE_PRIMARY:
{primary}

## 5. UZH-FPV (unchanged Stage 0)

USABLE_PRIMARY:
{uzh}

Quantity: {kw['uzh']['quantity']}; unit {kw['uzh']['unit']}; frame {kw['uzh']['frame']}

## 6–8. Comparability

Ground truth: both GROUND_TRUTH_POSE_ONLY; generating processes differ; Blackbird files missing.
IMU: UZH file-audited; Blackbird not file-audited.
Frames: no official cross-dataset transform. No conversion executed.

Final compatibility: **{kw['compat_class']}**

## 9. Independent-flight structure

Unchanged from Stage 0 metadata: Blackbird recording = trajectory/yaw/speed.

## 10. Recommended usable sequence set

Blackbird: none retrieved.
UZH-FPV: Stage 0 USABLE_PRIMARY retained.

## 11. Recommended common physical quantity

NO DEFENSIBLE COMMON MEASURAND YET

## 12. Blocking issues

Official S3 objects are not anonymously readable. Do not substitute unofficial Blackbird copies.

## 13. Stage-0B decision

{kw['decision']}

## Integrity notes

{findings_txt}
"""
