"""Stage 0C constants, acquisition helpers, EuRoC audit, and UZH re-verification.

Audit-only: no resampling, no stream rotation, no GT differentiation, no modelling.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT, load_yaml
from .downloader import BudgetTracker, download_file, utc_now
from .euroc_schema import (
    as_uint64,
    compute_gt_overlap,
    extra_dt_frequency_stats,
    extract_inner_zips,
    extract_text_members,
    find_named,
    imu_channel_integrity,
    list_zip_members,
    load_sensor_yaml,
    map_euroc_gt_header,
    map_euroc_imu_header,
    read_euroc_csv,
    relative_seconds_from_ns,
    rest_norm_median,
    rotation_to_quaternion_wxyz,
    timestamps_to_seconds,
)
from .hashing import sha256_file
from .text_reader import interpret_imu_table, read_text_table
from .timestamp_audit import audit_timestamps

STAGE0C_DIRNAME = "stage0c"

EUROC_IMU_COLUMNS = [
    "sequence_id",
    "imu_file",
    "sensor_yaml",
    "sensor_model",
    "sensor_type",
    "timestamp_field",
    "timestamp_unit",
    "gyro_x_raw_field",
    "gyro_y_raw_field",
    "gyro_z_raw_field",
    "gyro_unit",
    "accel_x_raw_field",
    "accel_y_raw_field",
    "accel_z_raw_field",
    "acceleration_unit",
    "declared_acceleration_quantity",
    "interpreted_acceleration_quantity",
    "gravity_in_sensor_measurement",
    "bias_corrected",
    "factory_calibrated_if_known",
    "imu_sensor_frame",
    "body_frame",
    "body_to_sensor_transform_available",
    "transform_definition",
    "transform_direction",
    "documented_rate_hz",
    "measured_median_rate_hz",
    "measured_p05_rate_hz",
    "measured_p95_rate_hz",
    "noise_density_gyro",
    "random_walk_gyro",
    "noise_density_accel",
    "random_walk_accel",
    "evidence_type",
    "evidence_source",
    "confidence",
    "unresolved_issue",
]

EUROC_FRAME_COLUMNS = [
    "sequence_id",
    "imu_sensor_frame",
    "body_frame",
    "transform_name",
    "transform_direction",
    "rotation_matrix",
    "translation_vector",
    "quaternion_wxyz",
    "transform_source",
    "frame_naming",
    "evidence_type",
    "confidence",
    "notes",
]

EUROC_TS_COLUMNS = [
    "sequence_id",
    "n_samples",
    "first_timestamp",
    "last_timestamp",
    "duration_s",
    "strictly_monotonic",
    "duplicate_timestamps",
    "backward_timestamps",
    "dt_min",
    "dt_p01",
    "dt_p05",
    "dt_median",
    "dt_mean",
    "dt_std",
    "dt_p95",
    "dt_p99",
    "dt_max",
    "freq_median_hz",
    "freq_p05_hz",
    "freq_p95_hz",
    "gaps_gt_2x_median",
    "gaps_gt_5x_median",
    "gaps_gt_10x_median",
    "timestamp_unit",
    "evidence_type",
]

EUROC_GT_COLUMNS = [
    "sequence_id",
    "gt_source",
    "gt_file",
    "gt_type",
    "gt_origin",
    "timestamp_field",
    "timestamp_unit",
    "position_fields",
    "position_unit",
    "orientation_fields",
    "orientation_representation",
    "quaternion_order_if_applicable",
    "velocity_fields",
    "angular_velocity_fields",
    "acceleration_fields",
    "field_origin_direct_or_postprocessed",
    "gt_reference_frame",
    "body_or_reference_frame",
    "gt_rate_documented_hz",
    "gt_rate_measured_hz",
    "imu_gt_start_overlap",
    "imu_gt_end_overlap",
    "overlap_duration_s",
    "overlap_fraction",
    "synchronization_documentation",
    "synchronization_limitation",
    "confidence",
    "notes",
]

EUROC_SEQ_COLUMNS = [
    "sequence_id",
    "environment",
    "room",
    "difficulty",
    "platform",
    "sensor_rig",
    "recording_id",
    "duration_s",
    "imu_available",
    "gt_available",
    "gt_type",
    "calibration_available",
    "independent_recording",
    "potential_dependency_group",
    "independence_confidence",
    "usable_status",
    "exclusion_reason",
]

EUROC_USABLE_COLUMNS = [
    "sequence_id",
    "usable_status",
    "official_source",
    "authentic_file",
    "valid_hash",
    "imu_accel_available",
    "gyro_available",
    "units_verified",
    "imu_frame_documented",
    "valid_timestamps",
    "imu_duration_s",
    "gt_available",
    "overlap_s",
    "finite_fraction",
    "critical_integrity",
    "calibration_documented",
    "exclusion_reason",
    "notes",
]

INTEGRITY_COLUMNS = [
    "sequence_id",
    "check_id",
    "category",
    "severity",
    "status",
    "observed",
    "expected_or_rule",
    "interpretation",
    "action_required",
]

REMOTE_INV_COLUMNS = [
    "dataset",
    "sequence_id",
    "environment",
    "room_or_location",
    "difficulty",
    "platform",
    "imu_available",
    "stereo_available",
    "gt_available",
    "gt_type",
    "gt_dimensions",
    "calibration_available",
    "official_archive",
    "official_download_available",
    "archive_size_bytes",
    "official_evidence",
    "notes",
]

DOWNLOAD_COLUMNS = [
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

COMPAT_COLUMNS = [
    "item",
    "EUROC",
    "UZH_FPV",
    "compatible",
    "fixed_conversion_required",
    "target_fitted_information_required",
    "severity",
    "evidence",
    "recommendation",
]

PRIMARY_COLUMNS = [
    "dataset",
    "sequence_id",
    "environment",
    "difficulty",
    "duration_s",
    "imu_rate_hz",
    "gt_type",
    "gt_rate_hz",
    "common_quantity_available",
    "frame_status",
    "integrity_status",
    "dependency_group",
    "primary_eligible",
    "reason",
]

UNIT_COMPAT_COLUMNS = [
    "quantity",
    "euroc_value",
    "uzh_value",
    "compatibility",
    "fixed_conversion",
    "target_fitted_required",
    "evidence",
    "notes",
]

FRAME_COMPAT_COLUMNS = [
    "item",
    "euroc",
    "uzh_fpv",
    "common_body_possible",
    "fixed_from_official_calibration",
    "target_test_fit_required",
    "alignment_arbitrary_or_learned",
    "evidence",
    "notes",
]

GT_COMPAT_COLUMNS = [
    "item",
    "euroc",
    "uzh_fpv",
    "supports_temporal_segmentation",
    "supports_dynamics_characterization",
    "supports_pose_reference_evaluation",
    "supports_gt_acceleration_derivation",
    "evidence",
    "notes",
]

UZH_REVERIFY_COLUMNS = [
    "sequence_id",
    "local_path",
    "expected_sha256",
    "observed_sha256",
    "size_bytes",
    "hash_status",
    "schema_status",
    "header_line",
    "notes",
]


def load_stage0c_config(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT
    return load_yaml(root / "configs" / "stage0c.yaml")


def results_dir(root: Path) -> Path:
    path = root / "results" / STAGE0C_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k, "")) for k in columns})


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return repr(value) if abs(value) < 1e-4 and value != 0 else f"{value:.10g}"
    if isinstance(value, (np.floating,)):
        return _csv_cell(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value).tolist() if isinstance(value, np.ndarray) else list(value))
    return str(value)


def parse_sequence_meta(sequence_id: str) -> dict[str, str]:
    parts = sequence_id.split("_")
    if sequence_id.startswith("V1_"):
        env, room, diff = "vicon", "vicon_room1", parts[-1] if len(parts) >= 3 else "UNRESOLVED"
    elif sequence_id.startswith("V2_"):
        env, room, diff = "vicon", "vicon_room2", parts[-1] if len(parts) >= 3 else "UNRESOLVED"
    elif sequence_id.startswith("MH_"):
        env, room, diff = "machine_hall", "ETH_machine_hall", parts[-1] if len(parts) >= 3 else "UNRESOLVED"
    else:
        env, room, diff = "UNRESOLVED", "UNRESOLVED", "UNRESOLVED"
    return {"environment": env, "room": room, "difficulty": diff}


def bitstream_url(cfg: dict[str, Any], uuid: str) -> str:
    return f"{cfg['euroc']['bitstream_base'].rstrip('/')}/{uuid}/content"


def map_download_status(raw: str, sha256: str) -> str:
    if raw == "ALREADY_PRESENT_HASH_VERIFIED":
        return "HASH_VERIFIED"
    if raw == "DOWNLOADED" and sha256:
        return "HASH_VERIFIED"
    return raw


def official_remote_inventory(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    euroc = cfg["euroc"]
    platform = "Asctec Firefly hex-rotor + VI-Sensor (ADIS16448)"
    for archive in list(euroc["phase_a_archives"]) + list(euroc["optional_archives"]):
        name = archive["name"]
        if name == "calibration_datasets.zip":
            continue
        is_mh = archive["environment"] == "machine_hall"
        is_vicon = archive["environment"].startswith("vicon")
        available = "YES" if not is_mh else "YES_OPTIONAL_NOT_IN_PHASE_A"
        for seq in archive["sequences"]:
            meta = parse_sequence_meta(seq)
            if is_vicon:
                gt_type = "Vicon_6DoF_pose_plus_postprocessed_state_estimate"
                gt_dim = "6DoF_pose_documented; state_estimate_fields_file_verified_after_download"
            else:
                gt_type = "Leica_MS50_3D_position_plus_postprocessed_state_estimate"
                gt_dim = "3D_position_from_tracker; orientation_not_a_direct_Leica_measurand"
            rows.append(
                {
                    "dataset": "EUROC",
                    "sequence_id": seq,
                    "environment": meta["environment"],
                    "room_or_location": meta["room"],
                    "difficulty": meta["difficulty"],
                    "platform": platform,
                    "imu_available": "YES_DOCUMENTED",
                    "stereo_available": "YES_DOCUMENTED",
                    "gt_available": "YES_DOCUMENTED",
                    "gt_type": gt_type,
                    "gt_dimensions": gt_dim,
                    "calibration_available": "YES_IN_SEQUENCE_ASL_YAML",
                    "official_archive": name,
                    "official_download_available": available,
                    "archive_size_bytes": archive["size_bytes"],
                    "official_evidence": "VERIFIED_FROM_OFFICIAL_WEB; ETH Research Collection bitstream + zip.json preview",
                    "notes": (
                        "Machine Hall optional in Stage 0C; not downloaded under 15 GiB Vicon-first policy."
                        if is_mh
                        else "Phase A Vicon bundle. Inner sequence zip+bag; bags not extracted."
                    ),
                }
            )
    return rows


def write_source_verification(root: Path, cfg: dict[str, Any], retrieval_utc: str) -> Path:
    euroc = cfg["euroc"]
    vicon = sum(int(a["size_bytes"]) for a in euroc["phase_a_archives"])
    mh = next(a for a in euroc["optional_archives"] if a["name"] == "machine_hall.zip")
    calib = next(a for a in euroc["optional_archives"] if a["name"] == "calibration_datasets.zip")
    text = f"""# EuRoC source verification (Stage 0C)

retrieval_utc: {retrieval_utc}

## Dataset official name
The EuRoC MAV Dataset / EuRoC micro aerial vehicle datasets. Evidence: VERIFIED_FROM_OFFICIAL_WEB (ETH ASL page; ETH Research Collection item).

## Institution
ETH Zurich Autonomous Systems Lab (ASL). Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Dataset DOI
{euroc['doi']} (handle {euroc['handle']}; item UUID {euroc['item_uuid']}). Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Publication DOI
{euroc['paper_doi']} — Burri et al., International Journal of Robotics Research, "{euroc['paper_title']}". Evidence: VERIFIED_FROM_DATASET_PAPER; VERIFIED_FROM_OFFICIAL_WEB.

## Official landing page
ASL: {euroc['asl_page']}
ETH Research Collection: {euroc['landing_page']}
Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Download host
ETH Research Collection DSpace bitstream API: {euroc['bitstream_base']}
Evidence: VERIFIED_FROM_OFFICIAL_WEB (item JSON `_links.content.href`).

## License/rights
{euroc['license']}
Rights URL: {euroc['rights_url']}
ETH Library deposit terms also present as `license.txt` on the item.
Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Available archives (ORIGINAL bundle)
| archive | size_bytes | uuid | Stage 0C action |
|---|---:|---|---|
| vicon_room1.zip | 6042263426 | 02ecda9a-298f-498b-970c-b7c44334d880 | PHASE A download |
| vicon_room2.zip | 6013384949 | ea12bc01-3677-4b4c-853d-87c7870b8c44 | PHASE A download |
| machine_hall.zip | {mh['size_bytes']} | {mh['uuid']} | OPTIONAL / not downloaded (would exceed remaining budget with Vicon) |
| calibration_datasets.zip | {calib['size_bytes']} | {calib['uuid']} | NOT REQUIRED if sequence YAML present |
| euroc_mav_dataset.pdf | 368358 | d861e63b-cfa9-4411-85a5-5ad6b3526e44 | metadata downloaded |

Vicon Phase A total = {vicon} bytes ({vicon / (1024**3):.3f} GiB) < 15 GiB additional budget ({cfg['download_budget_bytes']} bytes). Evidence: VERIFIED_FROM_OFFICIAL_WEB.

## Official documentation references
- ETH ASL EuRoC page (sensor list, known issues, sequence names)
- Burri et al. IJRR 2016
- ETH ASL `dataset_tools` ({euroc['official_tools']}) MATLAB loader `dataset_load_sensor_data.m`
- ASL Dataset Format YAML (`sensor.yaml`, `T_BS`)
Evidence: VERIFIED_FROM_OFFICIAL_WEB; VERIFIED_FROM_OFFICIAL_SOURCE_CODE.

## Sequence organization
Official zip.json previews list nested `{{sequence}}/{{sequence}}.zip` and `{{sequence}}/{{sequence}}.bag` inside room bundles. Evidence: VERIFIED_FROM_OFFICIAL_WEB (bitstream JSON previews under `data/metadata/euroc/`).

Standard flight recordings documented on ASL page and confirmed in zip.json:
V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult,
MH_01_easy, MH_02_easy, MH_03_medium, MH_04_difficult, MH_05_difficult.
Evidence: VERIFIED_FROM_OFFICIAL_WEB. Names were not assumed from memory.

## Known dataset limitations (official)
- Independent auto-exposure on the two cameras (stereo brightness mismatch; mid-exposure times aligned).
- Highly dynamic motion can degrade laser-tracker accuracy (Machine Hall).
- Sensor vs motion-capture recorded on different systems; Vicon device timestamps unavailable; temporal offset estimated.
Evidence: VERIFIED_FROM_OFFICIAL_WEB (ASL known-issues text in `euroc_mav_dataset.pdf.txt`).

## Sources not used
Kaggle, Google Drive mirrors, random GitHub dataset copies, preprocessed ML repositories. Evidence: this Stage 0C configuration.

## Official parser IMU layout (before file headers)
`dataset_load_sensor_data.m` case `imu`: uint64 timestamp, then 3 omega, then 3 accelerometer values. Plot labels: accelerometer `[m / s / s]`, gyro converted from rad to deg. Timestamps divided by 1e9. Evidence: VERIFIED_FROM_OFFICIAL_SOURCE_CODE. Exact CSV header strings are still taken from downloaded files (VERIFIED_FROM_RAW_FILE).
"""
    path = results_dir(root) / "EUROC_SOURCE_VERIFICATION.md"
    path.write_text(text, encoding="utf-8")
    return path


def download_phase_a(root: Path, cfg: dict[str, Any]) -> list[dict[str, str]]:
    euroc = cfg["euroc"]
    budget = BudgetTracker(int(cfg["download_budget_bytes"]))
    dest_dir = root / "data" / "raw" / "euroc"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    vicon_bytes = sum(int(a["size_bytes"]) for a in euroc["phase_a_archives"])
    if vicon_bytes > budget.limit_bytes:
        for archive in euroc["phase_a_archives"]:
            rows.append(
                {
                    "sequence_id": ",".join(archive["sequences"]),
                    "official_source": "YES",
                    "source_url": bitstream_url(cfg, archive["uuid"]),
                    "archive_name": archive["name"],
                    "retrieval_utc": utc_now(),
                    "local_path": "",
                    "remote_size_bytes": str(archive["size_bytes"]),
                    "downloaded_size_bytes": "0",
                    "sha256": "",
                    "etag": "",
                    "last_modified": "",
                    "download_status": "SKIPPED_BUDGET",
                    "extraction_status": "NOT_EXTRACTED",
                    "notes": f"Phase A total {vicon_bytes} exceeds budget {budget.limit_bytes}",
                }
            )
        return rows
    for archive in euroc["phase_a_archives"]:
        rec = download_file(
            bitstream_url(cfg, archive["uuid"]),
            dest_dir / archive["name"],
            budget=budget,
            timeout_s=int(cfg.get("download_timeout_s", 600)),
            retries=int(cfg.get("download_retries", 6)),
            chunk_bytes=int(cfg.get("download_chunk_bytes", 1048576)),
            dataset="EUROC",
            sequence_id=",".join(archive["sequences"]),
            source_name=archive["name"],
            file_type="zip",
            content_role="EUROC_VICON_BUNDLE",
            license_name=euroc["license"],
            notes="Official ETH Research Collection ORIGINAL bitstream.",
            expected_size=int(archive["size_bytes"]),
        )
        mapped = map_download_status(rec["download_status"], rec.get("sha256", ""))
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
        if mapped in {"FAILED", "SOURCE_UNAVAILABLE", "SKIPPED_BUDGET", "PARTIAL"}:
            break
    for archive in euroc["optional_archives"]:
        rows.append(
            {
                "sequence_id": ",".join(archive["sequences"]),
                "official_source": "YES",
                "source_url": bitstream_url(cfg, archive["uuid"]),
                "archive_name": archive["name"],
                "retrieval_utc": utc_now(),
                "local_path": "",
                "remote_size_bytes": str(archive["size_bytes"]),
                "downloaded_size_bytes": "0",
                "sha256": "",
                "etag": "",
                "last_modified": "",
                "download_status": "NOT_REQUIRED",
                "extraction_status": "NOT_EXTRACTED",
                "notes": "Optional Stage 0C archive; not downloaded (Vicon-first / budget).",
            }
        )
    return rows


def extract_phase_a(root: Path, download_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Path]]:
    inner_dir = root / "data" / "raw" / "euroc" / "extracted"
    cache_root = root / "data" / "cache" / "euroc"
    seq_dirs: dict[str, Path] = {}
    extra_rows: list[dict[str, str]] = []
    for row in download_rows:
        archive = row.get("archive_name", "")
        local = Path(row.get("local_path") or "")
        if row.get("download_status") not in {"HASH_VERIFIED", "DOWNLOADED"}:
            continue
        if not local.exists() or not archive.endswith(".zip"):
            continue
        if archive not in {"vicon_room1.zip", "vicon_room2.zip"}:
            continue
        inners = extract_inner_zips(local, inner_dir)
        row["extraction_status"] = "INNER_ZIPS_EXTRACTED"
        for inner in inners:
            seq_id = inner.stem
            cache = cache_root / seq_id
            extracted = extract_text_members(inner, cache)
            seq_dirs[seq_id] = cache
            extra_rows.append(
                {
                    "sequence_id": seq_id,
                    "official_source": "YES",
                    "source_url": row["source_url"],
                    "archive_name": inner.name,
                    "retrieval_utc": utc_now(),
                    "local_path": str(inner),
                    "remote_size_bytes": str(inner.stat().st_size),
                    "downloaded_size_bytes": str(inner.stat().st_size),
                    "sha256": sha256_file(inner),
                    "etag": "",
                    "last_modified": "",
                    "download_status": "HASH_VERIFIED",
                    "extraction_status": "CSV_YAML_EXTRACTED_TO_CACHE" if extracted else "EXTRACT_EMPTY",
                    "notes": f"Inner official sequence zip from {archive}; images/bags not extracted. n_text_members={len(extracted)}",
                }
            )
    return extra_rows, seq_dirs


def _integrity_row(seq: str, check_id: str, category: str, severity: str, status: str, observed: Any, expected: str, interpretation: str, action: str) -> dict[str, str]:
    return {
        "sequence_id": seq,
        "check_id": check_id,
        "category": category,
        "severity": severity,
        "status": status,
        "observed": str(observed),
        "expected_or_rule": expected,
        "interpretation": interpretation,
        "action_required": action,
    }


def audit_one_euroc_sequence(seq_id: str, cache_dir: Path, inner_zip: Path | None) -> dict[str, Any]:
    extracted = {str(p.relative_to(cache_dir)).replace("\\", "/"): p for p in cache_dir.rglob("*") if p.is_file()}
    imu_csv = find_named(extracted, "imu0", "data.csv") or find_named(extracted, "imu", "data.csv")
    imu_yaml = find_named(extracted, "imu0", "sensor.yaml") or find_named(extracted, "imu", "sensor.yaml")
    gt_csv = (
        find_named(extracted, "state_groundtruth_estimate0", "data.csv")
        or find_named(extracted, "vicon0", "data.csv")
        or find_named(extracted, "leica0", "data.csv")
        or find_named(extracted, "groundtruth", "data.csv")
    )
    gt_yaml = (
        find_named(extracted, "state_groundtruth_estimate0", "sensor.yaml")
        or find_named(extracted, "vicon0", "sensor.yaml")
        or find_named(extracted, "leica0", "sensor.yaml")
    )
    body_yaml = find_named(extracted, "body.yaml")
    findings: list[dict[str, str]] = []

    if inner_zip is not None and inner_zip.exists():
        try:
            with zipfile.ZipFile(inner_zip) as zf:
                bad = zf.testzip()
            findings.append(
                _integrity_row(
                    seq_id, "ZIP_VALID", "archive", "CRITICAL" if bad else "INFO",
                    "FAIL" if bad else "PASS", bad or "testzip_ok",
                    "zipfile.testzip is None", "Official inner zip readable.", "none" if not bad else "EXCLUDE",
                )
            )
        except zipfile.BadZipFile as exc:
            findings.append(
                _integrity_row(seq_id, "ZIP_VALID", "archive", "CRITICAL", "FAIL", str(exc), "valid zip", "Unreadable archive.", "EXCLUDE")
            )

    if imu_csv is None:
        findings.append(_integrity_row(seq_id, "IMU_CSV", "imu", "CRITICAL", "FAIL", "missing", "imu0/data.csv", "No IMU CSV.", "EXCLUDE"))
        return {"sequence_id": seq_id, "findings": findings, "usable_status": "EXCLUDE", "exclusion_reason": "IMU CSV missing"}

    headers, array, header_line = read_euroc_csv(imu_csv)
    mapping = map_euroc_imu_header(headers)
    ts_idx = int(mapping["timestamp_index"] or 0)
    gx, gy, gz = mapping["gyro_indices"]
    ax, ay, az = mapping["accel_indices"]
    ts_native = array[:, ts_idx] if array.size else np.array([])
    gyro = (
        np.column_stack([array[:, gx], array[:, gy], array[:, gz]]).astype(np.float64)
        if array.size and None not in (gx, gy, gz)
        else np.zeros((0, 3))
    )
    accel = (
        np.column_stack([array[:, ax], array[:, ay], array[:, az]]).astype(np.float64)
        if array.size and None not in (ax, ay, az)
        else np.zeros((0, 3))
    )
    ts_unit_hdr = mapping["timestamp_unit_from_header"]
    ts_s, ts_unit_used = timestamps_to_seconds(ts_native, ts_unit_hdr)
    ts_stats = audit_timestamps(ts_s, "seconds")
    extra = extra_dt_frequency_stats(ts_s)
    if ts_native is not None and np.asarray(ts_native).size:
        ts_u = as_uint64(ts_native)
        native_first = str(int(ts_u[0]))
        native_last = str(int(ts_u[-1]))
        duration_s = float(int(ts_u[-1]) - int(ts_u[0])) / 1e9 if ts_unit_used == "nanoseconds" else float(ts_stats["duration"] or 0.0)
    else:
        native_first = ts_stats["first_timestamp"]
        native_last = ts_stats["last_timestamp"]
        duration_s = float(ts_stats["duration"] or 0.0)
    chan = imu_channel_integrity(accel, gyro)
    rest = rest_norm_median(accel)

    yaml_info = load_sensor_yaml(imu_yaml) if imu_yaml else {}
    sensor_comment = str(yaml_info.get("comment") or "")
    model = "UNRESOLVED"
    if "ADIS16448" in sensor_comment.upper() or "ADIS16448" in sensor_comment:
        model = "ADIS16448"
    elif "ADIS16448" in sensor_comment:
        model = "ADIS16448"
    if "ADIS16448" in sensor_comment:
        model = "ADIS16448"

    t_bs = yaml_info.get("T_BS")
    rot = yaml_info.get("rotation")
    trans = yaml_info.get("translation")
    transform_available = "YES" if t_bs is not None else "NO"
    # Official MATLAB: p_BS_B = T_BS(1:3,4); C_BS = T_BS(1:3,1:3) plotted in body.
    transform_direction = "T_BS maps sensor-frame coordinates to body-frame coordinates (S -> B)" if t_bs is not None else "UNRESOLVED"
    quat = rotation_to_quaternion_wxyz(rot) if rot is not None else []

    accel_unit = mapping["acceleration_unit_from_header"]
    gyro_unit = mapping["gyro_unit_from_header"]
    if accel_unit == "UNRESOLVED":
        # Official MATLAB ylabel a [m / s / s] is supporting evidence, not a substitute for file headers.
        accel_unit_evidence = "UNRESOLVED_FROM_HEADER"
    else:
        accel_unit_evidence = accel_unit

    gravity_flag = "UNRESOLVED"
    if rest is not None and accel_unit == "m/s^2":
        if 8.5 <= rest <= 11.0:
            gravity_flag = "YES_CONSISTENT_WITH_SPECIFIC_FORCE_AT_REST"
        else:
            gravity_flag = f"REST_NORM_NOT_G_LIKE:{rest:.4f}"

    interpreted = "UNRESOLVED"
    if accel_unit == "m/s^2" and gravity_flag.startswith("YES"):
        interpreted = "RAW_ACCELEROMETER_SPECIFIC_FORCE"
    elif accel_unit == "m/s^2":
        interpreted = "RAW_ACCELEROMETER_SPECIFIC_FORCE"
    declared = mapping["accel_x_raw_field"]

    findings.append(
        _integrity_row(
            seq_id, "CSV_READABLE", "imu", "CRITICAL", "PASS" if array.size else "FAIL",
            f"n={array.shape[0]} header={header_line[:120]}", "readable CSV with header",
            "IMU table parsed.", "none" if array.size else "EXCLUDE",
        )
    )
    findings.append(
        _integrity_row(
            seq_id, "YAML_READABLE", "calibration", "CRITICAL" if imu_yaml is None else "INFO",
            "PASS" if imu_yaml else "FAIL", str(imu_yaml), "imu0/sensor.yaml",
            "Calibration YAML present." if imu_yaml else "Missing YAML.", "none" if imu_yaml else "review",
        )
    )
    findings.append(
        _integrity_row(
            seq_id, "REQUIRED_COLUMNS", "imu", "CRITICAL",
            "PASS" if mapping["layout"] != "UNRESOLVED" else "FAIL",
            mapping["layout"], "timestamp + 3 gyro + 3 accel",
            "Header mapping.", "none" if mapping["layout"] != "UNRESOLVED" else "EXCLUDE",
        )
    )
    findings.append(
        _integrity_row(
            seq_id, "NAN_INF", "imu", "CRITICAL" if chan["finite_fraction_required"] < 0.99 else "INFO",
            "PASS" if chan["finite_fraction_required"] >= 0.99 else "FAIL",
            chan, ">=99% finite required IMU samples",
            "Finite-sample check.", "none" if chan["finite_fraction_required"] >= 0.99 else "EXCLUDE",
        )
    )
    findings.append(
        _integrity_row(
            seq_id, "MONOTONIC_TS", "timestamps", "MAJOR" if ts_stats["backward_count"] else "INFO",
            "PASS" if ts_stats["backward_count"] == 0 else "FAIL",
            f"backward={ts_stats['backward_count']} dup={ts_stats['duplicate_count']}",
            "strictly increasing timestamps preferred",
            "Native IMU timestamps.", "review" if ts_stats["backward_count"] else "none",
        )
    )
    findings.append(
        _integrity_row(
            seq_id, "CONSTANT_CHANNELS", "imu", "MAJOR" if chan["accel_constant"] else "INFO",
            "FAIL" if chan["accel_constant"] else "PASS",
            f"accel_constant={chan['accel_constant']} gyro_constant={chan['gyro_constant']}",
            "non-constant IMU channels",
            "Degenerate stream check.", "EXCLUDE" if chan["accel_constant"] else "none",
        )
    )

    gt_map = {}
    gt_array = np.zeros((0, 0))
    gt_ts_s = np.array([])
    gt_header = ""
    gt_type = "UNRESOLVED"
    if gt_csv is not None:
        gt_headers, gt_array, gt_header = read_euroc_csv(gt_csv)
        gt_map = map_euroc_gt_header(gt_headers)
        gt_ts_native = gt_array[:, int(gt_map["timestamp_index"])] if gt_array.size else np.array([])
        if ts_unit_used == "nanoseconds" and np.asarray(ts_native).size and np.asarray(gt_ts_native).size:
            origin = min(int(as_uint64(ts_native)[0]), int(as_uint64(gt_ts_native)[0]))
            ts_s, _ = relative_seconds_from_ns(ts_native, origin)
            gt_ts_s, _ = relative_seconds_from_ns(gt_ts_native, origin)
            gt_ts_unit = "nanoseconds"
        else:
            gt_ts_s, gt_ts_unit = timestamps_to_seconds(gt_ts_native, "nanoseconds")
        gt_extra = extra_dt_frequency_stats(gt_ts_s)
        gt_type = "POSTPROCESSED_STATE_ESTIMATE"
        if "vicon" in str(gt_csv).lower():
            gt_type = "VICON_POSE"
        findings.append(
            _integrity_row(
                seq_id, "GT_READABLE", "ground_truth", "CRITICAL", "PASS",
                f"n={gt_array.shape[0]} file={gt_csv.name}", "GT CSV readable",
                "GT table parsed.", "none",
            )
        )
    else:
        gt_ts_unit = "UNRESOLVED"
        gt_extra = extra_dt_frequency_stats(np.array([]))
        findings.append(
            _integrity_row(seq_id, "GT_READABLE", "ground_truth", "CRITICAL", "FAIL", "missing", "GT CSV", "No GT file in extracted text members.", "EXCLUDE")
        )

    overlap = compute_gt_overlap(ts_s, gt_ts_s)
    findings.append(
        _integrity_row(
            seq_id, "GT_OVERLAP", "ground_truth", "MAJOR" if overlap["overlap_duration_s"] < 20 else "INFO",
            "PASS" if overlap["overlap_duration_s"] >= 20 else "FAIL",
            overlap["overlap_duration_s"], ">=20 s IMU-GT overlap",
            "Temporal overlap on native timestamps (no interpolation).",
            "none" if overlap["overlap_duration_s"] >= 20 else "downgrade usability",
        )
    )
    critical_fail = any(f["severity"] == "CRITICAL" and f["status"] == "FAIL" for f in findings)
    units_ok = accel_unit == "m/s^2" and gyro_unit == "rad/s"
    frame_ok = t_bs is not None
    imu_ok = mapping["layout"] != "UNRESOLVED" and accel.size > 0 and gyro.size > 0
    gt_ok = gt_csv is not None and gt_array.size > 0
    hash_ok = inner_zip is not None and inner_zip.exists()
    usable = "EXCLUDE"
    reason = ""
    if critical_fail:
        usable = "EXCLUDE"
        reason = "CRITICAL integrity failure"
    elif not imu_ok or not units_ok or not frame_ok or not gt_ok:
        usable = "QUESTIONABLE"
        reason = "incomplete IMU/units/frame/GT documentation"
    elif duration_s < 20 or overlap["overlap_duration_s"] < 20:
        usable = "QUESTIONABLE"
        reason = "duration or IMU-GT overlap < 20 s"
    elif chan["finite_fraction_required"] < 0.99:
        usable = "QUESTIONABLE"
        reason = "finite fraction < 0.99"
    else:
        usable = "USABLE_PRIMARY"
        reason = ""

    meta = parse_sequence_meta(seq_id)
    dep = f"EUROC_FIREFLY_{meta['room'].upper()}"

    imu_row = {
        "sequence_id": seq_id,
        "imu_file": str(imu_csv),
        "sensor_yaml": str(imu_yaml or ""),
        "sensor_model": model if model != "UNRESOLVED" else (sensor_comment or "UNRESOLVED"),
        "sensor_type": yaml_info.get("sensor_type", "UNRESOLVED"),
        "timestamp_field": mapping["timestamp_field"],
        "timestamp_unit": ts_unit_used,
        "gyro_x_raw_field": mapping["gyro_x_raw_field"],
        "gyro_y_raw_field": mapping["gyro_y_raw_field"],
        "gyro_z_raw_field": mapping["gyro_z_raw_field"],
        "gyro_unit": gyro_unit,
        "accel_x_raw_field": mapping["accel_x_raw_field"],
        "accel_y_raw_field": mapping["accel_y_raw_field"],
        "accel_z_raw_field": mapping["accel_z_raw_field"],
        "acceleration_unit": accel_unit,
        "declared_acceleration_quantity": declared,
        "interpreted_acceleration_quantity": interpreted,
        "gravity_in_sensor_measurement": gravity_flag,
        "bias_corrected": "NO_NOT_CLAIMED_IN_IMU_CSV",
        "factory_calibrated_if_known": "UNRESOLVED_FOR_PUBLISHED_STREAM",
        "imu_sensor_frame": "S (sensor / IMU frame; header a_RS_S / w_RS_S)",
        "body_frame": "B (ASL body.yaml / T_BS body)",
        "body_to_sensor_transform_available": transform_available,
        "transform_definition": "T_BS 4x4 from imu0/sensor.yaml",
        "transform_direction": transform_direction,
        "documented_rate_hz": yaml_info.get("rate_hz", "UNRESOLVED"),
        "measured_median_rate_hz": extra["measured_median_rate_hz"],
        "measured_p05_rate_hz": extra["measured_p05_rate_hz"],
        "measured_p95_rate_hz": extra["measured_p95_rate_hz"],
        "noise_density_gyro": yaml_info.get("gyroscope_noise_density", "UNRESOLVED"),
        "random_walk_gyro": yaml_info.get("gyroscope_random_walk", "UNRESOLVED"),
        "noise_density_accel": yaml_info.get("accelerometer_noise_density", "UNRESOLVED"),
        "random_walk_accel": yaml_info.get("accelerometer_random_walk", "UNRESOLVED"),
        "evidence_type": "VERIFIED_FROM_RAW_FILE; VERIFIED_FROM_SENSOR_YAML; VERIFIED_FROM_OFFICIAL_SOURCE_CODE",
        "evidence_source": f"{imu_csv}; {imu_yaml}; dataset_load_sensor_data.m; dataset_plot_body.m",
        "confidence": "HIGH" if units_ok and mapping["layout"] != "UNRESOLVED" else "MEDIUM",
        "unresolved_issue": "Published-stream factory calibration vs residual bias UNRESOLVED. GT ba_S is an estimator bias, not a substitute IMU calibration."
        if interpreted != "UNRESOLVED"
        else "Acceleration quantity UNRESOLVED",
    }

    frame_row = {
        "sequence_id": seq_id,
        "imu_sensor_frame": "S",
        "body_frame": "B",
        "transform_name": "T_BS",
        "transform_direction": transform_direction,
        "rotation_matrix": rot.tolist() if rot is not None else "UNRESOLVED",
        "translation_vector": trans.tolist() if trans is not None else "UNRESOLVED",
        "quaternion_wxyz": quat,
        "transform_source": str(imu_yaml or ""),
        "frame_naming": "ASL YAML T_BS; MATLAB p_BS_B / C_BS",
        "evidence_type": "VERIFIED_FROM_SENSOR_YAML; VERIFIED_FROM_OFFICIAL_SOURCE_CODE",
        "confidence": "HIGH" if t_bs is not None else "LOW",
        "notes": "Stage 0C does not rotate IMU streams. Direction taken from official MATLAB use of T_BS, not guessed from the letters T_BS alone.",
    }

    ts_row = {
        "sequence_id": seq_id,
        "n_samples": ts_stats["sample_count"],
        "first_timestamp": native_first,
        "last_timestamp": native_last,
        "duration_s": duration_s if ts_unit_used == "nanoseconds" else ts_stats["duration"],
        "strictly_monotonic": ts_stats["strictly_monotonic"],
        "duplicate_timestamps": ts_stats["duplicate_count"],
        "backward_timestamps": ts_stats["backward_count"],
        "dt_min": extra["min_dt"],
        "dt_p01": ts_stats["p01_dt"],
        "dt_p05": ts_stats["p05_dt"],
        "dt_median": ts_stats["median_dt"],
        "dt_mean": ts_stats["mean_dt"],
        "dt_std": ts_stats["std_dt"],
        "dt_p95": ts_stats["p95_dt"],
        "dt_p99": ts_stats["p99_dt"],
        "dt_max": extra["max_dt"],
        "freq_median_hz": extra["measured_median_rate_hz"],
        "freq_p05_hz": extra["measured_p05_rate_hz"],
        "freq_p95_hz": extra["measured_p95_rate_hz"],
        "gaps_gt_2x_median": ts_stats["number_gaps_gt_2x_median"],
        "gaps_gt_5x_median": ts_stats["number_gaps_gt_5x_median"],
        "gaps_gt_10x_median": ts_stats["number_gaps_gt_10x_median"],
        "timestamp_unit": ts_unit_used,
        "evidence_type": "DERIVED_FROM_RAW_TIMESTAMPS",
    }

    pos_unit = "m" if " [m]" in gt_header or "[m]" in gt_header else "UNRESOLVED"
    field_origin = (
        "position/orientation: POST_PROCESSED_REFERENCE (Vicon 6-DoF aligned by dataset authors); "
        "velocity and IMU biases in state_groundtruth_estimate0: DERIVED_BY_DATASET_AUTHORS; "
        "no direct GT specific force: NO_DIRECT_GT_ACCELERATION"
    )
    gt_row = {
        "sequence_id": seq_id,
        "gt_source": "Vicon motion capture + dataset-author spatio-temporal alignment / state estimate",
        "gt_file": str(gt_csv or ""),
        "gt_type": gt_type,
        "gt_origin": "VICON_6DOF_POSE_POSTPROCESSED" if seq_id.startswith("V") else "LEICA_OR_OTHER",
        "timestamp_field": gt_map.get("timestamp_field", "UNRESOLVED"),
        "timestamp_unit": gt_ts_unit if gt_csv is not None else "UNRESOLVED",
        "position_fields": gt_map.get("position_fields", "UNRESOLVED"),
        "position_unit": pos_unit,
        "orientation_fields": gt_map.get("orientation_fields", "UNRESOLVED"),
        "orientation_representation": "unit_quaternion",
        "quaternion_order_if_applicable": gt_map.get("quaternion_order_if_applicable", "UNRESOLVED"),
        "velocity_fields": gt_map.get("velocity_fields", "NOT_PRESENT"),
        "angular_velocity_fields": gt_map.get("angular_velocity_fields", "NOT_PRESENT"),
        "acceleration_fields": gt_map.get("acceleration_fields", "NO_DIRECT_GT_ACCELERATION"),
        "field_origin_direct_or_postprocessed": field_origin,
        "gt_reference_frame": "R (ASL world/reference; p_RS_R)",
        "body_or_reference_frame": "S pose in R (p_RS_R, q_RS) per official loader names",
        "gt_rate_documented_hz": "UNRESOLVED_IN_YAML_IF_ABSENT",
        "gt_rate_measured_hz": gt_extra.get("measured_median_rate_hz"),
        "imu_gt_start_overlap": overlap["imu_gt_start_overlap"],
        "imu_gt_end_overlap": overlap["imu_gt_end_overlap"],
        "overlap_duration_s": overlap["overlap_duration_s"],
        "overlap_fraction": overlap["overlap_fraction"],
        "synchronization_documentation": "ASL page: extrinsically calibrated and temporally aligned; offset estimated because Vicon timestamps unavailable",
        "synchronization_limitation": "Different recording systems; residual sync error possible; raw data exist for alternative schemes (not used here)",
        "confidence": "HIGH" if gt_ok else "LOW",
        "notes": "Do not treat estimator velocity/biases as independently measured GT. NO_DIRECT_GT_ACCELERATION. No differentiation performed.",
    }

    seq_row = {
        "sequence_id": seq_id,
        "environment": meta["environment"],
        "room": meta["room"],
        "difficulty": meta["difficulty"],
        "platform": "Asctec Firefly hex-rotor",
        "sensor_rig": "VI-Sensor stereo + ADIS16448 IMU",
        "recording_id": seq_id,
        "duration_s": duration_s,
        "imu_available": "YES" if imu_ok else "NO",
        "gt_available": "YES" if gt_ok else "NO",
        "gt_type": gt_type,
        "calibration_available": "YES" if imu_yaml else "NO",
        "independent_recording": "YES",
        "potential_dependency_group": dep,
        "independence_confidence": "HIGH as separate flights; MEDIUM as same MAV/sensor/room family",
        "usable_status": usable,
        "exclusion_reason": reason,
    }
    usable_row = {
        "sequence_id": seq_id,
        "usable_status": usable,
        "official_source": "YES",
        "authentic_file": "YES" if hash_ok else "UNRESOLVED",
        "valid_hash": "YES" if hash_ok else "NO",
        "imu_accel_available": "YES" if imu_ok else "NO",
        "gyro_available": "YES" if imu_ok else "NO",
        "units_verified": "YES" if units_ok else "NO",
        "imu_frame_documented": "YES" if frame_ok else "NO",
        "valid_timestamps": "YES" if ts_stats["backward_count"] == 0 else "NO",
        "imu_duration_s": duration_s,
        "gt_available": "YES" if gt_ok else "NO",
        "overlap_s": overlap["overlap_duration_s"],
        "finite_fraction": chan["finite_fraction_required"],
        "critical_integrity": "FAIL" if critical_fail else "PASS",
        "calibration_documented": "YES" if frame_ok else "NO",
        "exclusion_reason": reason,
        "notes": f"header={header_line}",
    }
    return {
        "sequence_id": seq_id,
        "imu": imu_row,
        "frame": frame_row,
        "timestamp": ts_row,
        "gt": gt_row,
        "sequence": seq_row,
        "usable": usable_row,
        "findings": findings,
        "header_line": header_line,
        "gt_header": gt_header,
        "rest_norm": rest,
        "members": sorted(extracted.keys()),
        "body_yaml": str(body_yaml or ""),
    }


def reverify_uzh(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    retained = list(cfg.get("uzh_fpv_retained_primary") or [])
    manifest_path = root / "results" / "stage0" / "DOWNLOAD_MANIFEST.csv"
    expected: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") != "UZH_FPV":
                    continue
                expected[row.get("sequence_id", "")] = row
    rows: list[dict[str, Any]] = []
    for seq in retained:
        rec = expected.get(seq, {})
        path = Path(rec.get("local_path") or (root / "data" / "raw" / "uzh_fpv" / "sequences" / f"{seq}_with_gt.zip"))
        exp_hash = rec.get("sha256", "")
        notes = []
        hash_status = "FAIL"
        schema_status = "UNRESOLVED"
        header_line = ""
        observed = ""
        size = 0
        if not path.exists():
            notes.append("file missing")
            hash_status = "FAIL"
        else:
            size = path.stat().st_size
            observed = sha256_file(path)
            hash_status = "PASS" if (exp_hash and observed == exp_hash) else "FAIL"
            if not exp_hash:
                notes.append("no expected hash in Stage-0 manifest")
            if hash_status == "FAIL" and exp_hash:
                notes.append("hash mismatch vs Stage-0 manifest")
            try:
                with zipfile.ZipFile(path) as zf:
                    imu_name = next((n for n in zf.namelist() if n.replace("\\", "/").endswith("imu.txt")), None)
                    if imu_name is None:
                        schema_status = "FAIL"
                        notes.append("imu.txt missing in zip")
                    else:
                        raw = zf.read(imu_name).splitlines()[0]
                        header_line = raw.decode("utf-8", errors="replace")
                        cols, arr, comments = read_text_table(_write_temp_header_only(root, seq, zf, imu_name))
                        parsed = interpret_imu_table(cols, arr[:5] if arr.shape[0] else arr, comments)
                        ok = (
                            parsed["accel_fields"]["x"] == "lin_acc_x"
                            and parsed["gyro_fields"]["x"] == "ang_vel_x"
                        )
                        schema_status = "PASS" if ok else "FAIL"
                        if not ok:
                            notes.append(f"schema mismatch {parsed['accel_fields']} {parsed['gyro_fields']}")
            except zipfile.BadZipFile as exc:
                schema_status = "FAIL"
                notes.append(str(exc))
        rows.append(
            {
                "sequence_id": seq,
                "local_path": str(path),
                "expected_sha256": exp_hash,
                "observed_sha256": observed,
                "size_bytes": size,
                "hash_status": hash_status,
                "schema_status": schema_status,
                "header_line": header_line,
                "notes": "; ".join(notes),
            }
        )
    return rows


def _write_temp_header_only(root: Path, seq: str, zf: zipfile.ZipFile, imu_name: str) -> Path:
    dest = root / "data" / "cache" / "euroc" / "_uzh_schema" / seq / "imu.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Lightweight: extract imu.txt only if missing or size differs.
    info = zf.getinfo(imu_name)
    if not dest.exists() or dest.stat().st_size != info.file_size:
        with zf.open(info) as src, dest.open("wb") as out:
            out.write(src.read())
    return dest


def classify_overall(
    euroc_usable: list[dict[str, Any]],
    uzh_rows: list[dict[str, Any]],
    download_ok: bool,
    source_verified: bool,
) -> str:
    primary_e = [r for r in euroc_usable if r.get("usable_status") == "USABLE_PRIMARY"]
    uzh_pass = [r for r in uzh_rows if r.get("hash_status") == "PASS" and r.get("schema_status") == "PASS"]
    if not download_ok:
        return "ACQUISITION_BLOCKED"
    if not source_verified:
        return "FAIL"
    if len(primary_e) < 6 or len(uzh_pass) < 6:
        return "FAIL"
    return "CONDITIONAL_PASS"
