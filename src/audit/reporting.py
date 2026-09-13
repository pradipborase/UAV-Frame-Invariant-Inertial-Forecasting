"""Generate Stage-0 CSV/JSON/Markdown artefacts from the audit result object."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .channel_inventory import CHANNEL_COLUMNS
from .compatibility import COMPAT_COLUMNS
from .config import Stage0Config
from .downloader import MANIFEST_COLUMNS
from .frame_audit import FRAME_COLUMNS
from .groundtruth_audit import GT_COLUMNS
from .timestamp_audit import REQUIRED_FIELDS

UTC = timezone.utc


def _write_csv(path: Path, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def write_all_tables(cfg: Stage0Config, result: dict[str, Any]) -> None:
    out = cfg.results_dir
    _write_csv(out / "DATASET_SUMMARY.csv", result["dataset_summary_columns"], result["dataset_summary"])
    _write_csv(out / "SEQUENCE_INVENTORY.csv", result["sequence_inventory_columns"], result["sequence_inventory"])
    _write_csv(out / "CHANNEL_INVENTORY.csv", CHANNEL_COLUMNS, result["channel_inventory"])
    _write_csv(out / "IMU_AUDIT.csv", result["imu_audit_columns"], result["imu_audit"])
    _write_csv(out / "GROUND_TRUTH_INVENTORY.csv", GT_COLUMNS, result["gt_inventory"])
    _write_csv(out / "FRAME_INVENTORY.csv", FRAME_COLUMNS, result["frame_inventory"])
    ts_cols = ["dataset", "sequence_id", "stream"] + list(REQUIRED_FIELDS)
    _write_csv(out / "TIMESTAMP_AUDIT.csv", ts_cols, result["timestamp_audit"])
    _write_csv(
        out / "INTEGRITY_FINDINGS.csv",
        [
            "dataset",
            "sequence_id",
            "finding_id",
            "category",
            "severity",
            "status",
            "observed",
            "expected_or_rule",
            "interpretation",
            "action_required",
        ],
        result["integrity_findings"],
    )
    _write_csv(out / "CROSS_DATASET_COMPATIBILITY.csv", COMPAT_COLUMNS, result["compatibility_rows"])
    _write_csv(
        out / "USABLE_SEQUENCES.csv",
        [
            "dataset",
            "sequence_id",
            "recording_id",
            "sensor_platform",
            "environment",
            "trajectory_family",
            "speed_condition",
            "usable_status",
            "public_gt",
            "overlap_with_imu_seconds",
            "exclusion_reason",
            "criteria_notes",
        ],
        result["usable_sequences"],
    )
    (out / "stage0_summary.json").write_text(json.dumps(result["summary_json"], indent=2), encoding="utf-8")
    (out / "DATASET_AUDIT.md").write_text(result["dataset_audit_md"], encoding="utf-8")
    (cfg.root / "STAGE0_REPORT.md").write_text(result["stage0_report_md"], encoding="utf-8")
    (cfg.root / "MEASURAND_AUDIT.md").write_text(result["measurand_md"], encoding="utf-8")
    (cfg.root / "FRAME_AUDIT.md").write_text(result["frame_audit_md"], encoding="utf-8")
    (cfg.root / "GROUND_TRUTH_AUDIT.md").write_text(result["gt_audit_md"], encoding="utf-8")
    (cfg.root / "STOP_HERE.md").write_text(result["stop_md"], encoding="utf-8")
    (cfg.root / ".stage0_complete").write_text(result["stage0_complete"], encoding="utf-8")
