"""Sequence usability and independence classification. Quality only — no forecasting."""

from __future__ import annotations

from typing import Any

USABLE_COLUMNS = [
    "dataset",
    "sequence_id",
    "usable_status",
    "exclusion_reason",
    "criteria_notes",
]

ALLOWED_STATUS = {
    "USABLE_PRIMARY",
    "USABLE_SECONDARY",
    "QUESTIONABLE",
    "EXCLUDE",
    "NOT_DOWNLOADED",
    "GT_WITHHELD",
}


def classify_sequence(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset", "")
    seq = row.get("sequence_id", "")
    downloaded = bool(row.get("downloaded"))
    public_gt = str(row.get("public_gt", "")).upper() in {"YES", "TRUE", "1"}
    withheld = str(row.get("public_gt", "")).upper() in {"NO", "FALSE", "0"} and dataset == "UZH_FPV"
    if withheld and not downloaded:
        return {
            "dataset": dataset,
            "sequence_id": seq,
            "usable_status": "GT_WITHHELD",
            "exclusion_reason": "GROUND_TRUTH_WITHHELD",
            "criteria_notes": "Benchmark GT must not be obtained from unofficial sources.",
        }
    if not downloaded:
        return {
            "dataset": dataset,
            "sequence_id": seq,
            "usable_status": "NOT_DOWNLOADED",
            "exclusion_reason": row.get("exclusion_reason") or "not downloaded in Stage 0",
            "criteria_notes": "Metadata inventory only.",
        }

    reasons: list[str] = []
    imu_ok = bool(row.get("imu_ok"))
    units_ok = bool(row.get("units_ok"))
    frame_ok = bool(row.get("frame_ok"))
    mono_ok = bool(row.get("timestamps_monotonic_or_reported"))
    overlap_ok = float(row.get("overlap_s") or 0) >= float(row.get("min_overlap_s") or 20)
    finite_ok = float(row.get("finite_fraction") or 0) >= 0.99
    corrupt = bool(row.get("corrupt"))
    gt_ok = bool(row.get("gt_ok")) and public_gt
    calib_ok = bool(row.get("calib_ok"))

    if corrupt:
        return {
            "dataset": dataset,
            "sequence_id": seq,
            "usable_status": "EXCLUDE",
            "exclusion_reason": "file corruption or unreadable official archive",
            "criteria_notes": row.get("integrity_notes", ""),
        }
    if not imu_ok:
        reasons.append("IMU acceleration/angular velocity not available")
    if not gt_ok:
        reasons.append("public ground-truth pose not available")
    if not units_ok:
        reasons.append("acceleration units not verified")
    if not frame_ok:
        reasons.append("sensor frame not sufficiently understood")
    if not overlap_ok:
        reasons.append("IMU-GT overlap < 20 s")
    if not finite_ok:
        reasons.append("required IMU channels < 99% finite")
    if not calib_ok:
        reasons.append("calibration/frame relationship incompletely documented")

    blocking = [r for r in reasons if r not in {"calibration/frame relationship incompletely documented", "sensor frame not sufficiently understood"}]
    # Frame/calib gaps can still allow SECONDARY if IMU+GT+units+overlap exist.
    if not imu_ok or not gt_ok or not finite_ok or corrupt:
        status = "EXCLUDE"
    elif not overlap_ok or not units_ok:
        status = "QUESTIONABLE"
    elif reasons:
        status = "USABLE_SECONDARY"
    else:
        status = "USABLE_PRIMARY"

    # Monotonicity: irregular sampling is reported, not automatic exclusion.
    notes = row.get("timestamp_notes", "")
    if not mono_ok:
        notes = (notes + " timestamps not strictly monotonic; reported, not auto-excluded.").strip()

    return {
        "dataset": dataset,
        "sequence_id": seq,
        "usable_status": status,
        "exclusion_reason": "; ".join(reasons) if status != "USABLE_PRIMARY" else "",
        "criteria_notes": notes,
    }
