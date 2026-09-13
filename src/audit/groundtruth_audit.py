"""Ground-truth field audit. No differentiation of pose is performed."""

from __future__ import annotations

from typing import Any

GT_COLUMNS = [
    "dataset",
    "sequence_id",
    "gt_source",
    "gt_topic_or_file",
    "gt_type",
    "timestamp_field",
    "position_fields",
    "orientation_fields",
    "velocity_fields",
    "acceleration_fields",
    "position_unit",
    "orientation_convention",
    "reference_frame",
    "sampling_rate_documented_hz",
    "sampling_rate_measured_median_hz",
    "public_gt",
    "coverage_start",
    "coverage_end",
    "overlap_with_imu_seconds",
    "overlap_fraction",
    "confidence",
    "notes",
]


def empty_gt_row(dataset: str, sequence_id: str, **kwargs: Any) -> dict[str, Any]:
    row = {key: kwargs.get(key, "") for key in GT_COLUMNS}
    row["dataset"] = dataset
    row["sequence_id"] = sequence_id
    row.setdefault("acceleration_fields", "NOT_PROVIDED")
    row.setdefault("velocity_fields", "NOT_PROVIDED")
    row.setdefault("gt_type", "UNRESOLVED")
    return row
