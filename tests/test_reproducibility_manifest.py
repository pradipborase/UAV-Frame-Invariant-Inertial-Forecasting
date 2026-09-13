"""Reproducibility manifest must list protocol, result, and environment hashes."""

from __future__ import annotations

from stage4_common import csv_rows, require_stage4


REQUIRED_ITEMS = {
    "protocol_md",
    "protocol_yaml",
    "stage2_config",
    "stage2_freeze",
    "stage3_config",
    "stage3_grid",
    "stage3_freeze",
    "table_primary",
    "table_vs_ridge",
    "horizon_data",
    "high_qf_data",
    "transfer_gap",
    "footprint",
    "claim_matrix",
    "fig2_data",
    "fig3_data",
    "software_python",
    "bootstrap",
}


def test_manifest_contains_required_items() -> None:
    require_stage4()
    rows = csv_rows("REPRODUCIBILITY_MANIFEST.csv")
    items = {r["item"] for r in rows}
    missing = REQUIRED_ITEMS - items
    assert not missing, missing
    hashed = [r for r in rows if r["item"] in {"protocol_md", "table_primary", "fig2_data"}]
    for row in hashed:
        assert len(row["value"]) == 64
