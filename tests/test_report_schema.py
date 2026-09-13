from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ROOT_DOCS = [
    "README.md",
    "PROTOCOL.md",
    "DECISIONS.md",
    "DATA_SOURCES.md",
    "DOWNLOAD_PLAN.md",
    "MEASURAND_AUDIT.md",
    "FRAME_AUDIT.md",
    "GROUND_TRUTH_AUDIT.md",
    "SEQUENCE_SELECTION_RULES.md",
    "STAGE0_REPORT.md",
    "STOP_HERE.md",
]

DATASET_AUDIT_SECTIONS = [
    "# Stage 0 — Blackbird and UZH-FPV Scientific Dataset Audit",
    "## 1. Executive decision",
    "## 2. Source authenticity",
    "## 3. Data downloaded",
    "## 4. MIT Blackbird dataset",
    "### 4.1 Sequence structure",
    "### 4.2 IMU fields",
    "### 4.3 Physical units",
    "### 4.4 Coordinate frames",
    "### 4.5 Actual sampling rates",
    "### 4.6 Ground truth",
    "### 4.7 Calibration",
    "### 4.8 Integrity",
    "### 4.9 Usable sequences",
    "### 4.10 Unresolved issues",
    "## 5. UZH-FPV dataset",
    "### 5.1 Sequence structure",
    "### 5.2 Sensor platform choice",
    "### 5.3 IMU fields",
    "### 5.4 Physical units",
    "### 5.5 Coordinate frames",
    "### 5.6 Actual sampling rates",
    "### 5.7 Ground truth",
    "### 5.8 Calibration",
    "### 5.9 Integrity",
    "### 5.10 Usable sequences",
    "### 5.11 Unresolved issues",
    "## 6. Ground-truth comparability",
    "## 7. IMU-measurement comparability",
    "## 8. Cross-dataset coordinate-frame compatibility",
    "## 9. Independent-flight structure",
    "## 10. Recommended usable sequence set",
    "## 11. Recommended common physical quantity",
    "## 12. Blocking issues before Stage 1",
    "## 13. Stage-0 decision",
]


@pytest.mark.skipif(not (ROOT / "results" / "stage0" / "DATASET_AUDIT.md").exists(), reason="audit not run yet")
def test_dataset_audit_required_sections() -> None:
    text = (ROOT / "results" / "stage0" / "DATASET_AUDIT.md").read_text(encoding="utf-8")
    missing = [s for s in DATASET_AUDIT_SECTIONS if s not in text]
    assert not missing, missing


@pytest.mark.skipif(not (ROOT / "STAGE0_REPORT.md").exists(), reason="audit not run yet")
def test_stage0_report_exists_and_has_decision() -> None:
    text = (ROOT / "STAGE0_REPORT.md").read_text(encoding="utf-8")
    assert text.startswith("# STAGE 0 REPORT")
    assert "OVERALL STAGE-0 DECISION:" in text
    assert "NO MODELLING WAS PERFORMED." in text


def test_protocol_exists_when_written() -> None:
    # PROTOCOL.md is a repository document created in Stage 0 scaffolding.
    path = ROOT / "PROTOCOL.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        assert "raw files immutable" in text.lower() or "immutable" in text.lower()
