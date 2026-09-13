"""Shared Stage-4 test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from stage4.io import read_csv

ROOT = Path(__file__).resolve().parents[1]
STAGE4 = ROOT / "results" / "stage4"


def require_stage4() -> Path:
    if not (STAGE4 / "TABLE_PRIMARY_RESULTS.csv").exists():
        pytest.skip("Stage-4 artefacts not built yet")
    return STAGE4


def csv_rows(name: str) -> list[dict[str, str]]:
    path = require_stage4() / name
    return read_csv(path)
