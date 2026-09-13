"""Stage 1 must not train models or use zero-phase / future-looking filters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (ROOT / "src", ROOT / "scripts")

FORBIDDEN = (
    ".fit(",
    ".predict(",
    "train_model",
    "forecast_model",
    "GridSearchCV",
    "filtfilt(",
    "sosfiltfilt(",
)


def test_stage1_no_modeling_or_zero_phase() -> None:
    hits: list[str] = []
    for folder in SCAN_DIRS:
        for path in folder.rglob("*.py"):
            if "official_tools" in str(path):
                continue
            if "stage2" in path.parts or "stage3" in path.parts or "stage4" in path.parts:
                continue
            if path.name.startswith("run_stage2") or path.name.startswith("run_stage3") or path.name.startswith("build_publication"):
                continue
            if path.name.startswith("test_stage2_") or path.name.startswith("test_stage3_") or path.name.startswith("test_stage4_"):
                continue
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                if line.strip().startswith("#"):
                    continue
                for tok in FORBIDDEN:
                    if tok in line:
                        hits.append(f"{path}:{i}:{tok}")
    assert not hits, hits
