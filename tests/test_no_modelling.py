"""Fail if Stage-0 source/scripts show modelling or training artefacts.

Documentation prohibition lists are not scanned. Only Python source under
src/, scripts/, and tests/ is inspected.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (ROOT / "src", ROOT / "scripts", ROOT / "tests")

FORBIDDEN_IMPORTS = (
    "import torch",
    "from torch",
    "import tensorflow",
    "from tensorflow",
    "import keras",
    "from keras",
    "import xgboost",
    "from xgboost",
    "import lightgbm",
    "from lightgbm",
    "import catboost",
    "from catboost",
)
FORBIDDEN_CALLS = (
    "train_model",
    "forecast_model",
    "neural_network",
)
FORBIDDEN_DIRS = ("models", "checkpoints", "predictions")
SKLEARN_FIT_MARKERS = (
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.neural_network",
    "sklearn.svm",
)


def _iter_python() -> list[Path]:
    files: list[Path] = []
    for folder in SCAN_DIRS:
        if folder.exists():
            files.extend(folder.rglob("*.py"))
    return files


def test_no_forbidden_ml_imports() -> None:
    hits: list[str] = []
    for path in _iter_python():
        if path.name in {
            "test_no_modelling.py",
            "test_stage0c_no_modeling.py",
            "test_stage1_no_modeling.py",
            "test_stage2_no_deeplearning.py",
            "test_stage4_no_training.py",
        }:
            continue
        if "stage2" in path.parts or "stage3" in path.parts or "stage4" in path.parts:
            continue
        if path.name.startswith("run_stage2") or path.name.startswith("run_stage3") or path.name.startswith("build_publication"):
            continue
        if path.name.startswith("test_stage2_") or path.name.startswith("test_stage3_") or path.name.startswith("test_stage4_") or path.name.startswith("test_lock_") or path.name.startswith("test_numerical_") or path.name.startswith("test_primary_table") or path.name.startswith("test_sequence_counts") or path.name.startswith("test_bootstrap_unit") or path.name.startswith("test_delta_sign") or path.name.startswith("test_transfer_gap") or path.name.startswith("test_high_qf") or path.name.startswith("test_figure_source") or path.name.startswith("test_reproducibility") or path.name.startswith("test_final_results") or path.name.startswith("test_publication_current") or path.name.startswith("test_ridge_") or path.name.startswith("test_fixed100") or path.name.startswith("test_filter_actual") or path.name.startswith("test_input_history"):
            continue
        if path.name == "stage4_common.py":
            continue
        text = path.read_text(encoding="utf-8")
        lower_lines = text.splitlines()
        for i, line in enumerate(lower_lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for token in FORBIDDEN_IMPORTS:
                if token in line:
                    hits.append(f"{path}:{i}:{token}")
            for token in FORBIDDEN_CALLS:
                if token in line:
                    hits.append(f"{path}:{i}:{token}")
            if ".fit(" in line and any(m in text for m in SKLEARN_FIT_MARKERS):
                hits.append(f"{path}:{i}:sklearn_fit")
    assert not hits, "Modelling markers in Stage-0 code:\n" + "\n".join(hits)


def test_no_model_directories() -> None:
    for name in FORBIDDEN_DIRS:
        path = ROOT / name
        assert not path.exists(), f"Forbidden directory present: {path}"
