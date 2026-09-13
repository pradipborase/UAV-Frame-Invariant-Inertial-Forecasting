"""Stage 0C must not contain modelling, training, or forecasting implementations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (ROOT / "src", ROOT / "scripts")

FORBIDDEN_SUBSTRINGS = (
    ".fit(",
    ".predict(",
    "train_model",
    "forecast_model",
    "Ridge(",
    "Lasso(",
    "RandomForest",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMRegressor",
    "CatBoost",
    "LSTM",
    "GRU",
    "Transformer(",
    "N-BEATS",
    "NBEATS",
    "GaussianProcessRegressor",
    "VAR(",
    "AutoReg",
    "neural_network",
    "hyperparameter selection",
    "GridSearchCV",
)


def _iter_python() -> list[Path]:
    files: list[Path] = []
    for folder in SCAN_DIRS:
        if folder.exists():
            files.extend(p for p in folder.rglob("*.py") if "euroc_official_tools" not in str(p) and "blackbird_official_tools" not in str(p))
    return files


def test_stage0c_no_modeling_markers() -> None:
    hits: list[str] = []
    for path in _iter_python():
        if path.name == "test_stage0c_no_modeling.py":
            continue
        if "stage2" in path.parts or "stage3" in path.parts or "stage4" in path.parts:
            continue
        if path.name.startswith("run_stage2") or path.name.startswith("run_stage3") or path.name.startswith("build_publication"):
            continue
        if path.name.startswith("test_stage2_") or path.name.startswith("test_stage3_") or path.name.startswith("test_stage4_"):
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for token in FORBIDDEN_SUBSTRINGS:
                if token in line:
                    hits.append(f"{path}:{i}:{token}")
    assert not hits, "Modelling markers in Stage 0C code:\n" + "\n".join(hits)


def test_no_training_artifact_dirs() -> None:
    for name in ("models", "checkpoints", "predictions", "forecasting", "training"):
        path = ROOT / name
        assert not path.exists(), f"Forbidden directory present: {path}"
