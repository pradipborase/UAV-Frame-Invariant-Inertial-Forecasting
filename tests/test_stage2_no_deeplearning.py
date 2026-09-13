"""Stage 2 may use Ridge, but must not import deep-learning packages."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
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


def test_no_deep_learning_imports() -> None:
    hits: list[str] = []
    for folder in (ROOT / "src", ROOT / "scripts"):
        for path in folder.rglob("*.py"):
            if "official_tools" in str(path):
                continue
            if "stage3" in path.parts or path.name.startswith("run_stage3") or path.name.startswith("test_stage3"):
                continue
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                if line.strip().startswith("#"):
                    continue
                for tok in FORBIDDEN:
                    if tok in line:
                        hits.append(f"{path}:{i}:{tok}")
    assert not hits, hits
