"""Stage-4 publication builder must not train or select new checkpoints."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = [
    ROOT / "src" / "stage4",
    ROOT / "scripts" / "build_publication_evidence.py",
    ROOT / "scripts" / "build_publication_current.py",
]
FORBIDDEN = (
    "optimizer" + ".step",
    "loss" + ".backward",
    "model" + ".fit",
    "train_model(",
    "backward()",
    "GridSearchCV",
    "import torch",
    "from torch",
)


def _iter_python() -> list[Path]:
    files: list[Path] = []
    for item in SCAN:
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            files.extend(item.rglob("*.py"))
    return files


def test_stage4_source_has_no_training_calls() -> None:
    hits: list[str] = []
    for path in _iter_python():
        if path.name == "test_stage4_no_training.py":
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            for token in FORBIDDEN:
                if token in line:
                    hits.append(f"{path}:{i}:{token}")
    assert not hits, "Training markers in Stage-4 code:\n" + "\n".join(hits)


def test_stage4_script_docstring_forbids_training() -> None:
    text = (ROOT / "scripts" / "build_publication_evidence.py").read_text(encoding="utf-8")
    assert "Does not train models" in text
    current = (ROOT / "scripts" / "build_publication_current.py").read_text(encoding="utf-8")
    assert "Does not train models" in current
    assert "Never selects a Ridge comparator by looking at target-domain RMSE" in current
