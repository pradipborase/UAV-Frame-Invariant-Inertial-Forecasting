"""Stage-0 path and YAML configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


@dataclass(frozen=True)
class Stage0Config:
    root: Path
    stage0: dict[str, Any]
    sources: dict[str, Any]

    @property
    def download_budget_bytes(self) -> int:
        return int(self.stage0["download_budget_bytes"])

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / "stage0"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.root / "data" / "cache"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "data" / "metadata"

    @property
    def blackbird_raw(self) -> Path:
        return self.raw_dir / "blackbird"

    @property
    def uzh_raw(self) -> Path:
        return self.raw_dir / "uzh_fpv"

    @property
    def blackbird_tools(self) -> Path:
        return self.root / "external" / "blackbird_official_tools"


def load_config(root: Path | None = None) -> Stage0Config:
    root = Path(root) if root is not None else ROOT
    return Stage0Config(
        root=root,
        stage0=load_yaml(root / "configs" / "stage0.yaml"),
        sources=load_yaml(root / "configs" / "sources.yaml"),
    )
