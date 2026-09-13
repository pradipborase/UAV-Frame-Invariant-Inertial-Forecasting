"""Predetermined compact hyperparameter grids. Written and hashed before any training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audit.stage0c import write_csv


ARCH_ORDER = ("TCN", "GRU", "TRANSFORMER")


def load_stage3_yaml(root: Path) -> dict[str, Any]:
    path = root / "configs" / "stage3_advanced.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("stage3_advanced.yaml must be a mapping")
    return data


def flatten_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arches = cfg["architectures"]
    for arch in ARCH_ORDER:
        block = arches[arch]
        configs = list(block["configs"])
        if len(configs) > int(block["max_configs"]):
            raise ValueError(f"{arch} has more than {block['max_configs']} configs")
        if len(configs) > 12:
            raise ValueError(f"{arch} exceeds the Stage-3 maximum of 12 configs")
        for item in configs:
            row = {"architecture": arch, **item}
            rows.append(row)
    return rows


def config_by_id(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in flatten_configs(cfg):
        cid = str(row["config_id"])
        if cid in out:
            raise ValueError(f"duplicate config_id {cid}")
        out[cid] = row
    return out


def write_hyperparameter_grid(root: Path, cfg: dict[str, Any]) -> Path:
    rows = flatten_configs(cfg)
    columns = [
        "architecture",
        "config_id",
        "channels",
        "kernel_size",
        "dilations",
        "hidden_size",
        "n_layers",
        "d_model",
        "n_heads",
        "ff_multiplier",
        "pooling",
        "dropout",
        "activation",
        "lr",
        "weight_decay",
    ]
    serialised: list[dict[str, Any]] = []
    for row in rows:
        item = {k: "" for k in columns}
        item.update({k: row[k] for k in columns if k in row})
        if "channels" in row and row["channels"] != "":
            item["channels"] = ",".join(str(int(v)) for v in row["channels"])
        if "dilations" in row and row["dilations"] != "":
            item["dilations"] = ",".join(str(int(v)) for v in row["dilations"])
        serialised.append(item)
    path = root / "results" / "stage3" / "HYPERPARAMETER_GRID.csv"
    write_csv(path, columns, serialised)
    return path
