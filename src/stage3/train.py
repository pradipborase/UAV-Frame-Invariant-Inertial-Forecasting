"""Equal-sequence minibatch training with source-train-only scalers and early stopping."""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from stage2.metrics import rmse_all
from stage2.weights import equal_recording_weights
from stage2.windows import SequenceWindows

from .models import build_model, count_parameters


def predict_from_checkpoint(
    path: str | os.PathLike[str],
    packs: dict[str, SequenceWindows],
    seq_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    model = build_model(str(payload["architecture"]), config, n_in=int(payload["n_in"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = scaler_from_dict(payload["scaler"])
    preds = predict_physical(model, packs, seq_ids, scaler, bool(payload["use_qw"]))
    return preds, payload

_PACKS: dict[str, SequenceWindows] | None = None


def configure_worker_threads(n_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(int(n_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def init_worker(packs: dict[str, SequenceWindows], n_threads: int = 1) -> None:
    global _PACKS
    _PACKS = packs
    configure_worker_threads(n_threads)


@dataclass
class ChannelScaler:
    """Equal-recording-weighted channel scaler. Fit on source training windows only."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    fit_dataset: str
    fit_sequences: list[str]
    n_windows: int
    use_qw: bool
    fit_purpose: str = "source_train_feature_and_target_scaler"

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "fit_dataset": self.fit_dataset,
            "fit_sequences": list(self.fit_sequences),
            "n_windows": self.n_windows,
            "use_qw": self.use_qw,
            "fit_purpose": self.fit_purpose,
        }


def features_for_pack(pack: SequenceWindows, use_qw: bool) -> np.ndarray:
    if use_qw:
        return np.stack([pack.hist_qf, pack.hist_qw], axis=-1).astype(np.float64, copy=False)
    return pack.hist_qf[..., None].astype(np.float64, copy=False)


def fit_channel_scaler(
    packs: dict[str, SequenceWindows],
    train_ids: list[str],
    *,
    dataset: str,
    use_qw: bool,
) -> ChannelScaler:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    seq_ids: list[str] = []
    for sid in train_ids:
        pack = packs[sid]
        x = features_for_pack(pack, use_qw)
        xs.append(x)
        ys.append(np.asarray(pack.future_qf, dtype=np.float64))
        seq_ids.extend([sid] * int(x.shape[0]))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = equal_recording_weights(seq_ids).astype(np.float64)
    x_mean_win = x.mean(axis=1)
    feat_mean = np.sum(w[:, None] * x_mean_win, axis=0)
    feat_var = np.sum(w[:, None] * ((x - feat_mean) ** 2).mean(axis=1), axis=0)
    feat_std = np.sqrt(np.maximum(feat_var, 0.0))
    feat_std = np.where(feat_std < 1e-12, 1.0, feat_std)
    y_mean_win = y.mean(axis=1)
    tgt_mean = float(np.sum(w * y_mean_win))
    tgt_var = float(np.sum(w * np.mean((y - tgt_mean) ** 2, axis=1)))
    tgt_std = float(np.sqrt(max(tgt_var, 0.0)))
    if tgt_std < 1e-12:
        tgt_std = 1.0
    return ChannelScaler(
        feature_mean=feat_mean.astype(np.float64),
        feature_std=feat_std.astype(np.float64),
        target_mean=tgt_mean,
        target_std=tgt_std,
        fit_dataset=dataset,
        fit_sequences=list(train_ids),
        n_windows=int(x.shape[0]),
        use_qw=bool(use_qw),
    )


def scaler_from_dict(payload: dict[str, Any]) -> ChannelScaler:
    return ChannelScaler(
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
        fit_dataset=str(payload["fit_dataset"]),
        fit_sequences=list(payload["fit_sequences"]),
        n_windows=int(payload["n_windows"]),
        use_qw=bool(payload["use_qw"]),
        fit_purpose=str(payload.get("fit_purpose", "source_train_feature_and_target_scaler")),
    )


@dataclass
class TrainBank:
    X: np.ndarray
    y: np.ndarray
    seq_ids: list[str]
    starts: np.ndarray
    counts: np.ndarray
    n_seq: int


def build_bank(
    packs: dict[str, SequenceWindows],
    seq_ids: list[str],
    scaler: ChannelScaler,
    use_qw: bool,
) -> TrainBank:
    xs = []
    ys = []
    starts = []
    counts = []
    offset = 0
    for sid in seq_ids:
        x = scaler.transform_x(features_for_pack(packs[sid], use_qw)).astype(np.float32)
        y = scaler.transform_y(np.asarray(packs[sid].future_qf, dtype=np.float64)).astype(np.float32)
        xs.append(x)
        ys.append(y)
        starts.append(offset)
        counts.append(int(x.shape[0]))
        offset += int(x.shape[0])
    return TrainBank(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        seq_ids=list(seq_ids),
        starts=np.asarray(starts, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        n_seq=len(seq_ids),
    )


def _sample_indices(bank: TrainBank, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    seq_choice = rng.integers(0, bank.n_seq, size=int(batch_size))
    local = (rng.random(int(batch_size)) * bank.counts[seq_choice]).astype(np.int64)
    return bank.starts[seq_choice] + local


@torch.no_grad()
def predict_physical(
    model: nn.Module,
    packs: dict[str, SequenceWindows],
    seq_ids: list[str],
    scaler: ChannelScaler,
    use_qw: bool,
    *,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, np.ndarray] = {}
    for sid in seq_ids:
        x = scaler.transform_x(features_for_pack(packs[sid], use_qw)).astype(np.float32)
        preds: list[np.ndarray] = []
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size])
            pred = model(xb).cpu().numpy()
            preds.append(scaler.inverse_y(pred))
        out[sid] = np.concatenate(preds, axis=0)
    return out


def equal_sequence_norm_mse(
    model: nn.Module,
    bank: TrainBank,
    *,
    batch_size: int = 512,
    max_windows_per_seq: int | None = 96,
) -> float:
    model.eval()
    seq_mse: list[float] = []
    with torch.no_grad():
        for i, _sid in enumerate(bank.seq_ids):
            start = int(bank.starts[i])
            count = int(bank.counts[i])
            if max_windows_per_seq is not None and count > int(max_windows_per_seq):
                pick = np.linspace(0, count - 1, int(max_windows_per_seq)).astype(np.int64)
                x = bank.X[start + pick]
                y = bank.y[start + pick]
            else:
                x = bank.X[start : start + count]
                y = bank.y[start : start + count]
            n = 0
            sse = 0.0
            for s in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[s : s + batch_size])
                yb = torch.from_numpy(y[s : s + batch_size])
                pred = model(xb)
                diff = pred - yb
                sse += float((diff * diff).sum().item())
                n += int(diff.numel())
            seq_mse.append(sse / max(n, 1))
    return float(np.mean(seq_mse)) if seq_mse else float("nan")


@dataclass
class FitResult:
    state_dict: dict[str, torch.Tensor]
    n_params: int
    best_epoch: int
    best_val_loss: float
    train_seconds: float
    scaler: ChannelScaler
    train_ids: list[str]
    val_ids: list[str]
    architecture: str
    config_id: str
    seed: int
    use_qw: bool
    n_in: int
    target_dataset_access_during_fit: str = "NO"
    leakage: dict[str, str] = field(default_factory=dict)


def train_model(
    packs: dict[str, SequenceWindows],
    train_ids: list[str],
    val_ids: list[str] | None,
    *,
    architecture: str,
    config: dict[str, Any],
    seed: int,
    dataset: str,
    use_qw: bool,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    windows_per_sequence_per_epoch: int,
    fixed_epochs: int | None = None,
) -> FitResult:
    if not train_ids:
        raise ValueError("train_ids is empty")
    n_in = 2 if use_qw else 1
    seed_everything(seed)
    scaler = fit_channel_scaler(packs, train_ids, dataset=dataset, use_qw=use_qw)
    train_bank = build_bank(packs, train_ids, scaler, use_qw)
    val_bank = build_bank(packs, val_ids, scaler, use_qw) if val_ids else None
    model = build_model(architecture, config, n_in=n_in)
    n_params = count_parameters(model)
    opt = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(int(seed))
    n_train = max(1, len(train_ids))
    steps = max(6, int(np.ceil(n_train * windows_per_sequence_per_epoch / batch_size)))
    use_early = val_bank is not None and fixed_epochs is None
    limit = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
    limit = max(1, min(limit, int(max_epochs)))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 1
    bad = 0
    t0 = time.perf_counter()
    x_train = torch.from_numpy(train_bank.X)
    y_train = torch.from_numpy(train_bank.y)
    for epoch in range(1, limit + 1):
        model.train()
        for _ in range(steps):
            idx = torch.from_numpy(_sample_indices(train_bank, batch_size, rng))
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()
        if use_early:
            val_loss = equal_sequence_norm_mse(model, val_bank)
            if val_loss < best_val - float(min_delta):
                best_val = float(val_loss)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= int(patience):
                    break
        else:
            best_epoch = epoch
    if not use_early:
        best_state = copy.deepcopy(model.state_dict())
        best_val = float("nan") if val_bank is None else equal_sequence_norm_mse(model, val_bank)
    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - t0
    leakage = {
        "object_name": f"{architecture}_{config['config_id']}_seed{seed}",
        "model": architecture,
        "fit_dataset": dataset,
        "fit_sequences": ",".join(train_ids),
        "validation_dataset": dataset if val_ids else "",
        "validation_sequences": ",".join(val_ids or []),
        "target_dataset_access_during_fit": "NO",
        "status": "PASS",
    }
    return FitResult(
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        n_params=n_params,
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
        train_seconds=float(elapsed),
        scaler=scaler,
        train_ids=list(train_ids),
        val_ids=list(val_ids or []),
        architecture=architecture,
        config_id=str(config["config_id"]),
        seed=int(seed),
        use_qw=bool(use_qw),
        n_in=n_in,
        leakage=leakage,
    )


def evaluate_fit(
    result: FitResult,
    packs: dict[str, SequenceWindows],
    seq_ids: list[str],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    model = build_model(result.architecture, config, n_in=result.n_in)
    model.load_state_dict(result.state_dict)
    model.eval()
    return predict_physical(model, packs, seq_ids, result.scaler, result.use_qw)


def execute_hp_job(job: dict[str, Any]) -> dict[str, Any]:
    configure_worker_threads(1)
    if _PACKS is None:
        raise RuntimeError("worker packs were not initialised")
    packs = _PACKS
    config = dict(job["config"])
    architecture = str(job["architecture"])
    train_ids = list(job["train_ids"])
    val_ids = list(job["val_ids"])
    dataset = str(job["dataset"])
    forbidden = set(job.get("forbidden_sequences", []))
    if forbidden.intersection(train_ids) or forbidden.intersection(val_ids):
        raise RuntimeError("forbidden sequence leaked into train/val")
    result = train_model(
        packs,
        train_ids,
        val_ids,
        architecture=architecture,
        config=config,
        seed=int(job["seed"]),
        dataset=dataset,
        use_qw=bool(job["use_qw"]),
        batch_size=int(job["batch_size"]),
        max_epochs=int(job["max_epochs"]),
        patience=int(job["patience"]),
        min_delta=float(job["min_delta"]),
        windows_per_sequence_per_epoch=int(job["windows_per_sequence_per_epoch"]),
        fixed_epochs=None,
    )
    preds = evaluate_fit(result, packs, val_ids, config)
    rmse_by_seq = {sid: rmse_all(preds[sid], packs[sid].future_qf) for sid in val_ids}
    return {
        "job_id": job["job_id"],
        "purpose": job["purpose"],
        "scope": job["scope"],
        "architecture": architecture,
        "config_id": config["config_id"],
        "seed": int(job["seed"]),
        "dataset": dataset,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "use_qw": bool(job["use_qw"]),
        "n_params": result.n_params,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "train_seconds": result.train_seconds,
        "rmse_by_seq": rmse_by_seq,
        "mean_val_rmse": float(np.mean(list(rmse_by_seq.values()))) if rmse_by_seq else float("nan"),
        "target_dataset_access_during_fit": "NO",
        "forbidden_ok": True,
    }


def execute_final_job(job: dict[str, Any]) -> dict[str, Any]:
    configure_worker_threads(1)
    if _PACKS is None:
        raise RuntimeError("worker packs were not initialised")
    packs = _PACKS
    config = dict(job["config"])
    architecture = str(job["architecture"])
    train_ids = list(job["train_ids"])
    eval_ids = list(job["eval_ids"])
    dataset = str(job["dataset"])
    forbidden = set(job.get("forbidden_sequences", []))
    if forbidden.intersection(train_ids):
        raise RuntimeError("forbidden sequence leaked into final training")
    result = train_model(
        packs,
        train_ids,
        None,
        architecture=architecture,
        config=config,
        seed=int(job["seed"]),
        dataset=dataset,
        use_qw=bool(job["use_qw"]),
        batch_size=int(job["batch_size"]),
        max_epochs=int(job["max_epochs"]),
        patience=int(job["patience"]),
        min_delta=float(job["min_delta"]),
        windows_per_sequence_per_epoch=int(job["windows_per_sequence_per_epoch"]),
        fixed_epochs=int(job["fixed_epochs"]),
    )
    save_path = job.get("save_path") or ""
    if save_path:
        torch.save(
            {
                "state_dict": result.state_dict,
                "scaler": result.scaler.as_dict(),
                "architecture": architecture,
                "config": config,
                "seed": int(job["seed"]),
                "n_in": result.n_in,
                "use_qw": result.use_qw,
                "n_params": result.n_params,
                "fixed_epochs": int(job["fixed_epochs"]),
                "best_epoch": result.best_epoch,
                "train_ids": train_ids,
                "fit_dataset": dataset,
            },
            save_path,
        )
    preds = evaluate_fit(result, packs, eval_ids, config) if eval_ids else {}
    rmse_by_seq = {sid: rmse_all(preds[sid], packs[sid].future_qf) for sid in eval_ids}
    return {
        "job_id": job["job_id"],
        "purpose": job["purpose"],
        "scope": job.get("scope", ""),
        "architecture": architecture,
        "config_id": config["config_id"],
        "seed": int(job["seed"]),
        "dataset": dataset,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "use_qw": bool(job["use_qw"]),
        "n_params": result.n_params,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "train_seconds": result.train_seconds,
        "rmse_by_seq": rmse_by_seq,
        "predictions": preds,
        "scaler": result.scaler.as_dict(),
        "save_path": save_path,
        "target_dataset_access_during_fit": "NO",
        "leakage": result.leakage,
        "n_in": result.n_in,
    }
