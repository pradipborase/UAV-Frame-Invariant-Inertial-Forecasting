"""Stage 3 orchestration: Phase 3A source freeze, then Phase 3B zero-shot transfer."""
# HISTORICAL AUDIT ORCHESTRATOR — NOT AUTHORITATIVE FOR MANUSCRIPT RIDGE REPORTING.
# This file preserves an earlier descriptive UZH_TO_EUROC B2 Ridge mapping for provenance.
# The publication-current primary Ridge comparator is B3_RIDGE_QF_QW in ALL four contexts,
# selected by source-domain validation only. See:
#   publication_current/RIDGE_COMPARATOR_MAP.csv
#   docs/POSTHOC_REPORTING_CORRECTIONS.md
# Do not use the historical BEST_RIDGE mapping below to regenerate manuscript primary results.

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from audit.hashing import sha256_file
from audit.stage0c import write_csv
from stage1.folds import EUROC_ROOM, UZH_GROUP
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY
from stage2.bootstrap import bootstrap_mean_ci, paired_sign_flip_p, win_loss_tie
from stage2.io_processed import load_all_processed
from stage2.metrics import equal_sequence_mean, mae_all, rmse_all, rmse_horizon_curve, sequence_result_row, skill_score
from stage2.selection import unique_groups
from stage2.windows import SequenceWindows, build_sequence_windows

from .figures import figure_a_rmse_ci, figure_b_horizon, figure_c_paired, figure_d_transfer_gap, figure_e_high_qf, figure_f_footprint
from .grids import config_by_id, flatten_configs, load_stage3_yaml, write_hyperparameter_grid
from .models import build_model, count_parameters
from .protocol import verify_stage3_prerequisites
from .train import (
    configure_worker_threads,
    execute_final_job,
    execute_hp_job,
    init_worker,
    predict_from_checkpoint,
    seed_everything,
)

UTC = timezone.utc
ARCHS = ("TCN", "GRU", "TRANSFORMER")
SEEDS = (20260912, 20260913, 20260914)
BEST_RIDGE = {
    "WITHIN_EUROC": "B3_RIDGE_QF_QW",
    "WITHIN_UZH": "B3_RIDGE_QF_QW",
    "EUROC_TO_UZH": "B3_RIDGE_QF_QW",
    "UZH_TO_EUROC": "B2_RIDGE_QF",
}
RIDGE_PARAMS = {
    "WITHIN_EUROC": 50,
    "WITHIN_UZH": 20,
    "EUROC_TO_UZH": 50,
    "UZH_TO_EUROC": 25,
}
DISPLAY = {
    "B0_PERSISTENCE": "Persistence",
    "BEST_RIDGE": "Best Ridge",
    "TCN": "TCN",
    "GRU": "GRU",
    "TRANSFORMER": "Transformer",
}
H_LABELS = list(range(1, 21))
SEQ_COLS = (
    [
        "evaluation_context",
        "dataset",
        "sequence_id",
        "model",
        "n_origins",
        "rmse_all_horizons",
        "mae_all_horizons",
        "rmse_50ms",
        "rmse_100ms",
        "rmse_200ms",
        "skill",
    ]
    + [f"rmse_h{i:02d}" for i in H_LABELS]
    + ["fold_id", "notes", "config_id", "seeds_mean"]
)


def utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def high_target_rmse(pred: np.ndarray, actual: np.ndarray, p95: float) -> float:
    mask = np.asarray(actual, dtype=np.float64) > float(p95)
    if not np.any(mask):
        return float("nan")
    err = np.asarray(pred, dtype=np.float64)[mask] - np.asarray(actual, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(err * err)))


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_hp_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["job_id"]] = rec
    return done


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(rec), ensure_ascii=True) + "\n")


def common_train_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": int(cfg["batch_size"]),
        "max_epochs": int(cfg["max_epochs"]),
        "patience": int(cfg["early_stopping_patience"]),
        "min_delta": float(cfg["min_delta_norm_loss"]),
        "windows_per_sequence_per_epoch": int(cfg["windows_per_sequence_per_epoch"]),
    }


def make_lgo_jobs(
    *,
    purpose: str,
    scope: str,
    dataset: str,
    seq_ids: list[str],
    groups: dict[str, str],
    configs: list[dict[str, Any]],
    seeds: tuple[int, ...],
    use_qw: bool,
    yaml_cfg: dict[str, Any],
    forbidden: list[str] | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    kw = common_train_kwargs(yaml_cfg)
    for config in configs:
        arch = str(config["architecture"])
        for seed in seeds:
            for g in unique_groups(seq_ids, groups):
                train_ids = [s for s in seq_ids if groups[s] != g]
                val_ids = [s for s in seq_ids if groups[s] == g]
                if not train_ids or not val_ids:
                    continue
                job_id = f"{purpose}|{scope}|{arch}|{config['config_id']}|seed{seed}|valgrp{g}|qw{int(use_qw)}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "purpose": purpose,
                        "scope": scope,
                        "architecture": arch,
                        "config": config,
                        "seed": int(seed),
                        "dataset": dataset,
                        "train_ids": train_ids,
                        "val_ids": val_ids,
                        "use_qw": bool(use_qw),
                        "forbidden_sequences": list(forbidden or []),
                        **kw,
                    }
                )
    return jobs


def run_hp_jobs(
    jobs: list[dict[str, Any]],
    packs: dict[str, SequenceWindows],
    *,
    n_workers: int,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    done = load_hp_jsonl(jsonl_path)
    pending = [j for j in jobs if j["job_id"] not in done]
    print(f"HP jobs: {len(jobs)} total, {len(done)} cached, {len(pending)} to run, workers={n_workers}", flush=True)
    t_all = time.perf_counter()
    if pending:
        workers = max(1, int(n_workers))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(packs, 1),
        ) as pool:
            futs = {pool.submit(execute_hp_job, job): job for job in pending}
            n_fin = 0
            for fut in as_completed(futs):
                job = futs[fut]
                rec = fut.result()
                append_jsonl(jsonl_path, rec)
                done[rec["job_id"]] = rec
                n_fin += 1
                if n_fin % 10 == 0 or n_fin == len(pending):
                    elapsed = time.perf_counter() - t_all
                    rate = n_fin / max(elapsed, 1e-6)
                    remain = (len(pending) - n_fin) / max(rate, 1e-9)
                    print(
                        f"  HP progress {n_fin}/{len(pending)} "
                        f"({100*n_fin/len(pending):.1f}%) "
                        f"ETA {remain/60:.1f} min last={job['job_id']}",
                        flush=True,
                    )
    return [done[j["job_id"]] for j in jobs]


def run_final_jobs(
    jobs: list[dict[str, Any]],
    packs: dict[str, SequenceWindows],
    *,
    n_workers: int,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    print(f"Final jobs: {len(jobs)}", flush=True)
    out: list[dict[str, Any]] = []
    workers = max(1, int(n_workers))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(packs, 1),
    ) as pool:
        futs = {pool.submit(execute_final_job, job): job for job in jobs}
        n_fin = 0
        for fut in as_completed(futs):
            rec = fut.result()
            out.append(rec)
            n_fin += 1
            if n_fin % 5 == 0 or n_fin == len(jobs):
                print(f"  final progress {n_fin}/{len(jobs)} last={rec['job_id']}", flush=True)
    return out


def select_config(
    records: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    param_lookup: dict[str, int],
) -> dict[str, Any]:
    by_cfg: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_cfg.setdefault(str(rec["config_id"]), []).append(rec)
    scored: list[dict[str, Any]] = []
    for config in configs:
        cid = str(config["config_id"])
        recs = by_cfg.get(cid, [])
        if not recs:
            continue
        seed_scores: list[float] = []
        epochs: list[int] = []
        seq_rmse_all: list[float] = []
        for seed in SEEDS:
            seed_recs = [r for r in recs if int(r["seed"]) == int(seed)]
            seq_vals: list[float] = []
            for r in seed_recs:
                seq_vals.extend(float(v) for v in r["rmse_by_seq"].values())
                epochs.append(int(r["best_epoch"]))
            if seq_vals:
                seed_scores.append(float(np.mean(seq_vals)))
                seq_rmse_all.extend(seq_vals)
        if not seed_scores:
            continue
        scored.append(
            {
                "config_id": cid,
                "architecture": config["architecture"],
                "score": float(np.mean(seed_scores)),
                "seed_scores": seed_scores,
                "selected_epochs": int(max(1, round(float(np.median(epochs))))),
                "n_params": int(param_lookup.get(cid, 0)),
                "n_records": len(recs),
                "mean_seq_rmse_all_draws": float(np.mean(seq_rmse_all)) if seq_rmse_all else float("nan"),
            }
        )
    if not scored:
        raise RuntimeError("no scores available for selection")
    scored.sort(key=lambda r: (r["score"], r["n_params"], r["config_id"]))
    best = dict(scored[0])
    best["all_scored"] = scored
    return best


def packs_from_root(root: Path) -> dict[str, SequenceWindows]:
    raw = load_all_processed(root)
    packs: dict[str, SequenceWindows] = {}
    for (_ds, sid), rec in raw.items():
        packs[sid] = build_sequence_windows(rec)
    return packs


def groups_for(dataset: str) -> dict[str, str]:
    return dict(EUROC_ROOM) if dataset == "EUROC" else dict(UZH_GROUP)


def seqs_for(dataset: str) -> list[str]:
    return list(EUROC_PRIMARY) if dataset == "EUROC" else list(UZH_PRIMARY)


def average_seed_metrics(
    seed_preds: list[dict[str, np.ndarray]],
    packs: dict[str, SequenceWindows],
    seq_ids: list[str],
    *,
    context: str,
    dataset: str,
    model: str,
    persistence: dict[str, float],
    extra: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sid in seq_ids:
        rmse_vals = [rmse_all(p[sid], packs[sid].future_qf) for p in seed_preds]
        mae_vals = [mae_all(p[sid], packs[sid].future_qf) for p in seed_preds]
        curves = [rmse_horizon_curve(p[sid], packs[sid].future_qf) for p in seed_preds]
        curve = np.mean(np.stack(curves, axis=0), axis=0)
        rmse = float(np.mean(rmse_vals))
        persist = persistence.get(sid)
        row: dict[str, object] = {
            "evaluation_context": context,
            "dataset": dataset,
            "sequence_id": sid,
            "model": model,
            "n_origins": packs[sid].n_windows,
            "rmse_all_horizons": rmse,
            "mae_all_horizons": float(np.mean(mae_vals)),
            "rmse_50ms": float(curve[4]),
            "rmse_100ms": float(curve[9]),
            "rmse_200ms": float(curve[19]),
            "skill": "" if persist is None else skill_score(rmse, persist),
            "fold_id": extra.get("fold_id", ""),
            "notes": extra.get("notes", "mean over three training seeds"),
            "config_id": extra.get("config_id", ""),
            "seeds_mean": "YES",
        }
        for i, val in enumerate(curve, start=1):
            row[f"rmse_h{i:02d}"] = float(val)
        rows.append(row)
    return rows


def stage2_sequence_map(root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for name in (
        "WITHIN_EUROC_SEQUENCE_RESULTS.csv",
        "WITHIN_UZH_SEQUENCE_RESULTS.csv",
        "TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv",
        "TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv",
    ):
        for row in read_csv_rows(root / "results" / "stage2" / name):
            out[(row["evaluation_context"], row["sequence_id"], row["model"])] = row
    return out


def copy_stage2_row(row: dict[str, str], model_alias: str) -> dict[str, object]:
    copied: dict[str, object] = dict(row)
    copied["model"] = model_alias
    copied["config_id"] = row.get("model", "")
    copied["seeds_mean"] = "NA_STAGE2"
    copied["notes"] = "Stage-2 frozen baseline; not refit"
    return copied


def summarise_context(
    rows: list[dict[str, object]],
    model: str,
    context: str,
    ridge_by_seq: dict[str, float],
    persist_by_seq: dict[str, float],
    seed_sd: float,
    n_params: int,
) -> dict[str, object]:
    sub = [r for r in rows if r["model"] == model and r["evaluation_context"] == context]
    rmses = np.asarray([float(r["rmse_all_horizons"]) for r in sub], dtype=np.float64)
    maes = np.asarray([float(r["mae_all_horizons"]) for r in sub], dtype=np.float64)
    r50 = np.asarray([float(r["rmse_50ms"]) for r in sub], dtype=np.float64)
    r100 = np.asarray([float(r["rmse_100ms"]) for r in sub], dtype=np.float64)
    r200 = np.asarray([float(r["rmse_200ms"]) for r in sub], dtype=np.float64)
    boot = bootstrap_mean_ci(rmses)
    persist_rmses = np.asarray([persist_by_seq[str(r["sequence_id"])] for r in sub], dtype=np.float64)
    skill = float(1.0 - boot["mean"] / float(np.mean(persist_rmses))) if np.mean(persist_rmses) else float("nan")
    deltas = np.asarray(
        [float(r["rmse_all_horizons"]) - ridge_by_seq[str(r["sequence_id"])] for r in sub],
        dtype=np.float64,
    )
    wins, losses, ties = win_loss_tie(deltas)
    return {
        "context": context,
        "model": DISPLAY[model],
        "n_sequences": int(rmses.size),
        "mean_seq_rmse": boot["mean"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "median_seq_rmse": boot["median"],
        "mean_mae": float(np.mean(maes)),
        "rmse_50ms": float(np.mean(r50)),
        "rmse_100ms": float(np.mean(r100)),
        "rmse_200ms": float(np.mean(r200)),
        "skill_vs_persistence": skill,
        "delta_vs_ridge": float(np.mean(deltas)),
        "wins_vs_ridge": wins,
        "losses_vs_ridge": losses,
        "ties_vs_ridge": ties,
        "seed_sd": seed_sd,
        "parameters": n_params,
    }


def time_inference_ms(architecture: str, config: dict[str, Any], n_in: int) -> float:
    seed_everything(20260912)
    model = build_model(architecture, config, n_in=n_in)
    model.eval()
    x = torch.randn(64, 100, n_in)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        times = []
        for _ in range(40):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) / 64.0)
    return float(np.median(times) * 1000.0)


def classify_stage3(
    summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
) -> tuple[str, str, str]:
    """Pre-specified practical classification. Not rewritten after seeing numbers."""

    def ctx_row(ctx: str, model: str) -> dict[str, object] | None:
        hits = [r for r in summary if r["context"] == ctx and r["model"] == model]
        return hits[0] if hits else None

    def seq_deltas(ctx: str, model: str) -> np.ndarray:
        key = "TRANSFORMER" if model == "Transformer" else model
        vals = [float(r["delta_vs_ridge"]) for r in paired if r["evaluation_context"] == ctx and r["model"] == key]
        return np.asarray(vals, dtype=np.float64)

    def seed_sd(ctx: str, model: str) -> float:
        key = "TRANSFORMER" if model == "Transformer" else model
        hits = [r for r in seed_rows if r["context"] == ctx and r["model"] == key]
        if len(hits) < 2:
            return float("nan")
        return float(np.std([float(r["mean_seq_rmse"]) for r in hits], ddof=1))

    def within_gain(model: str, ctx: str) -> tuple[bool, bool]:
        row = ctx_row(ctx, model)
        d = seq_deltas(ctx, model)
        if row is None or d.size == 0:
            return False, False
        mean_imp = float(row["delta_vs_ridge"]) < 0
        wins = int(row["wins_vs_ridge"])
        losses = int(row["losses_vs_ridge"])
        majority = wins > losses and wins >= int(np.ceil(d.size / 2.0))
        if d.size >= 3:
            remain = np.delete(d, int(np.argmin(d)))
            not_outlier = bool(np.mean(remain) < 0) or int(np.sum(remain < 0)) >= int(np.ceil(remain.size / 2.0))
        else:
            not_outlier = majority
        sd = seed_sd(ctx, model)
        vs_seed = True if not np.isfinite(sd) else abs(float(row["delta_vs_ridge"])) > sd
        practical = mean_imp and majority and not_outlier
        return practical, practical and vs_seed

    def transfer_ok(model: str, ctx: str) -> tuple[bool, bool]:
        row = ctx_row(ctx, model)
        persist = ctx_row(ctx, "Persistence")
        ridge = ctx_row(ctx, "Best Ridge")
        if row is None or persist is None or ridge is None:
            return False, True
        improved = float(row["delta_vs_ridge"]) < 0 and int(row["wins_vs_ridge"]) > int(row["losses_vs_ridge"])
        collapse = float(row["mean_seq_rmse"]) > 1.5 * float(persist["mean_seq_rmse"]) and float(row["mean_seq_rmse"]) > 1.25 * float(ridge["mean_seq_rmse"])
        return improved, not collapse

    candidates = [("TCN", "TCN"), ("GRU", "GRU"), ("TRANSFORMER", "Transformer")]
    any_clear = False
    any_domain = False
    any_unstable = False
    any_collapse = False
    for arch_name, disp_name in candidates:
        e_ok, e_seed = within_gain(disp_name, "WITHIN_EUROC")
        u_ok, u_seed = within_gain(disp_name, "WITHIN_UZH")
        t1_imp, t1_ok = transfer_ok(disp_name, "EUROC_TO_UZH")
        t2_imp, t2_ok = transfer_ok(disp_name, "UZH_TO_EUROC")
        if not t1_ok or not t2_ok:
            any_collapse = True
        within_both = e_ok and u_ok
        within_any = e_ok or u_ok
        seed_ok = e_seed or u_seed
        if within_both and (t1_imp or t2_imp) and t1_ok and t2_ok and (e_seed and u_seed):
            any_clear = True
        if within_any and not (t1_imp or t2_imp):
            any_domain = True
        if within_any and not seed_ok:
            any_unstable = True
    if any_collapse and not any_clear:
        label = "ADVANCED_MODELS_UNSTABLE" if any_unstable else "RIDGE_REMAINS_COMPETITIVE"
        if any_collapse and any_domain:
            label = "ADVANCED_MODEL_DOMAIN_SPECIFIC_GAIN"
    elif any_clear:
        label = "ADVANCED_MODEL_CLEAR_GAIN"
    elif any_unstable and any_domain:
        label = "ADVANCED_MODELS_UNSTABLE"
    elif any_domain:
        label = "ADVANCED_MODEL_DOMAIN_SPECIFIC_GAIN"
    else:
        label = "RIDGE_REMAINS_COMPETITIVE"
    paper = {
        "ADVANCED_MODEL_CLEAR_GAIN": "OPTION 1: Frame-invariant cross-UAV telemetry forecasting",
        "ADVANCED_MODEL_DOMAIN_SPECIFIC_GAIN": "OPTION 3: Complexity-generalization trade-off in cross-UAV telemetry forecasting",
        "RIDGE_REMAINS_COMPETITIVE": "OPTION 3: Complexity-generalization trade-off in cross-UAV telemetry forecasting",
        "ADVANCED_MODELS_UNSTABLE": "OPTION 2: Cross-domain evaluation protocol for inertial forecasting",
    }[label]
    decision = "PASS"
    if any_collapse and label == "ADVANCED_MODELS_UNSTABLE":
        decision = "CONDITIONAL_PASS"
    return label, paper, decision


def write_freeze_md(
    path: Path,
    *,
    utc: str,
    grid_hash: str,
    yaml_hash: str,
    selected: dict[str, dict[str, dict[str, Any]]],
    cfg_lookup: dict[str, dict[str, Any]],
    param_lookup: dict[str, int],
    sequences: dict[str, list[str]],
) -> None:
    lines = [
        "# Source models frozen (Phase 3A)",
        "",
        f"utc: {utc}",
        f"configs/stage3_advanced.yaml sha256: {yaml_hash}",
        f"results/stage3/HYPERPARAMETER_GRID.csv sha256: {grid_hash}",
        "",
        "Do not modify this file after Phase 3B begins.",
        "Hyperparameters below were selected with SOURCE-DOMAIN validation only.",
        "Target datasets were not accessed during architecture choice, scaling, or early stopping.",
        "",
        "Training seed policy: evaluate each selected configuration at seeds 20260912, 20260913, 20260914.",
        "Hyperparameter selection used the mean source-validation equal-sequence RMSE over the three seeds.",
        "Final source fits train on all source sequences for the median best source-validation epoch.",
        "Minibatches sample a recording uniformly, then a valid window from that recording.",
        "Early-stopping source-validation MSE uses a deterministic strided subset of up to 96 windows per validation sequence;",
        "configuration ranking uses full-window equal-sequence RMSE in m/s^2.",
        "Scaler: equal-recording-weighted standard scaler on features and targets, source-train windows only.",
        "Reported metrics are inverse-transformed to m/s^2.",
        "",
    ]
    for source in ("EUROC", "UZH_FPV"):
        lines.append(f"## {source}")
        lines.append("")
        lines.append(f"- source sequences: {', '.join(sequences[source])}")
        lines.append("- validation methodology: leave-trajectory-group-out on source recordings")
        lines.append("- scaler source: all source sequences of this dataset (final freeze fit)")
        lines.append("")
        for arch in ARCHS:
            sel = selected[source][arch]
            cfg = cfg_lookup[sel["config_id"]]
            lines.append(f"### {arch}  `{sel['config_id']}`")
            lines.append("")
            lines.append(f"- architecture: {arch}")
            lines.append("- input channels: q_f, q_w (n_in=2)")
            for key in (
                "channels",
                "kernel_size",
                "dilations",
                "activation",
                "hidden_size",
                "n_layers",
                "d_model",
                "n_heads",
                "ff_multiplier",
                "pooling",
                "dropout",
                "lr",
                "weight_decay",
            ):
                if key in cfg and cfg[key] != "" and cfg[key] is not None:
                    lines.append(f"- {key}: {cfg[key]}")
            lines.append(f"- selected_epochs: {sel['selected_epochs']}")
            lines.append(f"- source-validation mean RMSE (m/s^2): {sel['score']:.10g}")
            lines.append(f"- parameter count: {param_lookup[sel['config_id']]}")
            lines.append(f"- checkpoint pattern: artifacts/stage3/{source}_{arch}_seed{{seed}}.pt")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_report(ctx: dict[str, Any]) -> str:
    s = ctx["summary_rows"]
    def grab(context: str, model: str) -> dict[str, object]:
        hits = [r for r in s if r["context"] == context and r["model"] == model]
        return hits[0]

    def fmt(row: dict[str, object]) -> str:
        return (
            f"mean RMSE {float(row['mean_seq_rmse']):.4f} "
            f"(CI {float(row['ci_low']):.4f}–{float(row['ci_high']):.4f}); "
            f"50/100/200 ms = {float(row['rmse_50ms']):.4f}/{float(row['rmse_100ms']):.4f}/{float(row['rmse_200ms']):.4f}"
        )

    lines = [
        "# STAGE 3 REPORT",
        "",
        f"Protocol verified: {ctx['protocol_match']}",
        f"PROTOCOL_LOCK.md sha256: {ctx['protocol_md']}",
        f"configs/protocol_lock.yaml sha256: {ctx['protocol_yaml']}",
        f"Stage-2 SOURCE_MODELS_FROZEN.md sha256: {ctx['stage2_freeze']}",
        f"HYPERPARAMETER_GRID.csv sha256 (pre-training): {ctx['grid_hash']}",
        f"configs/stage3_advanced.yaml sha256: {ctx['yaml_hash']}",
        "",
        "Models:",
        "TCN",
        "GRU",
        "Transformer",
        "",
        "Input:",
        "Primary `[q_f, q_w]` history of 100 samples at 100 Hz. Direct 20-step `q_f` head. No recursive rollout.",
        "No derivatives, moving averages, spectral features, GT, dataset, or environment labels.",
        "",
        "Target:",
        "`q_f(t) = ||f(t)||_2` (m/s^2), accelerometer specific-force magnitude.",
        "",
        "Parameter counts:",
        ctx["param_text"],
        "",
        "Selected EuRoC configurations:",
        ctx["euroc_sel_text"],
        "",
        "Selected UZH configurations:",
        ctx["uzh_sel_text"],
        "",
        "WITHIN EUROC:",
        f"- Persistence: {fmt(grab('WITHIN_EUROC', 'Persistence'))}",
        f"- Best Ridge: {fmt(grab('WITHIN_EUROC', 'Best Ridge'))}",
        f"- TCN: {fmt(grab('WITHIN_EUROC', 'TCN'))}",
        f"- GRU: {fmt(grab('WITHIN_EUROC', 'GRU'))}",
        f"- Transformer: {fmt(grab('WITHIN_EUROC', 'Transformer'))}",
        "",
        "WITHIN UZH:",
        f"- Persistence: {fmt(grab('WITHIN_UZH', 'Persistence'))}",
        f"- Best Ridge: {fmt(grab('WITHIN_UZH', 'Best Ridge'))}",
        f"- TCN: {fmt(grab('WITHIN_UZH', 'TCN'))}",
        f"- GRU: {fmt(grab('WITHIN_UZH', 'GRU'))}",
        f"- Transformer: {fmt(grab('WITHIN_UZH', 'Transformer'))}",
        "",
        "EUROC -> UZH:",
        f"- Persistence: {fmt(grab('EUROC_TO_UZH', 'Persistence'))}",
        f"- Best Ridge: {fmt(grab('EUROC_TO_UZH', 'Best Ridge'))}",
        f"- TCN: {fmt(grab('EUROC_TO_UZH', 'TCN'))}",
        f"- GRU: {fmt(grab('EUROC_TO_UZH', 'GRU'))}",
        f"- Transformer: {fmt(grab('EUROC_TO_UZH', 'Transformer'))}",
        "",
        "UZH -> EUROC:",
        f"- Persistence: {fmt(grab('UZH_TO_EUROC', 'Persistence'))}",
        f"- Best Ridge: {fmt(grab('UZH_TO_EUROC', 'Best Ridge'))}",
        f"- TCN: {fmt(grab('UZH_TO_EUROC', 'TCN'))}",
        f"- GRU: {fmt(grab('UZH_TO_EUROC', 'GRU'))}",
        f"- Transformer: {fmt(grab('UZH_TO_EUROC', 'Transformer'))}",
        "",
        f"Best advanced model: {ctx['best_advanced']}",
        "",
        "Improvement over Ridge:",
        ctx["improv_text"],
        "",
        "50 ms:",
        ctx["h50_text"],
        "",
        "100 ms:",
        ctx["h100_text"],
        "",
        "200 ms:",
        ctx["h200_text"],
        "",
        "q_w ablation:",
        ctx["ablation_text"],
        "",
        "Seed variability:",
        ctx["seed_text"],
        "",
        "High-specific-force results:",
        ctx["highqf_text"],
        "",
        "Transfer gaps:",
        ctx["gap_text"],
        "",
        "Computational cost:",
        ctx["cost_text"],
        "",
        f"Target-fitted objects: {ctx['n_target_fitted']}",
        "",
        f"Source freeze hash: {ctx['freeze_hash']}",
        "",
        f"Freeze preserved: {ctx['freeze_preserved']}",
        "",
        "Critical findings:",
        ctx["critical_text"],
        "",
        "Major findings:",
        ctx["major_text"],
        "",
        "pytest:",
        "SEE_PYTEST passed, SEE_PYTEST failed",
        "",
        f"STAGE-3 SCIENTIFIC CLASSIFICATION: {ctx['classification']}",
        "",
        f"RECOMMENDED PAPER DIRECTION: {ctx['paper']}",
        "",
        f"OVERALL STAGE-3 DECISION: {ctx['decision']}",
        "",
        "Environment:",
        ctx["env_text"],
        "",
        "This stage does not write a manuscript.",
        "",
    ]
    return "\n".join(lines)


def run_stage3(root: Path) -> dict[str, Any]:
    configure_worker_threads(1)
    lock = verify_stage3_prerequisites(root)
    if lock["match"] != "PASS":
        print("PROTOCOL_LOCK_MISMATCH", lock, flush=True)
        return {"decision": "PROTOCOL_LOCK_MISMATCH", **lock}

    yaml_cfg = load_stage3_yaml(root)
    out_dir = root / "results" / "stage3"
    fig_dir = out_dir / "figures"
    art_dir = root / "artifacts" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    grid_path = write_hyperparameter_grid(root, yaml_cfg)
    grid_hash = sha256_file(grid_path)
    yaml_hash = sha256_file(root / "configs" / "stage3_advanced.yaml")
    (out_dir / "HYPERPARAMETER_GRID.sha256").write_text(grid_hash + "\n", encoding="utf-8")
    (out_dir / "STAGE3_YAML.sha256").write_text(yaml_hash + "\n", encoding="utf-8")
    print(f"Hashed hyperparameter grid {grid_hash}", flush=True)

    cfg_lookup = config_by_id(yaml_cfg)
    all_configs = flatten_configs(yaml_cfg)
    param_rows: list[dict[str, Any]] = []
    param_lookup: dict[str, int] = {}
    for config in all_configs:
        for n_in, tag in ((2, "qf_qw"), (1, "qf_only")):
            model = build_model(str(config["architecture"]), config, n_in=n_in)
            n_params = count_parameters(model)
            budget = int(yaml_cfg["param_budget"][config["architecture"]])
            param_rows.append(
                {
                    "architecture": config["architecture"],
                    "config_id": config["config_id"],
                    "input": tag,
                    "n_in": n_in,
                    "trainable_parameters": n_params,
                    "budget": budget,
                    "within_budget": "YES" if n_params <= budget else "NO",
                    "approx_state_bytes": n_params * 4,
                }
            )
            if n_in == 2:
                param_lookup[str(config["config_id"])] = n_params
                if n_params > budget:
                    raise RuntimeError(f"{config['config_id']} has {n_params} params > budget {budget}")
    write_csv(
        out_dir / "MODEL_PARAMETER_COUNTS.csv",
        ["architecture", "config_id", "input", "n_in", "trainable_parameters", "budget", "within_budget", "approx_state_bytes"],
        param_rows,
    )

    n_workers = min(int(yaml_cfg.get("n_workers", 8)), os.cpu_count() or 1)
    packs = packs_from_root(root)
    euroc_ids = seqs_for("EUROC")
    uzh_ids = seqs_for("UZH_FPV")
    configs_by_arch = {arch: [c for c in all_configs if c["architecture"] == arch] for arch in ARCHS}

    hp_jobs: list[dict[str, Any]] = []
    for arch in ARCHS:
        hp_jobs.extend(
            make_lgo_jobs(
                purpose="SOURCE_LGO",
                scope="EUROC_ALL",
                dataset="EUROC",
                seq_ids=euroc_ids,
                groups=EUROC_ROOM,
                configs=configs_by_arch[arch],
                seeds=SEEDS,
                use_qw=True,
                yaml_cfg=yaml_cfg,
            )
        )
        hp_jobs.extend(
            make_lgo_jobs(
                purpose="SOURCE_LGO",
                scope="UZH_ALL",
                dataset="UZH_FPV",
                seq_ids=uzh_ids,
                groups=UZH_GROUP,
                configs=configs_by_arch[arch],
                seeds=SEEDS,
                use_qw=True,
                yaml_cfg=yaml_cfg,
            )
        )
        for held in euroc_ids:
            remain = [s for s in euroc_ids if s != held]
            hp_jobs.extend(
                make_lgo_jobs(
                    purpose="NESTED_LORO",
                    scope=f"EUROC_HOLD_{held}",
                    dataset="EUROC",
                    seq_ids=remain,
                    groups=EUROC_ROOM,
                    configs=configs_by_arch[arch],
                    seeds=SEEDS,
                    use_qw=True,
                    yaml_cfg=yaml_cfg,
                    forbidden=[held],
                )
            )
        for held in uzh_ids:
            remain = [s for s in uzh_ids if s != held]
            hp_jobs.extend(
                make_lgo_jobs(
                    purpose="NESTED_LORO",
                    scope=f"UZH_HOLD_{held}",
                    dataset="UZH_FPV",
                    seq_ids=remain,
                    groups=UZH_GROUP,
                    configs=configs_by_arch[arch],
                    seeds=SEEDS,
                    use_qw=True,
                    yaml_cfg=yaml_cfg,
                    forbidden=[held],
                )
            )

    hp_records = run_hp_jobs(hp_jobs, packs, n_workers=n_workers, jsonl_path=out_dir / "_hp_jobs.jsonl")
    source_val_rows = []
    for rec in hp_records:
        for sid, rmse in rec["rmse_by_seq"].items():
            source_val_rows.append(
                {
                    "purpose": rec["purpose"],
                    "scope": rec["scope"],
                    "architecture": rec["architecture"],
                    "config_id": rec["config_id"],
                    "seed": rec["seed"],
                    "dataset": rec["dataset"],
                    "val_sequence": sid,
                    "train_sequences": ",".join(rec["train_ids"]),
                    "rmse_all_horizons": rmse,
                    "best_epoch": rec["best_epoch"],
                    "n_params": rec["n_params"],
                    "train_seconds": rec["train_seconds"],
                    "target_dataset_access_during_fit": rec["target_dataset_access_during_fit"],
                }
            )
    write_csv(
        out_dir / "SOURCE_VALIDATION_RESULTS.csv",
        [
            "purpose",
            "scope",
            "architecture",
            "config_id",
            "seed",
            "dataset",
            "val_sequence",
            "train_sequences",
            "rmse_all_horizons",
            "best_epoch",
            "n_params",
            "train_seconds",
            "target_dataset_access_during_fit",
        ],
        source_val_rows,
    )

    selected_source: dict[str, dict[str, dict[str, Any]]] = {"EUROC": {}, "UZH_FPV": {}}
    selected_rows: list[dict[str, Any]] = []
    for source, scope in (("EUROC", "EUROC_ALL"), ("UZH_FPV", "UZH_ALL")):
        for arch in ARCHS:
            recs = [r for r in hp_records if r["purpose"] == "SOURCE_LGO" and r["scope"] == scope and r["architecture"] == arch]
            sel = select_config(recs, configs_by_arch[arch], param_lookup)
            selected_source[source][arch] = sel
            selected_rows.append(
                {
                    "selection_level": "SOURCE_FREEZE",
                    "source_dataset": source,
                    "held_out_sequence": "",
                    "architecture": arch,
                    "config_id": sel["config_id"],
                    "mean_seed_val_rmse": sel["score"],
                    "selected_epochs": sel["selected_epochs"],
                    "n_params": sel["n_params"],
                    "seed_scores": ",".join(f"{x:.10g}" for x in sel["seed_scores"]),
                }
            )

    nested_sel: dict[tuple[str, str, str], dict[str, Any]] = {}
    for held in euroc_ids:
        for arch in ARCHS:
            recs = [r for r in hp_records if r["scope"] == f"EUROC_HOLD_{held}" and r["architecture"] == arch]
            sel = select_config(recs, configs_by_arch[arch], param_lookup)
            nested_sel[("EUROC", held, arch)] = sel
            selected_rows.append(
                {
                    "selection_level": "LORO_FOLD",
                    "source_dataset": "EUROC",
                    "held_out_sequence": held,
                    "architecture": arch,
                    "config_id": sel["config_id"],
                    "mean_seed_val_rmse": sel["score"],
                    "selected_epochs": sel["selected_epochs"],
                    "n_params": sel["n_params"],
                    "seed_scores": ",".join(f"{x:.10g}" for x in sel["seed_scores"]),
                }
            )
    for held in uzh_ids:
        for arch in ARCHS:
            recs = [r for r in hp_records if r["scope"] == f"UZH_HOLD_{held}" and r["architecture"] == arch]
            sel = select_config(recs, configs_by_arch[arch], param_lookup)
            nested_sel[("UZH_FPV", held, arch)] = sel
            selected_rows.append(
                {
                    "selection_level": "LORO_FOLD",
                    "source_dataset": "UZH_FPV",
                    "held_out_sequence": held,
                    "architecture": arch,
                    "config_id": sel["config_id"],
                    "mean_seed_val_rmse": sel["score"],
                    "selected_epochs": sel["selected_epochs"],
                    "n_params": sel["n_params"],
                    "seed_scores": ",".join(f"{x:.10g}" for x in sel["seed_scores"]),
                }
            )
    write_csv(
        out_dir / "SELECTED_MODELS.csv",
        [
            "selection_level",
            "source_dataset",
            "held_out_sequence",
            "architecture",
            "config_id",
            "mean_seed_val_rmse",
            "selected_epochs",
            "n_params",
            "seed_scores",
        ],
        selected_rows,
    )

    kw = common_train_kwargs(yaml_cfg)
    final_jobs: list[dict[str, Any]] = []
    for held in euroc_ids:
        remain = [s for s in euroc_ids if s != held]
        for arch in ARCHS:
            sel = nested_sel[("EUROC", held, arch)]
            cfg = cfg_lookup[sel["config_id"]]
            for seed in SEEDS:
                final_jobs.append(
                    {
                        "job_id": f"LORO|EUROC|{held}|{arch}|seed{seed}",
                        "purpose": "WITHIN_LORO",
                        "scope": held,
                        "architecture": arch,
                        "config": cfg,
                        "seed": int(seed),
                        "dataset": "EUROC",
                        "train_ids": remain,
                        "eval_ids": [held],
                        "use_qw": True,
                        "forbidden_sequences": [held],
                        "fixed_epochs": int(sel["selected_epochs"]),
                        "save_path": "",
                        "return_predictions": True,
                        **kw,
                    }
                )
    for held in uzh_ids:
        remain = [s for s in uzh_ids if s != held]
        for arch in ARCHS:
            sel = nested_sel[("UZH_FPV", held, arch)]
            cfg = cfg_lookup[sel["config_id"]]
            for seed in SEEDS:
                final_jobs.append(
                    {
                        "job_id": f"LORO|UZH|{held}|{arch}|seed{seed}",
                        "purpose": "WITHIN_LORO",
                        "scope": held,
                        "architecture": arch,
                        "config": cfg,
                        "seed": int(seed),
                        "dataset": "UZH_FPV",
                        "train_ids": remain,
                        "eval_ids": [held],
                        "use_qw": True,
                        "forbidden_sequences": [held],
                        "fixed_epochs": int(sel["selected_epochs"]),
                        "save_path": "",
                        "return_predictions": True,
                        **kw,
                    }
                )

    freeze_jobs: list[dict[str, Any]] = []
    for source, seqs, dataset_name in (("EUROC", euroc_ids, "EUROC"), ("UZH_FPV", uzh_ids, "UZH_FPV")):
        for arch in ARCHS:
            sel = selected_source[source][arch]
            cfg = cfg_lookup[sel["config_id"]]
            for seed in SEEDS:
                save_path = str(art_dir / f"{source}_{arch}_seed{seed}.pt")
                freeze_jobs.append(
                    {
                        "job_id": f"FREEZE|{source}|{arch}|seed{seed}",
                        "purpose": "SOURCE_FREEZE_FIT",
                        "scope": source,
                        "architecture": arch,
                        "config": cfg,
                        "seed": int(seed),
                        "dataset": dataset_name,
                        "train_ids": seqs,
                        "eval_ids": [],
                        "use_qw": True,
                        "forbidden_sequences": [],
                        "fixed_epochs": int(sel["selected_epochs"]),
                        "save_path": save_path,
                        "return_predictions": False,
                        **kw,
                    }
                )

    print("Phase 3A final LORO fits", flush=True)
    loro_results = run_final_jobs(final_jobs, packs, n_workers=n_workers)
    print("Phase 3A freeze fits (no target evaluation)", flush=True)
    freeze_results = run_final_jobs(freeze_jobs, packs, n_workers=n_workers)

    best_arch_by_source: dict[str, str] = {}
    for source in ("EUROC", "UZH_FPV"):
        best_arch_by_source[source] = min(ARCHS, key=lambda a: selected_source[source][a]["score"])
    ablation_jobs: list[dict[str, Any]] = []
    for source, seqs, groups, dataset_name in (
        ("EUROC", euroc_ids, EUROC_ROOM, "EUROC"),
        ("UZH_FPV", uzh_ids, UZH_GROUP, "UZH_FPV"),
    ):
        arch = best_arch_by_source[source]
        sel = selected_source[source][arch]
        cfg = cfg_lookup[sel["config_id"]]
        for use_qw in (False, True):
            ablation_jobs.extend(
                make_lgo_jobs(
                    purpose="QW_ABLATION",
                    scope=f"{source}_{arch}_qw{int(use_qw)}",
                    dataset=dataset_name,
                    seq_ids=seqs,
                    groups=groups,
                    configs=[cfg],
                    seeds=SEEDS,
                    use_qw=use_qw,
                    yaml_cfg=yaml_cfg,
                )
            )
    print("Phase 3A q_w ablation (source validation only)", flush=True)
    ablation_records = run_hp_jobs(ablation_jobs, packs, n_workers=n_workers, jsonl_path=out_dir / "_ablation_jobs.jsonl")

    freeze_path = out_dir / "STAGE3_SOURCE_MODELS_FROZEN.md"
    write_freeze_md(
        freeze_path,
        utc=utc_now(),
        grid_hash=grid_hash,
        yaml_hash=yaml_hash,
        selected=selected_source,
        cfg_lookup=cfg_lookup,
        param_lookup=param_lookup,
        sequences={"EUROC": euroc_ids, "UZH_FPV": uzh_ids},
    )
    freeze_hash = sha256_file(freeze_path)
    (out_dir / "STAGE3_SOURCE_MODELS_FROZEN.sha256").write_text(freeze_hash + "\n", encoding="utf-8")
    print(f"Phase 3A freeze hash {freeze_hash}", flush=True)

    if sha256_file(grid_path) != grid_hash:
        raise RuntimeError("HYPERPARAMETER_GRID.csv changed after hashing")

    print("Phase 3B zero-shot transfer (frozen checkpoints only)", flush=True)
    freeze_hash_before_b = freeze_hash
    s2map = stage2_sequence_map(root)
    persist_rmse: dict[tuple[str, str], float] = {}
    ridge_rmse: dict[tuple[str, str], float] = {}
    baseline_rows: list[dict[str, object]] = []
    for context, dataset, seqs in (
        ("WITHIN_EUROC", "EUROC", euroc_ids),
        ("WITHIN_UZH", "UZH_FPV", uzh_ids),
        ("EUROC_TO_UZH", "UZH_FPV", uzh_ids),
        ("UZH_TO_EUROC", "EUROC", euroc_ids),
    ):
        ridge_name = BEST_RIDGE[context]
        for sid in seqs:
            p = s2map[(context, sid, "B0_PERSISTENCE")]
            r = s2map[(context, sid, ridge_name)]
            persist_rmse[(context, sid)] = float(p["rmse_all_horizons"])
            ridge_rmse[(context, sid)] = float(r["rmse_all_horizons"])
            baseline_rows.append(copy_stage2_row(p, "B0_PERSISTENCE"))
            baseline_rows.append(copy_stage2_row(r, "BEST_RIDGE"))

    def persist_map(context: str) -> dict[str, float]:
        return {sid: persist_rmse[(context, sid)] for sid in {k[1] for k in persist_rmse if k[0] == context}}

    within_rows: list[dict[str, object]] = list(baseline_rows)
    seed_var_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, str]] = []
    high_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []

    for rec in freeze_results:
        leakage_rows.append(
            {
                "object_name": rec["job_id"],
                "model": rec["architecture"],
                "transfer_direction": "NONE_YET_SOURCE_FIT",
                "fit_dataset": rec["dataset"],
                "fit_sequences": ",".join(rec["train_ids"]),
                "validation_dataset": "",
                "validation_sequences": "",
                "target_dataset_access_during_fit": rec["target_dataset_access_during_fit"],
                "status": "PASS" if rec["target_dataset_access_during_fit"] == "NO" else "CRITICAL FAIL",
            }
        )
        cost_rows.append(
            {
                "model": rec["architecture"],
                "source_dataset": rec["dataset"],
                "seed": rec["seed"],
                "config_id": rec["config_id"],
                "parameters": rec["n_params"],
                "training_time_s": rec["train_seconds"],
                "artifact_path": rec["save_path"],
                "artifact_bytes": Path(rec["save_path"]).stat().st_size if rec["save_path"] else "",
            }
        )

    for rec in loro_results:
        leakage_rows.append(
            {
                "object_name": rec["job_id"],
                "model": rec["architecture"],
                "transfer_direction": "WITHIN_DOMAIN",
                "fit_dataset": rec["dataset"],
                "fit_sequences": ",".join(rec["train_ids"]),
                "validation_dataset": rec["dataset"],
                "validation_sequences": ",".join(rec["eval_ids"]),
                "target_dataset_access_during_fit": rec["target_dataset_access_during_fit"],
                "status": "PASS" if rec["target_dataset_access_during_fit"] == "NO" else "CRITICAL FAIL",
            }
        )

    def collect_loro(dataset: str, seqs: list[str], context: str) -> None:
        for arch in ARCHS:
            seed_preds: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
            seed_rmse_lists: dict[int, list[float]] = {seed: [] for seed in SEEDS}
            for rec in loro_results:
                if rec["architecture"] != arch or rec["dataset"] != dataset:
                    continue
                seed = int(rec["seed"])
                seed_preds[seed].update(rec["predictions"])
                seed_rmse_lists[seed].extend(rec["rmse_by_seq"].values())
            preds_list = [seed_preds[seed] for seed in SEEDS]
            extra = {"notes": "nested LORO; mean over seeds", "config_id": "NESTED"}
            within_rows.extend(
                average_seed_metrics(
                    preds_list,
                    packs,
                    seqs,
                    context=context,
                    dataset=dataset,
                    model=arch,
                    persistence=persist_map(context),
                    extra=extra,
                )
            )
            for seed in SEEDS:
                seed_var_rows.append(
                    {
                        "context": context,
                        "model": arch,
                        "seed": seed,
                        "mean_seq_rmse": float(np.mean(seed_rmse_lists[seed])),
                        "min_seq_rmse": float(np.min(seed_rmse_lists[seed])),
                        "max_seq_rmse": float(np.max(seed_rmse_lists[seed])),
                    }
                )
            for sid in seqs:
                high_vals = [high_target_rmse(seed_preds[seed][sid], packs[sid].future_qf, packs[sid].p95_qf) for seed in SEEDS]
                high_rows.append(
                    {
                        "evaluation_context": context,
                        "dataset": dataset,
                        "sequence_id": sid,
                        "model": arch,
                        "rmse_all": float(np.mean([rmse_all(seed_preds[seed][sid], packs[sid].future_qf) for seed in SEEDS])),
                        "rmse_high_p95_targets": float(np.nanmean(high_vals)),
                        "p95_threshold": packs[sid].p95_qf,
                        "notes": "sequence-specific p95 of processed q_f; mean over seeds",
                    }
                )

    collect_loro("EUROC", euroc_ids, "WITHIN_EUROC")
    collect_loro("UZH_FPV", uzh_ids, "WITHIN_UZH")

    transfer_specs = (
        ("EUROC_TO_UZH", "EUROC", "UZH_FPV", uzh_ids),
        ("UZH_TO_EUROC", "UZH_FPV", "EUROC", euroc_ids),
    )
    for context, source, target_ds, target_ids in transfer_specs:
        for arch in ARCHS:
            seed_preds: dict[int, dict[str, np.ndarray]] = {}
            seed_means: list[float] = []
            for seed in SEEDS:
                ckpt = art_dir / f"{source}_{arch}_seed{seed}.pt"
                preds, payload = predict_from_checkpoint(ckpt, packs, target_ids)
                seed_preds[seed] = preds
                seed_means.append(float(np.mean([rmse_all(preds[sid], packs[sid].future_qf) for sid in target_ids])))
                leakage_rows.append(
                    {
                        "object_name": f"TRANSFER_{context}_{arch}_seed{seed}",
                        "model": arch,
                        "transfer_direction": context,
                        "fit_dataset": str(payload["fit_dataset"]),
                        "fit_sequences": ",".join(payload["train_ids"]),
                        "validation_dataset": "",
                        "validation_sequences": "",
                        "target_dataset_access_during_fit": "NO",
                        "status": "PASS",
                    }
                )
            within_rows.extend(
                average_seed_metrics(
                    [seed_preds[seed] for seed in SEEDS],
                    packs,
                    target_ids,
                    context=context,
                    dataset=target_ds,
                    model=arch,
                    persistence=persist_map(context),
                    extra={"notes": "frozen source checkpoint; mean over seeds", "config_id": selected_source[source][arch]["config_id"]},
                )
            )
            for seed, mean_rmse in zip(SEEDS, seed_means, strict=True):
                seed_var_rows.append(
                    {
                        "context": context,
                        "model": arch,
                        "seed": seed,
                        "mean_seq_rmse": mean_rmse,
                        "min_seq_rmse": mean_rmse,
                        "max_seq_rmse": mean_rmse,
                    }
                )
            for sid in target_ids:
                high_vals = [high_target_rmse(seed_preds[seed][sid], packs[sid].future_qf, packs[sid].p95_qf) for seed in SEEDS]
                high_rows.append(
                    {
                        "evaluation_context": context,
                        "dataset": target_ds,
                        "sequence_id": sid,
                        "model": arch,
                        "rmse_all": float(np.mean([rmse_all(seed_preds[seed][sid], packs[sid].future_qf) for seed in SEEDS])),
                        "rmse_high_p95_targets": float(np.nanmean(high_vals)),
                        "p95_threshold": packs[sid].p95_qf,
                        "notes": "sequence-specific p95 of processed q_f; mean over seeds",
                    }
                )

    freeze_hash_after_b = sha256_file(freeze_path)
    freeze_preserved = "PASS" if freeze_hash_after_b == freeze_hash_before_b else "FAIL"
    if freeze_preserved != "PASS":
        raise RuntimeError("STAGE3 freeze hash changed during Phase 3B")

    for row in read_csv_rows(root / "results" / "stage2" / "HIGH_QF_SECONDARY.csv"):
        ctx = row["evaluation_context"]
        model = row["model"]
        alias = None
        if model == "B0_PERSISTENCE":
            alias = "B0_PERSISTENCE"
        elif model == BEST_RIDGE.get(ctx, ""):
            alias = "BEST_RIDGE"
        if alias is None:
            continue
        high_rows.append(
            {
                "evaluation_context": ctx,
                "dataset": row["dataset"],
                "sequence_id": row["sequence_id"],
                "model": alias,
                "rmse_all": row["rmse_all"],
                "rmse_high_p95_targets": row["rmse_high_p95_targets"],
                "p95_threshold": row["p95_threshold"],
                "notes": "Stage-2 frozen baseline; threshold reused",
            }
        )

    euroc_seq_rows = [r for r in within_rows if r["evaluation_context"] == "WITHIN_EUROC"]
    uzh_seq_rows = [r for r in within_rows if r["evaluation_context"] == "WITHIN_UZH"]
    t_eu_rows = [r for r in within_rows if r["evaluation_context"] == "EUROC_TO_UZH"]
    t_ue_rows = [r for r in within_rows if r["evaluation_context"] == "UZH_TO_EUROC"]
    write_csv(out_dir / "WITHIN_EUROC_ADVANCED.csv", SEQ_COLS, euroc_seq_rows)
    write_csv(out_dir / "WITHIN_UZH_ADVANCED.csv", SEQ_COLS, uzh_seq_rows)
    write_csv(out_dir / "TRANSFER_EUROC_TO_UZH_ADVANCED.csv", SEQ_COLS, t_eu_rows)
    write_csv(out_dir / "TRANSFER_UZH_TO_EUROC_ADVANCED.csv", SEQ_COLS, t_ue_rows)

    seed_lookup: dict[tuple[str, str], float] = {}
    for context in BEST_RIDGE:
        for model in ARCHS:
            vals = [float(r["mean_seq_rmse"]) for r in seed_var_rows if r["context"] == context and r["model"] == model]
            seed_lookup[(context, model)] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            seed_var_rows.append(
                {
                    "context": context,
                    "model": model,
                    "seed": "SD",
                    "mean_seq_rmse": seed_lookup[(context, model)],
                    "min_seq_rmse": min(vals) if vals else "",
                    "max_seq_rmse": max(vals) if vals else "",
                }
            )
    write_csv(
        out_dir / "SEED_VARIABILITY.csv",
        ["context", "model", "seed", "mean_seq_rmse", "min_seq_rmse", "max_seq_rmse"],
        seed_var_rows,
    )

    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    vs_ridge_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for context, seqs in (
        ("WITHIN_EUROC", euroc_ids),
        ("WITHIN_UZH", uzh_ids),
        ("EUROC_TO_UZH", uzh_ids),
        ("UZH_TO_EUROC", euroc_ids),
    ):
        ridge_by = {sid: ridge_rmse[(context, sid)] for sid in seqs}
        persist_by = {sid: persist_rmse[(context, sid)] for sid in seqs}
        for model, n_params in (
            ("B0_PERSISTENCE", 0),
            ("BEST_RIDGE", RIDGE_PARAMS[context]),
            ("TCN", param_lookup[selected_source["EUROC" if context in {"WITHIN_EUROC", "EUROC_TO_UZH"} else "UZH_FPV"]["TCN"]["config_id"]]),
            ("GRU", param_lookup[selected_source["EUROC" if context in {"WITHIN_EUROC", "EUROC_TO_UZH"} else "UZH_FPV"]["GRU"]["config_id"]]),
            ("TRANSFORMER", param_lookup[selected_source["EUROC" if context in {"WITHIN_EUROC", "EUROC_TO_UZH"} else "UZH_FPV"]["TRANSFORMER"]["config_id"]]),
        ):
            sd = 0.0 if model in {"B0_PERSISTENCE", "BEST_RIDGE"} else seed_lookup[(context, model)]
            summary_rows.append(summarise_context(within_rows, model, context, ridge_by, persist_by, sd, n_params))
            sub = [r for r in within_rows if r["model"] == model and r["evaluation_context"] == context]
            for h in range(1, 21):
                vals = [float(r[f"rmse_h{h:02d}"]) for r in sub]
                horizon_rows.append(
                    {
                        "evaluation_context": context,
                        "model": model,
                        "horizon_index": h,
                        "horizon_ms": h * 10,
                        "equal_sequence_mean_rmse": float(np.mean(vals)),
                    }
                )
        for model in ARCHS:
            sub = [r for r in within_rows if r["model"] == model and r["evaluation_context"] == context]
            deltas = np.asarray([float(r["rmse_all_horizons"]) - ridge_by[str(r["sequence_id"])] for r in sub], dtype=np.float64)
            boot = bootstrap_mean_ci(deltas)
            wins, losses, ties = win_loss_tie(deltas)
            vs_ridge_rows.append(
                {
                    "context": context,
                    "model": model,
                    "n_sequences": int(deltas.size),
                    "mean_paired_delta": boot["mean"],
                    "median_paired_delta": float(np.median(deltas)),
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "sign_flip_p_exploratory": paired_sign_flip_p(deltas),
                }
            )
            for r, d in zip(sub, deltas, strict=True):
                paired_rows.append(
                    {
                        "evaluation_context": context,
                        "sequence_id": r["sequence_id"],
                        "model": model,
                        "rmse_advanced": r["rmse_all_horizons"],
                        "rmse_ridge": ridge_by[str(r["sequence_id"])],
                        "delta_vs_ridge": float(d),
                    }
                )

    write_csv(
        out_dir / "ADVANCED_SUMMARY.csv",
        [
            "context",
            "model",
            "n_sequences",
            "mean_seq_rmse",
            "ci_low",
            "ci_high",
            "median_seq_rmse",
            "mean_mae",
            "rmse_50ms",
            "rmse_100ms",
            "rmse_200ms",
            "skill_vs_persistence",
            "delta_vs_ridge",
            "wins_vs_ridge",
            "losses_vs_ridge",
            "ties_vs_ridge",
            "seed_sd",
            "parameters",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / "ADVANCED_VS_RIDGE.csv",
        [
            "context",
            "model",
            "n_sequences",
            "mean_paired_delta",
            "median_paired_delta",
            "ci_low",
            "ci_high",
            "wins",
            "losses",
            "ties",
            "sign_flip_p_exploratory",
        ],
        vs_ridge_rows,
    )
    write_csv(
        out_dir / "HORIZON_ADVANCED.csv",
        ["evaluation_context", "model", "horizon_index", "horizon_ms", "equal_sequence_mean_rmse"],
        horizon_rows,
    )

    def mean_rmse(context: str, model: str) -> float:
        hits = [r for r in summary_rows if r["context"] == context and r["model"] == DISPLAY.get(model, model)]
        if not hits:
            hits = [r for r in summary_rows if r["context"] == context and r["model"] == model]
        return float(hits[0]["mean_seq_rmse"])

    for model, disp in (("BEST_RIDGE", "BEST_RIDGE"), ("TCN", "TCN"), ("GRU", "GRU"), ("TRANSFORMER", "TRANSFORMER")):
        for direction, within_ctx in (("EUROC_TO_UZH", "WITHIN_UZH"), ("UZH_TO_EUROC", "WITHIN_EUROC")):
            tr = mean_rmse(direction, DISPLAY[model])
            wi = mean_rmse(within_ctx, DISPLAY[model])
            gap = tr - wi
            gap_rows.append(
                {
                    "direction": direction,
                    "model": model,
                    "zero_shot_rmse": tr,
                    "within_target_rmse": wi,
                    "transfer_gap": gap,
                    "relative_transfer_gap": gap / wi if wi else float("nan"),
                }
            )
    write_csv(
        out_dir / "DOMAIN_TRANSFER_GAP.csv",
        ["direction", "model", "zero_shot_rmse", "within_target_rmse", "transfer_gap", "relative_transfer_gap"],
        gap_rows,
    )

    write_csv(
        out_dir / "HIGH_QF_ADVANCED.csv",
        ["evaluation_context", "dataset", "sequence_id", "model", "rmse_all", "rmse_high_p95_targets", "p95_threshold", "notes"],
        high_rows,
    )
    write_csv(
        out_dir / "TARGET_LEAKAGE_AUDIT.csv",
        [
            "object_name",
            "model",
            "transfer_direction",
            "fit_dataset",
            "fit_sequences",
            "validation_dataset",
            "validation_sequences",
            "target_dataset_access_during_fit",
            "status",
        ],
        leakage_rows,
    )

    ablation_summary: list[dict[str, object]] = []
    for source in ("EUROC", "UZH_FPV"):
        arch = best_arch_by_source[source]
        for use_qw in (False, True):
            recs = [r for r in ablation_records if r["scope"] == f"{source}_{arch}_qw{int(use_qw)}"]
            sel = select_config(recs, [cfg_lookup[selected_source[source][arch]["config_id"]]], param_lookup)
            ablation_summary.append(
                {
                    "source_dataset": source,
                    "architecture": arch,
                    "input": "q_f+q_w" if use_qw else "q_f",
                    "config_id": selected_source[source][arch]["config_id"],
                    "mean_source_val_rmse": sel["score"],
                    "selected_epochs_ref": selected_source[source][arch]["selected_epochs"],
                }
            )
    write_csv(
        out_dir / "QW_ABLATION.csv",
        ["source_dataset", "architecture", "input", "config_id", "mean_source_val_rmse", "selected_epochs_ref"],
        ablation_summary,
    )

    euroc_qf = np.concatenate([packs[s].q_f for s in euroc_ids])
    euroc_qw = np.concatenate([packs[s].q_w for s in euroc_ids])
    uzh_qf = np.concatenate([packs[s].q_f for s in uzh_ids])
    uzh_qw = np.concatenate([packs[s].q_w for s in uzh_ids])

    def dist_row(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, object]:
        return {
            "quantity": name,
            "euroc_median": float(np.median(a)),
            "euroc_iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
            "euroc_p05": float(np.percentile(a, 5)),
            "euroc_p95": float(np.percentile(a, 95)),
            "uzh_median": float(np.median(b)),
            "uzh_iqr": float(np.subtract(*np.percentile(b, [75, 25]))),
            "uzh_p05": float(np.percentile(b, 5)),
            "uzh_p95": float(np.percentile(b, 95)),
            "wasserstein": float(wasserstein_distance(a, b)),
            "notes": "descriptive only; not used for adaptation",
        }

    write_csv(
        out_dir / "DOMAIN_SHIFT_CHARACTERIZATION.csv",
        [
            "quantity",
            "euroc_median",
            "euroc_iqr",
            "euroc_p05",
            "euroc_p95",
            "uzh_median",
            "uzh_iqr",
            "uzh_p05",
            "uzh_p95",
            "wasserstein",
            "notes",
        ],
        [dist_row("q_f", euroc_qf, uzh_qf), dist_row("q_w", euroc_qw, uzh_qw)],
    )

    inf_rows = []
    for source in ("EUROC", "UZH_FPV"):
        for arch in ARCHS:
            cfg = cfg_lookup[selected_source[source][arch]["config_id"]]
            inf_ms = time_inference_ms(arch, cfg, 2)
            matches = [r for r in cost_rows if r["model"] == arch and r["source_dataset"] == ("EUROC" if source == "EUROC" else "UZH_FPV")]
            inf_rows.append(
                {
                    "model": arch,
                    "source_dataset": source,
                    "config_id": cfg["config_id"],
                    "parameters": param_lookup[cfg["config_id"]],
                    "mean_training_time_s": float(np.mean([float(r["training_time_s"]) for r in matches])),
                    "median_inference_ms_per_window": inf_ms,
                    "mean_artifact_bytes": float(np.mean([float(r["artifact_bytes"]) for r in matches if r["artifact_bytes"] != ""])),
                    "device": "cpu",
                    "notes": "computational footprint; not a real-time deployability claim",
                }
            )
    write_csv(
        out_dir / "COMPUTATIONAL_COST.csv",
        [
            "model",
            "source_dataset",
            "config_id",
            "parameters",
            "mean_training_time_s",
            "median_inference_ms_per_window",
            "mean_artifact_bytes",
            "device",
            "notes",
        ],
        inf_rows,
    )

    fig_summary = []
    for r in summary_rows:
        item = dict(r)
        item["model"] = {v: k for k, v in DISPLAY.items()}[str(r["model"])]
        fig_summary.append(item)
    figure_a_rmse_ci(fig_summary, pdf=fig_dir / "figure_a_rmse_ci.pdf", png=fig_dir / "figure_a_rmse_ci.png", csv_path=fig_dir / "figure_a_rmse_ci.csv")
    fig_h = []
    for r in horizon_rows:
        fig_h.append(dict(r))
    figure_b_horizon(fig_h, pdf=fig_dir / "figure_b_horizon.pdf", png=fig_dir / "figure_b_horizon.png", csv_path=fig_dir / "figure_b_horizon.csv")
    figure_c_paired(paired_rows, pdf=fig_dir / "figure_c_paired.pdf", png=fig_dir / "figure_c_paired.png", csv_path=fig_dir / "figure_c_paired.csv")
    figure_d_transfer_gap(gap_rows, pdf=fig_dir / "figure_d_transfer_gap.pdf", png=fig_dir / "figure_d_transfer_gap.png", csv_path=fig_dir / "figure_d_transfer_gap.csv")
    figure_e_high_qf(high_rows, pdf=fig_dir / "figure_e_high_qf.pdf", png=fig_dir / "figure_e_high_qf.png", csv_path=fig_dir / "figure_e_high_qf.csv")
    footprint = []
    for model in ("BEST_RIDGE", "TCN", "GRU", "TRANSFORMER"):
        we = mean_rmse("WITHIN_EUROC", DISPLAY[model])
        wu = mean_rmse("WITHIN_UZH", DISPLAY[model])
        params = 50 if model == "BEST_RIDGE" else param_lookup[selected_source["EUROC"][model if model != "BEST_RIDGE" else "TCN"]["config_id"]]
        if model == "BEST_RIDGE":
            params = 50
        else:
            params = param_lookup[selected_source["EUROC"][model]["config_id"]]
        footprint.append({"model": model, "parameters": params, "mean_within_rmse": 0.5 * (we + wu)})
    figure_f_footprint(footprint, pdf=fig_dir / "figure_f_footprint.pdf", png=fig_dir / "figure_f_footprint.png", csv_path=fig_dir / "figure_f_footprint.csv")

    n_target_fitted = sum(1 for r in leakage_rows if r["target_dataset_access_during_fit"] != "NO")
    classification, paper, decision = classify_stage3(summary_rows, paired_rows, [r for r in seed_var_rows if r["seed"] != "SD"])

    def rowc(ctx: str, model: str) -> dict[str, object]:
        return [r for r in summary_rows if r["context"] == ctx and r["model"] == model][0]

    adv_means = []
    for model in ("TCN", "GRU", "Transformer"):
        vals = [float(rowc(c, model)["delta_vs_ridge"]) for c in ("WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC")]
        adv_means.append((model, float(np.mean(vals))))
    best_advanced = min(adv_means, key=lambda t: t[1])[0]

    param_text = "\n".join(
        f"- {r['architecture']} {r['config_id']} n_in={r['n_in']}: {r['trainable_parameters']} params"
        for r in param_rows
        if r["n_in"] == 2
    )
    euroc_sel_text = "\n".join(
        f"- {arch}: {selected_source['EUROC'][arch]['config_id']} "
        f"(epochs={selected_source['EUROC'][arch]['selected_epochs']}, "
        f"val RMSE={selected_source['EUROC'][arch]['score']:.4f}, "
        f"params={selected_source['EUROC'][arch]['n_params']})"
        for arch in ARCHS
    )
    uzh_sel_text = "\n".join(
        f"- {arch}: {selected_source['UZH_FPV'][arch]['config_id']} "
        f"(epochs={selected_source['UZH_FPV'][arch]['selected_epochs']}, "
        f"val RMSE={selected_source['UZH_FPV'][arch]['score']:.4f}, "
        f"params={selected_source['UZH_FPV'][arch]['n_params']})"
        for arch in ARCHS
    )
    improv_lines = []
    for r in vs_ridge_rows:
        improv_lines.append(
            f"- {r['context']} {r['model']}: mean Δ={float(r['mean_paired_delta']):.4f} "
            f"(median {float(r['median_paired_delta']):.4f}), "
            f"CI [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}], "
            f"wins/losses/ties={r['wins']}/{r['losses']}/{r['ties']}"
        )
    h50 = "\n".join(f"- {c} {m}: {float(rowc(c, m)['rmse_50ms']):.4f}" for c in BEST_RIDGE for m in ("Persistence", "Best Ridge", "TCN", "GRU", "Transformer"))
    h100 = "\n".join(f"- {c} {m}: {float(rowc(c, m)['rmse_100ms']):.4f}" for c in BEST_RIDGE for m in ("Persistence", "Best Ridge", "TCN", "GRU", "Transformer"))
    h200 = "\n".join(f"- {c} {m}: {float(rowc(c, m)['rmse_200ms']):.4f}" for c in BEST_RIDGE for m in ("Persistence", "Best Ridge", "TCN", "GRU", "Transformer"))
    abl_text = "\n".join(
        f"- {r['source_dataset']} {r['architecture']} {r['input']}: source-val RMSE {float(r['mean_source_val_rmse']):.4f}"
        for r in ablation_summary
    )
    seed_text_lines = []
    for context in BEST_RIDGE:
        for model in ARCHS:
            vals = [float(r["mean_seq_rmse"]) for r in seed_var_rows if r["context"] == context and r["model"] == model and r["seed"] != "SD"]
            if not vals:
                continue
            seed_text_lines.append(
                f"- {context} {model}: mean={np.mean(vals):.4f} sd={np.std(vals, ddof=1):.4f} min={min(vals):.4f} max={max(vals):.4f}"
            )
    high_txt = []
    for ctx in BEST_RIDGE:
        for model in ("B0_PERSISTENCE", "BEST_RIDGE", "TCN", "GRU", "TRANSFORMER"):
            vals = [
                float(r["rmse_high_p95_targets"])
                for r in high_rows
                if r["evaluation_context"] == ctx and r["model"] == model and r["rmse_high_p95_targets"] != ""
            ]
            if vals:
                high_txt.append(f"- {ctx} {DISPLAY.get(model, model)}: mean high-q_f RMSE {float(np.mean(vals)):.4f}")
    gap_text = "\n".join(
        f"- {r['direction']} {r['model']}: gap={float(r['transfer_gap']):.4f} ({100*float(r['relative_transfer_gap']):.1f}% of within-target)"
        for r in gap_rows
    )
    cost_text = "\n".join(
        f"- {r['source_dataset']} {r['model']} {r['config_id']}: {r['parameters']} params, "
        f"train {float(r['mean_training_time_s']):.1f}s, "
        f"median inference {float(r['median_inference_ms_per_window']):.4f} ms/window, "
        f"artifact {float(r['mean_artifact_bytes']):.0f} bytes (computational footprint)"
        for r in inf_rows
    )
    critical = []
    if n_target_fitted:
        critical.append(f"- CRITICAL: {n_target_fitted} objects accessed target data during fit")
    else:
        critical.append("- None. Target-fitted objects = 0.")
    if freeze_preserved != "PASS":
        critical.append("- CRITICAL: freeze hash not preserved")
    major = [
        f"- Scientific classification: {classification}",
        f"- Best advanced model by mean Δ vs Ridge across four contexts: {best_advanced}",
        "- Sequence remains the inferential unit; three seeds are not extra experimental samples.",
        "- Principal Ridge comparator is the frozen Stage-2 winner in each context (B3 except UZH→EuRoC, which uses B2).",
    ]

    env_text = (
        f"- Python {sys.version.split()[0]}\n"
        f"- numpy {np.__version__}, torch {torch.__version__}\n"
        f"- platform {platform.platform()}\n"
        f"- processor {platform.processor()}\n"
        f"- CUDA: not used (CPU torch {torch.__version__})\n"
        "- Determinism: Python/NumPy/PyTorch RNGs seeded. CPU kernels are not claimed bitwise deterministic across hardware."
    )

    report_ctx = {
        "protocol_match": lock["match"],
        "protocol_md": lock["protocol_md"],
        "protocol_yaml": lock["protocol_yaml"],
        "stage2_freeze": lock["stage2_freeze"],
        "grid_hash": grid_hash,
        "yaml_hash": yaml_hash,
        "param_text": param_text,
        "euroc_sel_text": euroc_sel_text,
        "uzh_sel_text": uzh_sel_text,
        "best_advanced": best_advanced,
        "improv_text": "\n".join(improv_lines),
        "h50_text": h50,
        "h100_text": h100,
        "h200_text": h200,
        "ablation_text": abl_text,
        "seed_text": "\n".join(seed_text_lines),
        "highqf_text": "\n".join(high_txt),
        "gap_text": gap_text,
        "cost_text": cost_text,
        "n_target_fitted": n_target_fitted,
        "freeze_hash": freeze_hash_after_b,
        "freeze_preserved": freeze_preserved,
        "critical_text": "\n".join(critical),
        "major_text": "\n".join(major),
        "classification": classification,
        "paper": paper,
        "decision": decision,
        "env_text": env_text,
        "summary_rows": summary_rows,
    }
    report = build_report(report_ctx)
    write_report(root / "STAGE3_REPORT.md", report)
    write_report(out_dir / "STAGE3_REPORT.md", report)

    banner = {
        "decision": decision,
        "classification": classification,
        "paper": paper,
        "freeze_preserved": freeze_preserved,
        "n_target_fitted": n_target_fitted,
        "best_advanced": best_advanced,
        "summary_rows": summary_rows,
        "vs_ridge_rows": vs_ridge_rows,
        "ablation_summary": ablation_summary,
        "selected_source": selected_source,
        "seed_var_rows": seed_var_rows,
        "cost_rows": inf_rows,
        "grid_hash": grid_hash,
        "freeze_hash": freeze_hash_after_b,
        "protocol_match": lock["match"],
    }
    (root / ".stage3_complete").write_text(
        f"decision={decision}\nclassification={classification}\nfreeze={freeze_preserved}\n",
        encoding="utf-8",
    )
    return banner
