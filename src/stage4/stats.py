"""Recompute publication statistics from sequence-level files. No model fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

from stage2.bootstrap import bootstrap_mean_ci, win_loss_tie
from stage2.metrics import skill_score

from .io import ADV_MODELS, BOOT_REPS, BOOT_SEED, CONTEXTS, PUB_MODELS, fnum, sequence_ids, sequence_values


def summarize_model(rows: list[dict[str, Any]], context: str, model: str) -> dict[str, Any]:
    rmse = sequence_values(rows, context, model, "rmse_all_horizons")
    mae = sequence_values(rows, context, model, "mae_all_horizons")
    r50 = sequence_values(rows, context, model, "rmse_50ms")
    r100 = sequence_values(rows, context, model, "rmse_100ms")
    r200 = sequence_values(rows, context, model, "rmse_200ms")
    persist = sequence_values(rows, context, "Persistence", "rmse_all_horizons")
    boot = bootstrap_mean_ci(rmse, n_reps=BOOT_REPS, seed=BOOT_SEED)
    persist_mean = float(np.mean(persist)) if persist.size else float("nan")
    return {
        "context": context,
        "model": model,
        "n_sequences": int(rmse.size),
        "sequence_ids": sequence_ids(rows, context, model),
        "mean_seq_rmse": boot["mean"],
        "median_seq_rmse": boot["median"],
        "sd_seq_rmse": boot["std"],
        "min_seq_rmse": boot["min"],
        "max_seq_rmse": boot["max"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "mean_mae": float(np.mean(mae)) if mae.size else float("nan"),
        "rmse_50ms": float(np.mean(r50)) if r50.size else float("nan"),
        "rmse_100ms": float(np.mean(r100)) if r100.size else float("nan"),
        "rmse_200ms": float(np.mean(r200)) if r200.size else float("nan"),
        "skill_vs_persistence": skill_score(boot["mean"], persist_mean),
        "rmse_values": rmse,
    }


def paired_vs_ridge(rows: list[dict[str, Any]], context: str, model: str) -> dict[str, Any]:
    adv_map = {
        str(r["sequence_id"]): fnum(r["rmse_all_horizons"])
        for r in rows
        if r["evaluation_context"] == context and r.get("pub_model") == model
    }
    ridge_map = {
        str(r["sequence_id"]): fnum(r["rmse_all_horizons"])
        for r in rows
        if r["evaluation_context"] == context and r.get("pub_model") == "Best frozen Ridge"
    }
    seqs = [s for s in adv_map if s in ridge_map]
    delta = np.asarray([adv_map[s] - ridge_map[s] for s in seqs], dtype=np.float64)
    boot = bootstrap_mean_ci(delta, n_reps=BOOT_REPS, seed=BOOT_SEED)
    wins, losses, ties = win_loss_tie(delta)
    ridge_mean = float(np.mean([ridge_map[s] for s in seqs]))
    adv_mean = float(np.mean([adv_map[s] for s in seqs]))
    abs_imp = ridge_mean - adv_mean
    rel_imp = 100.0 * abs_imp / ridge_mean if ridge_mean else float("nan")
    return {
        "context": context,
        "model": model,
        "n_sequences": int(delta.size),
        "sequence_ids": seqs,
        "delta": delta,
        "mean_paired_delta": boot["mean"],
        "median_paired_delta": float(np.median(delta)) if delta.size else float("nan"),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "ridge_mean": ridge_mean,
        "advanced_mean": adv_mean,
        "absolute_improvement": abs_imp,
        "relative_improvement_pct": rel_imp,
    }


def evidence_class(paired: dict[str, Any]) -> str:
    mean_d = float(paired["mean_paired_delta"])
    lo = float(paired["ci_low"])
    hi = float(paired["ci_high"])
    wins = int(paired["wins"])
    losses = int(paired["losses"])
    n = int(paired["n_sequences"])
    rel = abs(float(paired["relative_improvement_pct"]))
    if mean_d > 0 and losses > wins:
        if rel < 1.0 and lo <= 0 <= hi:
            return "C. NO MATERIAL GAIN"
        return "D. DEGRADATION"
    if mean_d < 0 and wins >= int(np.ceil(n / 2.0)) and hi <= 0:
        return "A. CLEAR PRACTICAL GAIN"
    if mean_d < 0:
        return "B. MODEST / UNCERTAIN GAIN"
    if rel < 1.0:
        return "C. NO MATERIAL GAIN"
    return "C. NO MATERIAL GAIN"


def horizon_curve(rows: list[dict[str, Any]], context: str, model: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in range(1, 21):
        field = f"rmse_h{h:02d}"
        vals = sequence_values(rows, context, model, field)
        boot = bootstrap_mean_ci(vals, n_reps=BOOT_REPS, seed=BOOT_SEED)
        out.append(
            {
                "context": context,
                "model": model,
                "horizon_step": h,
                "horizon_ms": h * 10,
                "mean_sequence_rmse": boot["mean"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
            }
        )
    return out


def all_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [summarize_model(rows, ctx, m) for ctx in CONTEXTS for m in PUB_MODELS]


def all_paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [paired_vs_ridge(rows, ctx, m) for ctx in CONTEXTS for m in ADV_MODELS]
