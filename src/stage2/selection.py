"""Source-only hyperparameter selection. Target data must never enter this module's scores."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import equal_sequence_mean, rmse_all
from .models import fit_ridge, ridge_predict
from .windows import SequenceWindows


def unique_groups(seq_ids: list[str], groups: dict[str, str]) -> list[str]:
    out: list[str] = []
    for sid in seq_ids:
        g = groups[sid]
        if g not in out:
            out.append(g)
    return out


def select_ridge_config(
    seq_ids: list[str],
    packs: dict[str, SequenceWindows],
    groups: dict[str, str],
    *,
    dataset: str,
    model_name: str,
    lags: list[int],
    alphas: list[float],
    use_qw: bool,
    tie_relative_tol: float,
    fit_log: list[dict[str, str]],
    context: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    group_names = unique_groups(seq_ids, groups)
    for lag in lags:
        for alpha in alphas:
            seq_rmse: list[float] = []
            seq_used: list[str] = []
            for g in group_names:
                train_ids = [s for s in seq_ids if groups[s] != g]
                val_ids = [s for s in seq_ids if groups[s] == g]
                if not train_ids or not val_ids:
                    continue
                fitted = fit_ridge(
                    [packs[s] for s in train_ids],
                    lag=lag,
                    alpha=alpha,
                    use_qw=use_qw,
                    dataset=dataset,
                    model_name=model_name,
                )
                fit_log.append(
                    {
                        "object": f"{model_name}_lag{lag}_alpha{alpha}_valgrp_{g}",
                        "object_type": "Ridge+SourceStandardScaler",
                        "fit_dataset": dataset,
                        "fit_sequences": ",".join(fitted.train_sequences),
                        "fit_purpose": "source_validation_selection",
                        "evaluation_context": context,
                    }
                )
                for sid in val_ids:
                    pred = ridge_predict(fitted, packs[sid])
                    seq_rmse.append(rmse_all(pred, packs[sid].future_qf))
                    seq_used.append(sid)
            score = equal_sequence_mean(seq_rmse)
            records.append(
                {
                    "lag": int(lag),
                    "alpha": float(alpha),
                    "score": float(score),
                    "n_val_sequences": len(seq_rmse),
                    "val_sequences": ",".join(seq_used),
                }
            )
    finite = [r for r in records if np.isfinite(r["score"])]
    if not finite:
        raise RuntimeError(f"no finite selection scores for {model_name} on {dataset}")
    best = min(r["score"] for r in finite)
    tied = [r for r in finite if r["score"] <= best * (1.0 + float(tie_relative_tol))]
    tied.sort(key=lambda r: (r["lag"], -r["alpha"]))
    chosen = tied[0]
    chosen["n_tied"] = len(tied)
    chosen["best_score"] = best
    chosen["all_records"] = records
    return chosen
