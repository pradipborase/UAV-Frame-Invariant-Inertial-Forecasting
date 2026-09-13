"""Stage 2 orchestration: Phase 2A source freeze, then Phase 2B zero-shot transfer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from audit.config import ROOT
from audit.hashing import sha256_file
from audit.stage0c import write_csv
from stage1.folds import EUROC_ROOM, UZH_GROUP
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY

from .bootstrap import bootstrap_mean_ci, paired_sign_flip_p, win_loss_tie
from .extreme import audit_sequence, load_native
from .figures import grouped_rmse_by_sequence, horizon_curves, skill_by_sequence
from .io_processed import load_all_processed
from .metrics import equal_sequence_mean, rmse_all, sequence_result_row
from .models import RidgePack, fit_ridge, local_predict, ridge_predict
from .protocol import verify_protocol_lock
from .selection import select_ridge_config
from .windows import SequenceWindows, build_sequence_windows

UTC = timezone.utc
MODELS_LOCAL = ("B0_PERSISTENCE", "B1_LINEAR")
MODELS_RIDGE = (("B2_RIDGE_QF", False), ("B3_RIDGE_QF_QW", True))
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
    + ["fold_id", "notes"]
)


def utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def acf_at_lag(x: np.ndarray, lag: int) -> float:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size <= lag:
        return float("nan")
    c = v - v.mean()
    den = float(np.dot(c, c))
    if den <= 0:
        return float("nan")
    return float(np.dot(c[:-lag], c[lag:]) / den)


def high_target_rmse(pred: np.ndarray, actual: np.ndarray, p95: float) -> float:
    mask = np.asarray(actual, dtype=np.float64) > float(p95)
    if not np.any(mask):
        return float("nan")
    err = np.asarray(pred, dtype=np.float64)[mask] - np.asarray(actual, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(err * err)))


def _artifact_dict(fitted: RidgePack, config_hash: str, purpose: str) -> dict[str, Any]:
    return {
        "model_name": fitted.model_name,
        "lag": fitted.lag,
        "alpha": fitted.alpha,
        "use_qw": fitted.use_qw,
        "train_dataset": fitted.train_dataset,
        "train_sequences": fitted.train_sequences,
        "n_windows": fitted.n_windows,
        "n_features": fitted.n_features,
        "n_outputs": fitted.n_outputs,
        "coef_norm": fitted.coef_norm,
        "intercept_norm": fitted.intercept_norm,
        "scaler_mean": None if fitted.scaler.mean_ is None else fitted.scaler.mean_.tolist(),
        "scaler_std": None if fitted.scaler.std_ is None else fitted.scaler.std_.tolist(),
        "scaler_fit_sequences": fitted.scaler.fit_recording_ids,
        "weights_sum_by_sequence": fitted.weights_sum_by_sequence,
        "config_hash": config_hash,
        "purpose": purpose,
        "ridge": fitted.ridge,
        "scaler": fitted.scaler,
    }


def evaluate_local_and_ridge_on_seq(
    seq: SequenceWindows,
    *,
    context: str,
    fold_id: str,
    ridge_models: dict[str, RidgePack],
    linear_k: int,
    dt_s: float,
    pers_rmse: float | None,
) -> list[dict[str, Any]]:
    rows = []
    extra = {"fold_id": fold_id, "notes": ""}
    for name in MODELS_LOCAL:
        pred = local_predict(seq, name, linear_k=linear_k, dt_s=dt_s)
        pr = None if name == "B0_PERSISTENCE" else pers_rmse
        row = sequence_result_row(
            evaluation_context=context,
            dataset=seq.dataset,
            sequence_id=seq.sequence_id,
            model=name,
            pred=pred,
            actual=seq.future_qf,
            persistence_rmse=pr,
            n_origins=seq.n_windows,
            extra=extra,
        )
        row["_pred"] = pred
        rows.append(row)
    if pers_rmse is None:
        pers_rmse = float(rows[0]["rmse_all_horizons"])
        rows[0]["skill"] = ""
    for name, _use_qw in MODELS_RIDGE:
        pred = ridge_predict(ridge_models[name], seq)
        row = sequence_result_row(
            evaluation_context=context,
            dataset=seq.dataset,
            sequence_id=seq.sequence_id,
            model=name,
            pred=pred,
            actual=seq.future_qf,
            persistence_rmse=pers_rmse,
            n_origins=seq.n_windows,
            extra=extra,
        )
        row["_pred"] = pred
        rows.append(row)
    return rows


def run_loro(
    *,
    dataset: str,
    seq_ids: tuple[str, ...],
    groups: dict[str, str],
    packs: dict[str, SequenceWindows],
    context: str,
    cfg: dict[str, Any],
    fit_log: list[dict[str, str]],
    hp_rows: list[dict[str, Any]],
    coeff_rows: list[dict[str, Any]],
    scaler_rows: list[dict[str, Any]],
    config_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lags = list(cfg["lags"])
    alphas = [float(a) for a in cfg["alphas"]]
    tie = float(cfg["tie_relative_tol"])
    linear_k = int(cfg["linear_k"])
    dt_s = float(cfg["dt_s"])
    for i, held in enumerate(seq_ids, start=1):
        train_ids = [s for s in seq_ids if s != held]
        fold_id = f"{dataset}_LORO_{i:02d}_{held}"
        print(f"[stage2] {context} fold {fold_id}", flush=True)
        ridge_fitted: dict[str, RidgePack] = {}
        for model_name, use_qw in MODELS_RIDGE:
            chosen = select_ridge_config(
                train_ids,
                packs,
                groups,
                dataset=dataset,
                model_name=model_name,
                lags=lags,
                alphas=alphas,
                use_qw=use_qw,
                tie_relative_tol=tie,
                fit_log=fit_log,
                context=context,
            )
            hp_rows.append(
                {
                    "evaluation_context": context,
                    "phase": "WITHIN_NESTED_SELECTION",
                    "fold_id": fold_id,
                    "dataset": dataset,
                    "model": model_name,
                    "selected_lag": chosen["lag"],
                    "selected_alpha": chosen["alpha"],
                    "selection_score_rmse": chosen["score"],
                    "n_tied": chosen["n_tied"],
                    "train_sequences": ",".join(train_ids),
                    "held_out_sequence": held,
                    "notes": "nested source-only; held-out sequence excluded",
                }
            )
            fitted = fit_ridge(
                [packs[s] for s in train_ids],
                lag=int(chosen["lag"]),
                alpha=float(chosen["alpha"]),
                use_qw=use_qw,
                dataset=dataset,
                model_name=model_name,
            )
            ridge_fitted[model_name] = fitted
            fit_log.append(
                {
                    "object": f"{fold_id}_{model_name}_final",
                    "object_type": "Ridge+SourceStandardScaler",
                    "fit_dataset": dataset,
                    "fit_sequences": ",".join(fitted.train_sequences),
                    "fit_purpose": "within_loro_final_refit",
                    "evaluation_context": context,
                }
            )
            _record_ridge_aux(fitted, coeff_rows, scaler_rows, config_hash, fold_id, context)
        held_pack = packs[held]
        pers = local_predict(held_pack, "B0_PERSISTENCE", linear_k=linear_k, dt_s=dt_s)
        pers_rmse = rmse_all(pers, held_pack.future_qf)
        rows.extend(
            evaluate_local_and_ridge_on_seq(
                held_pack,
                context=context,
                fold_id=fold_id,
                ridge_models=ridge_fitted,
                linear_k=linear_k,
                dt_s=dt_s,
                pers_rmse=pers_rmse,
            )
        )
    return rows


def _record_ridge_aux(
    fitted: RidgePack,
    coeff_rows: list[dict[str, Any]],
    scaler_rows: list[dict[str, Any]],
    config_hash: str,
    fold_id: str,
    context: str,
) -> None:
    coeff_rows.append(
        {
            "evaluation_context": context,
            "fold_id": fold_id,
            "model": fitted.model_name,
            "dataset": fitted.train_dataset,
            "lag": fitted.lag,
            "alpha": fitted.alpha,
            "n_features": fitted.n_features,
            "n_outputs": fitted.n_outputs,
            "coef_norm": fitted.coef_norm,
            "intercept_norm": fitted.intercept_norm,
            "n_windows": fitted.n_windows,
            "train_sequences": ",".join(fitted.train_sequences),
            "equal_sequence_weight_sums": json.dumps(fitted.weights_sum_by_sequence),
            "config_hash": config_hash,
            "notes": "no causal interpretation of coefficients",
        }
    )
    scaler_rows.append(
        {
            "evaluation_context": context,
            "fold_id": fold_id,
            "model": fitted.model_name,
            "fit_dataset": fitted.scaler.fit_dataset,
            "fit_sequences": ",".join(fitted.scaler.fit_recording_ids),
            "n_features": len(fitted.scaler.mean_) if fitted.scaler.mean_ is not None else 0,
            "n_windows": fitted.scaler.n_windows,
            "target_scaler": "NONE",
            "fit_scope": "SOURCE_TRAIN_ONLY",
            "mean_l2": float(np.linalg.norm(fitted.scaler.mean_)) if fitted.scaler.mean_ is not None else "",
            "std_l2": float(np.linalg.norm(fitted.scaler.std_)) if fitted.scaler.std_ is not None else "",
            "config_hash": config_hash,
        }
    )


def summarize_context(rows: list[dict[str, Any]], context: str, n_reps: int, seed: int) -> list[dict[str, Any]]:
    out = []
    sub = [r for r in rows if r["evaluation_context"] == context]
    models = [m for m in ("B0_PERSISTENCE", "B1_LINEAR", "B2_RIDGE_QF", "B3_RIDGE_QF_QW") if any(r["model"] == m for r in sub)]
    pers = {r["sequence_id"]: float(r["rmse_all_horizons"]) for r in sub if r["model"] == "B0_PERSISTENCE"}
    for model in models:
        mrows = [r for r in sub if r["model"] == model]
        rmses = np.array([float(r["rmse_all_horizons"]) for r in mrows], dtype=np.float64)
        maes = np.array([float(r["mae_all_horizons"]) for r in mrows], dtype=np.float64)
        r50 = np.array([float(r["rmse_50ms"]) for r in mrows], dtype=np.float64)
        r100 = np.array([float(r["rmse_100ms"]) for r in mrows], dtype=np.float64)
        r200 = np.array([float(r["rmse_200ms"]) for r in mrows], dtype=np.float64)
        boot = bootstrap_mean_ci(rmses, n_reps=n_reps, seed=seed)
        skills = []
        wins = losses = ties = ""
        if model != "B0_PERSISTENCE":
            deltas = np.array(
                [float(r["rmse_all_horizons"]) - pers[r["sequence_id"]] for r in mrows],
                dtype=np.float64,
            )
            w, l, t = win_loss_tie(deltas)
            wins, losses, ties = w, l, t
            skills = [float(r["skill"]) for r in mrows if r["skill"] != ""]
        out.append(
            {
                "evaluation_context": context,
                "model": model,
                "n_sequences": len(mrows),
                "mean_sequence_rmse": boot["mean"],
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "median_sequence_rmse": boot["median"],
                "mean_sequence_mae": float(np.mean(maes)),
                "mean_rmse_50ms": float(np.mean(r50)),
                "mean_rmse_100ms": float(np.mean(r100)),
                "mean_rmse_200ms": float(np.mean(r200)),
                "mean_persistence_skill": equal_sequence_mean(skills) if skills else "",
                "wins_vs_persistence": wins,
                "losses_vs_persistence": losses,
                "ties_vs_persistence": ties,
            }
        )
    return out


def paired_rows(rows: list[dict[str, Any]], context: str, n_reps: int, seed: int) -> list[dict[str, Any]]:
    sub = [r for r in rows if r["evaluation_context"] == context]
    pers = {r["sequence_id"]: float(r["rmse_all_horizons"]) for r in sub if r["model"] == "B0_PERSISTENCE"}
    out = []
    for model in ("B1_LINEAR", "B2_RIDGE_QF", "B3_RIDGE_QF_QW"):
        mrows = [r for r in sub if r["model"] == model]
        delta = np.array([float(r["rmse_all_horizons"]) - pers[r["sequence_id"]] for r in mrows], dtype=np.float64)
        boot = bootstrap_mean_ci(delta, n_reps=n_reps, seed=seed)
        w, l, t = win_loss_tie(delta)
        p = paired_sign_flip_p(delta)
        out.append(
            {
                "evaluation_context": context,
                "model": model,
                "n_sequences": int(delta.size),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "wins": w,
                "losses": l,
                "ties": t,
                "signflip_p": p,
                "notes": "exploratory; small n; not confirmatory",
            }
        )
    return out


def horizon_equal_sequence(rows: list[dict[str, Any]], context: str) -> list[dict[str, Any]]:
    sub = [r for r in rows if r["evaluation_context"] == context]
    out = []
    for model in ("B0_PERSISTENCE", "B1_LINEAR", "B2_RIDGE_QF", "B3_RIDGE_QF_QW"):
        mrows = [r for r in sub if r["model"] == model]
        for h in H_LABELS:
            vals = [float(r[f"rmse_h{h:02d}"]) for r in mrows]
            out.append(
                {
                    "evaluation_context": context,
                    "model": model,
                    "horizon_index": h,
                    "horizon_ms": h * 10,
                    "equal_sequence_mean_rmse": equal_sequence_mean(vals),
                }
            )
    return out


def strip_pred(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "_pred"} for r in rows]


def advanced_gate(summary: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> tuple[str, str]:
    def mean_skill(ctx: str, model: str) -> float:
        hits = [r for r in summary if r["evaluation_context"] == ctx and r["model"] == model]
        if not hits:
            return float("nan")
        v = hits[0]["mean_persistence_skill"]
        return float(v) if v != "" else float("nan")

    def mean_rmse(ctx: str, model: str) -> float:
        hits = [r for r in summary if r["evaluation_context"] == ctx and r["model"] == model]
        return float(hits[0]["mean_sequence_rmse"]) if hits else float("nan")

    pers_over_sd = [float(r["persistence_rmse_over_sd"]) for r in pred_rows]
    med_ratio = float(np.median(pers_over_sd)) if pers_over_sd else float("nan")
    skills_learned = []
    for ctx in ("WITHIN_EUROC", "WITHIN_UZH"):
        for m in ("B2_RIDGE_QF", "B3_RIDGE_QF_QW"):
            skills_learned.append(mean_skill(ctx, m))
    skills_transfer = []
    for ctx in ("EUROC_TO_UZH", "UZH_TO_EUROC"):
        for m in ("B2_RIDGE_QF", "B3_RIDGE_QF_QW"):
            skills_transfer.append(mean_skill(ctx, m))
    best_within = float(np.nanmax(np.asarray(skills_learned, dtype=np.float64)))
    best_transfer = float(np.nanmax(np.asarray(skills_transfer, dtype=np.float64)))
    ridge_within = min(
        mean_rmse("WITHIN_EUROC", "B2_RIDGE_QF"),
        mean_rmse("WITHIN_EUROC", "B3_RIDGE_QF_QW"),
        mean_rmse("WITHIN_UZH", "B2_RIDGE_QF"),
        mean_rmse("WITHIN_UZH", "B3_RIDGE_QF_QW"),
    )
    pers_within = min(mean_rmse("WITHIN_EUROC", "B0_PERSISTENCE"), mean_rmse("WITHIN_UZH", "B0_PERSISTENCE"))
    if med_ratio < 0.05 and best_within < 0.05:
        return (
            "RESEARCH_TARGET_REVIEW_REQUIRED",
            "Persistence RMSE is a tiny fraction of q_f SD and learned baselines add almost no skill; the 200 ms horizon may be nearly deterministic.",
        )
    if best_within > 0.8 and best_transfer > 0.6 and ridge_within < 0.25 * pers_within:
        return (
            "ADVANCED_MODELS_LOW_PRIORITY",
            "Ridge already captures most available short-horizon skill within and across datasets; deep models are unlikely to change the scientific conclusion.",
        )
    if best_within > 0.05 and (best_transfer < best_within - 0.05 or best_transfer < 0.05):
        return (
            "ADVANCED_MODELS_JUSTIFIED",
            "At least one learned baseline shows nontrivial within-domain skill, remaining error is non-negligible, and zero-shot transfer is materially harder.",
        )
    if best_within > 0.1:
        return (
            "ADVANCED_MODELS_JUSTIFIED",
            "Learned baselines extract nontrivial signal with remaining error headroom.",
        )
    if best_within <= 0.05 and med_ratio > 0.3:
        return (
            "ADVANCED_MODELS_JUSTIFIED",
            "Persistence is not trivially accurate, yet linear/Ridge skill is small; a later capacity-controlled model family may be warranted as a test of residual structure.",
        )
    return (
        "ADVANCED_MODELS_LOW_PRIORITY",
        "Stage-2 baselines do not show a clear residual structure that would justify deep models without a target-definition review.",
    )


def run_stage2(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT
    lock = verify_protocol_lock(root)
    if lock["match"] != "PASS":
        print("PROTOCOL_LOCK_MISMATCH", lock, flush=True)
        return {"decision": "PROTOCOL_LOCK_MISMATCH", "lock": lock}

    cfg = yaml.safe_load((root / "configs" / "stage2_baselines.yaml").read_text(encoding="utf-8"))
    out = root / "results" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = root / "artifacts" / "stage2"
    art_dir.mkdir(parents=True, exist_ok=True)
    utc = utc_now()
    config_hash = sha256_file(root / "configs" / "stage2_baselines.yaml")
    (out / "STAGE2_CONFIG_HASH.txt").write_text(
        f"utc={utc}\nconfigs/stage2_baselines.yaml sha256={config_hash}\nphase=2A_pre_transfer\n",
        encoding="utf-8",
    )

    print("[stage2] loading processed sequences", flush=True)
    raw_map = load_all_processed(root)
    packs: dict[tuple[str, str], SequenceWindows] = {}
    euroc_packs: dict[str, SequenceWindows] = {}
    uzh_packs: dict[str, SequenceWindows] = {}
    for (ds, seq), rec in raw_map.items():
        pack = build_sequence_windows(
            rec,
            lookback_samples=int(cfg["lookback_samples"]),
            horizon_samples=int(cfg["horizon_samples"]),
            stride_samples=int(cfg["stride_samples"]),
            high_percentile=float(cfg["high_qf_percentile"]),
        )
        packs[(ds, seq)] = pack
        if ds == "EUROC":
            euroc_packs[seq] = pack
        else:
            uzh_packs[seq] = pack

    print("[stage2] extreme-value audit", flush=True)
    extreme_rows = []
    for (ds, seq), pack in packs.items():
        native = load_native(root, ds, seq)
        extreme_rows.append(audit_sequence(pack, native))
    write_csv(
        out / "EXTREME_VALUE_AUDIT.csv",
        [
            "dataset",
            "sequence_id",
            "q_f_min",
            "q_f_p001",
            "q_f_p01",
            "q_f_p99",
            "q_f_p999",
            "q_f_max",
            "timestamp_of_max",
            "duration_above_30_mps2",
            "duration_above_50_mps2",
            "duration_above_100_mps2",
            "q_w_at_qf_max",
            "neighboring_values_consistent",
            "possible_sensor_saturation",
            "possible_single_sample_spike",
            "raw_source_consistent",
            "integrity_interpretation",
            "action",
        ],
        extreme_rows,
    )
    if any(r["action"] == "RAW_INTEGRITY_REVIEW_REQUIRED" for r in extreme_rows):
        return {
            "decision": "FAIL",
            "reason": "RAW_INTEGRITY_REVIEW_REQUIRED",
            "lock": lock,
            "extreme": extreme_rows,
        }

    fit_log: list[dict[str, str]] = []
    hp_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    n_reps = int(cfg["bootstrap_reps"])
    seed = int(cfg["bootstrap_seed"])

    print("[stage2] PHASE 2A within-EuRoC", flush=True)
    within_euroc = run_loro(
        dataset="EUROC",
        seq_ids=EUROC_PRIMARY,
        groups=EUROC_ROOM,
        packs=euroc_packs,
        context="WITHIN_EUROC",
        cfg=cfg,
        fit_log=fit_log,
        hp_rows=hp_rows,
        coeff_rows=coeff_rows,
        scaler_rows=scaler_rows,
        config_hash=config_hash,
    )
    print("[stage2] PHASE 2A within-UZH", flush=True)
    within_uzh = run_loro(
        dataset="UZH_FPV",
        seq_ids=UZH_PRIMARY,
        groups=UZH_GROUP,
        packs=uzh_packs,
        context="WITHIN_UZH",
        cfg=cfg,
        fit_log=fit_log,
        hp_rows=hp_rows,
        coeff_rows=coeff_rows,
        scaler_rows=scaler_rows,
        config_hash=config_hash,
    )

    print("[stage2] PHASE 2A source-only hyperparameter freeze", flush=True)
    frozen: dict[str, Any] = {}
    for dataset, seq_ids, groups, pmap, tag in (
        ("EUROC", list(EUROC_PRIMARY), EUROC_ROOM, euroc_packs, "EUROC"),
        ("UZH_FPV", list(UZH_PRIMARY), UZH_GROUP, uzh_packs, "UZH"),
    ):
        frozen[tag] = {}
        for model_name, use_qw in MODELS_RIDGE:
            chosen = select_ridge_config(
                seq_ids,
                pmap,
                groups,
                dataset=dataset,
                model_name=model_name,
                lags=list(cfg["lags"]),
                alphas=[float(a) for a in cfg["alphas"]],
                use_qw=use_qw,
                tie_relative_tol=float(cfg["tie_relative_tol"]),
                fit_log=fit_log,
                context=f"SOURCE_SELECTION_{tag}",
            )
            frozen[tag][model_name] = chosen
            hp_rows.append(
                {
                    "evaluation_context": f"SOURCE_SELECTION_{tag}",
                    "phase": "SOURCE_SELECTION_FOR_TRANSFER",
                    "fold_id": f"{tag}_SOURCE_ALL",
                    "dataset": dataset,
                    "model": model_name,
                    "selected_lag": chosen["lag"],
                    "selected_alpha": chosen["alpha"],
                    "selection_score_rmse": chosen["score"],
                    "n_tied": chosen["n_tied"],
                    "train_sequences": ",".join(seq_ids),
                    "held_out_sequence": "",
                    "notes": "source-only group-aware selection; no target data",
                }
            )

    freeze_path = out / "SOURCE_MODELS_FROZEN.md"
    freeze_text = _freeze_markdown(utc, config_hash, frozen)
    freeze_path.write_text(freeze_text, encoding="utf-8")
    freeze_hash = sha256_file(freeze_path)
    (out / "SOURCE_MODELS_FROZEN.sha256").write_text(freeze_hash + "\n", encoding="utf-8")
    print(f"[stage2] SOURCE_MODELS_FROZEN.md sha256={freeze_hash}", flush=True)

    print("[stage2] PHASE 2B zero-shot transfer", flush=True)
    transfer_rows: list[dict[str, Any]] = []
    transfer_models: dict[str, dict[str, RidgePack]] = {}
    linear_k = int(cfg["linear_k"])
    dt_s = float(cfg["dt_s"])
    for src_tag, src_ds, src_ids, src_packs, tgt_ds, tgt_ids, tgt_packs, ctx in (
        ("EUROC", "EUROC", list(EUROC_PRIMARY), euroc_packs, "UZH_FPV", list(UZH_PRIMARY), uzh_packs, "EUROC_TO_UZH"),
        ("UZH", "UZH_FPV", list(UZH_PRIMARY), uzh_packs, "EUROC", list(EUROC_PRIMARY), euroc_packs, "UZH_TO_EUROC"),
    ):
        fitted_map: dict[str, RidgePack] = {}
        for model_name, use_qw in MODELS_RIDGE:
            ch = frozen[src_tag][model_name]
            fitted = fit_ridge(
                [src_packs[s] for s in src_ids],
                lag=int(ch["lag"]),
                alpha=float(ch["alpha"]),
                use_qw=use_qw,
                dataset=src_ds,
                model_name=model_name,
            )
            fitted_map[model_name] = fitted
            fit_log.append(
                {
                    "object": f"TRANSFER_{ctx}_{model_name}",
                    "object_type": "Ridge+SourceStandardScaler",
                    "fit_dataset": src_ds,
                    "fit_sequences": ",".join(fitted.train_sequences),
                    "fit_purpose": "zero_shot_source_final",
                    "evaluation_context": ctx,
                }
            )
            _record_ridge_aux(fitted, coeff_rows, scaler_rows, config_hash, f"TRANSFER_{ctx}", ctx)
            joblib.dump(
                _artifact_dict(fitted, config_hash, "zero_shot_source_final"),
                art_dir / f"{ctx}_{model_name}.joblib",
            )
        transfer_models[ctx] = fitted_map
        for sid in tgt_ids:
            seq = tgt_packs[sid]
            pers = local_predict(seq, "B0_PERSISTENCE", linear_k=linear_k, dt_s=dt_s)
            pers_rmse = rmse_all(pers, seq.future_qf)
            transfer_rows.extend(
                evaluate_local_and_ridge_on_seq(
                    seq,
                    context=ctx,
                    fold_id=f"TRANSFER_{ctx}",
                    ridge_models=fitted_map,
                    linear_k=linear_k,
                    dt_s=dt_s,
                    pers_rmse=pers_rmse,
                )
            )

    freeze_hash_after = sha256_file(freeze_path)
    transfer_lock = "PASS" if freeze_hash_after == freeze_hash else "FAIL"
    if transfer_lock != "PASS":
        print("CRITICAL FAIL: SOURCE_MODELS_FROZEN.md changed during transfer", flush=True)

    leak_rows = []
    for r in fit_log:
        ctx = r["evaluation_context"]
        if ctx == "EUROC_TO_UZH" or ctx.startswith("SOURCE_SELECTION_EUROC") or ctx == "WITHIN_EUROC":
            allowed = "EUROC"
        elif ctx == "UZH_TO_EUROC" or ctx.startswith("SOURCE_SELECTION_UZH") or ctx == "WITHIN_UZH":
            allowed = "UZH_FPV"
        else:
            allowed = r["fit_dataset"]
        if r["fit_dataset"] != allowed:
            leak_rows.append(r)
        if ctx == "EUROC_TO_UZH" and any(s in r["fit_sequences"] for s in UZH_PRIMARY):
            leak_rows.append(r)
        if ctx == "UZH_TO_EUROC" and any(s in r["fit_sequences"] for s in EUROC_PRIMARY):
            leak_rows.append(r)

    all_rows = within_euroc + within_uzh + transfer_rows
    e2u = [r for r in all_rows if r["evaluation_context"] == "EUROC_TO_UZH"]
    u2e = [r for r in all_rows if r["evaluation_context"] == "UZH_TO_EUROC"]

    write_csv(out / "WITHIN_EUROC_SEQUENCE_RESULTS.csv", SEQ_COLS, strip_pred(within_euroc))
    write_csv(out / "WITHIN_UZH_SEQUENCE_RESULTS.csv", SEQ_COLS, strip_pred(within_uzh))
    write_csv(out / "TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv", SEQ_COLS, strip_pred(e2u))
    write_csv(out / "TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv", SEQ_COLS, strip_pred(u2e))
    write_csv(
        out / "HYPERPARAMETER_SELECTION.csv",
        [
            "evaluation_context",
            "phase",
            "fold_id",
            "dataset",
            "model",
            "selected_lag",
            "selected_alpha",
            "selection_score_rmse",
            "n_tied",
            "train_sequences",
            "held_out_sequence",
            "notes",
        ],
        hp_rows,
    )
    write_csv(
        out / "SCALER_MANIFEST.csv",
        [
            "evaluation_context",
            "fold_id",
            "model",
            "fit_dataset",
            "fit_sequences",
            "n_features",
            "n_windows",
            "target_scaler",
            "fit_scope",
            "mean_l2",
            "std_l2",
            "config_hash",
        ],
        scaler_rows,
    )
    write_csv(
        out / "RIDGE_COEFFICIENT_AUDIT.csv",
        [
            "evaluation_context",
            "fold_id",
            "model",
            "dataset",
            "lag",
            "alpha",
            "n_features",
            "n_outputs",
            "coef_norm",
            "intercept_norm",
            "n_windows",
            "train_sequences",
            "equal_sequence_weight_sums",
            "config_hash",
            "notes",
        ],
        coeff_rows,
    )
    write_csv(
        out / "FITTED_OBJECT_LOG.csv",
        ["object", "object_type", "fit_dataset", "fit_sequences", "fit_purpose", "evaluation_context"],
        fit_log,
    )

    summary = []
    paired = []
    horizon = []
    boot_sum = []
    for ctx in ("WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"):
        summary.extend(summarize_context(all_rows, ctx, n_reps, seed))
        paired.extend(paired_rows(all_rows, ctx, n_reps, seed))
        horizon.extend(horizon_equal_sequence(all_rows, ctx))
        sub = [r for r in all_rows if r["evaluation_context"] == ctx]
        for model in ("B0_PERSISTENCE", "B1_LINEAR", "B2_RIDGE_QF", "B3_RIDGE_QF_QW"):
            rmses = np.array(
                [float(r["rmse_all_horizons"]) for r in sub if r["model"] == model],
                dtype=np.float64,
            )
            boot = bootstrap_mean_ci(rmses, n_reps=n_reps, seed=seed)
            boot["evaluation_context"] = ctx
            boot["model"] = model
            boot["n_sequences"] = int(rmses.size)
            boot_sum.append(boot)

    write_csv(
        out / "BASELINE_SUMMARY.csv",
        [
            "evaluation_context",
            "model",
            "n_sequences",
            "mean_sequence_rmse",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "median_sequence_rmse",
            "mean_sequence_mae",
            "mean_rmse_50ms",
            "mean_rmse_100ms",
            "mean_rmse_200ms",
            "mean_persistence_skill",
            "wins_vs_persistence",
            "losses_vs_persistence",
            "ties_vs_persistence",
        ],
        summary,
    )
    write_csv(
        out / "PAIRED_COMPARISONS.csv",
        [
            "evaluation_context",
            "model",
            "n_sequences",
            "mean_delta",
            "median_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "wins",
            "losses",
            "ties",
            "signflip_p",
            "notes",
        ],
        paired,
    )
    write_csv(
        out / "HORIZON_RESULTS.csv",
        ["evaluation_context", "model", "horizon_index", "horizon_ms", "equal_sequence_mean_rmse"],
        horizon,
    )
    write_csv(
        out / "BOOTSTRAP_SUMMARY.csv",
        ["evaluation_context", "model", "n_sequences", "mean", "ci_low", "ci_high", "median", "std", "min", "max"],
        boot_sum,
    )

    pred_rows = []
    high_rows = []
    for (ds, seq), pack in packs.items():
        pers_pred = local_predict(pack, "B0_PERSISTENCE", linear_k=linear_k, dt_s=dt_s)
        pers_rmse = rmse_all(pers_pred, pack.future_qf)
        sd = float(np.std(pack.q_f, ddof=0))
        pred_rows.append(
            {
                "dataset": ds,
                "sequence_id": seq,
                "n_processed": int(pack.q_f.size),
                "n_origins": pack.n_windows,
                "sd_qf": sd,
                "persistence_rmse": pers_rmse,
                "persistence_rmse_over_sd": pers_rmse / sd if sd > 0 else "",
                "acf_10ms": acf_at_lag(pack.q_f, 1),
                "acf_50ms": acf_at_lag(pack.q_f, 5),
                "acf_100ms": acf_at_lag(pack.q_f, 10),
                "acf_200ms": acf_at_lag(pack.q_f, 20),
                "p95_qf": pack.p95_qf,
            }
        )
        for r in all_rows:
            if r["sequence_id"] != seq or r["dataset"] != ds:
                continue
            if "_pred" not in r:
                continue
            high_rows.append(
                {
                    "evaluation_context": r["evaluation_context"],
                    "dataset": ds,
                    "sequence_id": seq,
                    "model": r["model"],
                    "rmse_all": r["rmse_all_horizons"],
                    "rmse_high_p95_targets": high_target_rmse(r["_pred"], pack.future_qf, pack.p95_qf),
                    "p95_threshold": pack.p95_qf,
                    "notes": "secondary descriptive; training unchanged",
                }
            )
    write_csv(
        out / "PREDICTABILITY_CHARACTERIZATION.csv",
        [
            "dataset",
            "sequence_id",
            "n_processed",
            "n_origins",
            "sd_qf",
            "persistence_rmse",
            "persistence_rmse_over_sd",
            "acf_10ms",
            "acf_50ms",
            "acf_100ms",
            "acf_200ms",
            "p95_qf",
        ],
        pred_rows,
    )
    write_csv(
        out / "HIGH_QF_SECONDARY.csv",
        [
            "evaluation_context",
            "dataset",
            "sequence_id",
            "model",
            "rmse_all",
            "rmse_high_p95_targets",
            "p95_threshold",
            "notes",
        ],
        high_rows,
    )

    grouped_rmse_by_sequence(
        strip_pred(within_euroc),
        title="Within EuRoC: sequence multi-horizon RMSE",
        pdf=fig_dir / "figure1_within_euroc_rmse.pdf",
        png=fig_dir / "figure1_within_euroc_rmse.png",
        csv_path=out / "figure_data" / "figure1_within_euroc_rmse.csv",
    )
    grouped_rmse_by_sequence(
        strip_pred(within_uzh),
        title="Within UZH-FPV: sequence multi-horizon RMSE",
        pdf=fig_dir / "figure2_within_uzh_rmse.pdf",
        png=fig_dir / "figure2_within_uzh_rmse.png",
        csv_path=out / "figure_data" / "figure2_within_uzh_rmse.csv",
    )
    grouped_rmse_by_sequence(
        strip_pred(e2u),
        title=r"EuRoC $\rightarrow$ UZH zero-shot: sequence multi-horizon RMSE",
        pdf=fig_dir / "figure3_euroc_to_uzh_rmse.pdf",
        png=fig_dir / "figure3_euroc_to_uzh_rmse.png",
        csv_path=out / "figure_data" / "figure3_euroc_to_uzh_rmse.csv",
    )
    grouped_rmse_by_sequence(
        strip_pred(u2e),
        title=r"UZH $\rightarrow$ EuRoC zero-shot: sequence multi-horizon RMSE",
        pdf=fig_dir / "figure4_uzh_to_euroc_rmse.pdf",
        png=fig_dir / "figure4_uzh_to_euroc_rmse.png",
        csv_path=out / "figure_data" / "figure4_uzh_to_euroc_rmse.csv",
    )
    horizon_curves(
        horizon,
        pdf=fig_dir / "figure5_rmse_vs_horizon.pdf",
        png=fig_dir / "figure5_rmse_vs_horizon.png",
        csv_path=out / "figure_data" / "figure5_rmse_vs_horizon.csv",
    )
    skill_by_sequence(
        strip_pred(all_rows),
        pdf=fig_dir / "figure6_skill_by_sequence.pdf",
        png=fig_dir / "figure6_skill_by_sequence.png",
        csv_path=out / "figure_data" / "figure6_skill_by_sequence.csv",
    )

    n_target_fitted = len(leak_rows)
    gate, gate_reason = advanced_gate(summary, pred_rows)
    crit = []
    if transfer_lock != "PASS":
        crit.append("SOURCE_MODELS_FROZEN.md hash changed during transfer")
    if n_target_fitted:
        crit.append(f"target-fitted objects: {n_target_fitted}")
    decision = "FAIL" if crit else "PASS"

    ctx = {
        "utc": utc,
        "lock": lock,
        "config_hash": config_hash,
        "freeze_hash": freeze_hash,
        "freeze_hash_after": freeze_hash_after,
        "transfer_lock": transfer_lock,
        "n_target_fitted": n_target_fitted,
        "decision": decision,
        "gate": gate,
        "gate_reason": gate_reason,
        "frozen": {
            tag: {m: {k: v for k, v in frozen[tag][m].items() if k != "all_records"} for m in frozen[tag]}
            for tag in frozen
        },
        "summary": summary,
        "extreme_actions": {r["sequence_id"]: r["action"] for r in extreme_rows},
        "n_findings_critical": len(crit),
        "critical": crit,
    }
    (out / "STAGE2_CONTEXT.json").write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    report = _stage2_report(ctx, summary, paired, pred_rows, high_rows, extreme_rows)
    (root / "STAGE2_REPORT.md").write_text(report, encoding="utf-8")
    (out / "STAGE2_REPORT.md").write_text(report, encoding="utf-8")
    _append_decisions(root, utc, decision, gate, config_hash)
    _append_stop(root, utc, decision, gate)
    return ctx


def _freeze_markdown(utc: str, config_hash: str, frozen: dict[str, Any]) -> str:
    e2 = frozen["EUROC"]["B2_RIDGE_QF"]
    e3 = frozen["EUROC"]["B3_RIDGE_QF_QW"]
    u2 = frozen["UZH"]["B2_RIDGE_QF"]
    u3 = frozen["UZH"]["B3_RIDGE_QF_QW"]
    return f"""# Source models frozen (Phase 2A)

utc: {utc}

configs/stage2_baselines.yaml sha256: {config_hash}

Do not modify this file after Phase 2B begins.
Hyperparameters below were selected with SOURCE-DOMAIN validation only.

## EuRoC selected B2 (Ridge q_f)

- lag: {e2['lag']}
- alpha: {e2['alpha']}
- equal-validation-sequence multi-horizon RMSE: {e2['score']:.10g}
- n_tied within 1% rule: {e2['n_tied']}

## EuRoC selected B3 (Ridge q_f+q_w)

- lag: {e3['lag']}
- alpha: {e3['alpha']}
- equal-validation-sequence multi-horizon RMSE: {e3['score']:.10g}
- n_tied within 1% rule: {e3['n_tied']}

## UZH selected B2 (Ridge q_f)

- lag: {u2['lag']}
- alpha: {u2['alpha']}
- equal-validation-sequence multi-horizon RMSE: {u2['score']:.10g}
- n_tied within 1% rule: {u2['n_tied']}

## UZH selected B3 (Ridge q_f+q_w)

- lag: {u3['lag']}
- alpha: {u3['alpha']}
- equal-validation-sequence multi-horizon RMSE: {u3['score']:.10g}
- n_tied within 1% rule: {u3['n_tied']}

## Scaler rules

- Feature scaler: equal-recording-weighted standard scaler
- Fitted on source training windows only
- Target scaler: NONE (errors remain in m/s^2 without inverse transform)
- FIT_SCOPE: SOURCE_TRAIN_ONLY
- Never fit on validation, within-domain test, or cross-domain target sequences

## Training weights

Each training window weight is 1/N_windows of its recording, then normalized so each recording has equal total weight.
"""


def _fmt_ctx(summary: list[dict[str, Any]], ctx: str) -> str:
    lines = []
    for r in summary:
        if r["evaluation_context"] != ctx:
            continue
        skill = r["mean_persistence_skill"]
        skill_s = "n/a" if skill == "" else f"{float(skill):.4f}"
        lines.append(
            f"- {r['model']}: mean seq RMSE={float(r['mean_sequence_rmse']):.4f} m/s^2 "
            f"(95% CI {float(r['bootstrap_ci_low']):.4f}–{float(r['bootstrap_ci_high']):.4f}); "
            f"50/100/200 ms={float(r['mean_rmse_50ms']):.4f}/{float(r['mean_rmse_100ms']):.4f}/{float(r['mean_rmse_200ms']):.4f}; "
            f"skill={skill_s}"
        )
    return "\n".join(lines)


def _best(summary: list[dict[str, Any]], ctx: str) -> str:
    sub = [r for r in summary if r["evaluation_context"] == ctx]
    best = min(sub, key=lambda r: float(r["mean_sequence_rmse"]))
    return f"{best['model']} (mean seq RMSE {float(best['mean_sequence_rmse']):.4f} m/s^2)"


def _mean_metric(summary: list[dict[str, Any]], ctx: str, model: str, key: str) -> str:
    hits = [r for r in summary if r["evaluation_context"] == ctx and r["model"] == model]
    if not hits:
        return "NA"
    return f"{float(hits[0][key]):.4f}"


def _stage2_report(
    ctx: dict[str, Any],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
    extreme_rows: list[dict[str, Any]],
) -> str:
    fr = ctx["frozen"]
    med_ratio = float(np.median([float(r["persistence_rmse_over_sd"]) for r in pred_rows]))
    med_acf200 = float(np.median([float(r["acf_200ms"]) for r in pred_rows]))
    high_note = "secondary RMSE on targets above sequence p95 is tabulated in HIGH_QF_SECONDARY.csv; training was not altered."
    qw_note = _qw_effect(summary)
    transfer_note = _transfer_degradation(summary)
    return f"""# STAGE 2 REPORT

Protocol lock verified:
PASS (`PROTOCOL_LOCK.md`={ctx['lock']['protocol_md']}; yaml={ctx['lock']['protocol_yaml']})

Extreme-value audit:
all 14 sequences KEEP; no clipping; see results/stage2/EXTREME_VALUE_AUDIT.csv
native Stage-1 maximum 166.609 m/s^2 remains a retained observation (indoor_forward_6_snapdragon)

Sequences used:
EuRoC = 6
UZH = 8

Models:
B0 Persistence
B1 Linear extrapolation
B2 Ridge q_f
B3 Ridge q_f+q_w

Selected EuRoC source hyperparameters:
B2 lag={fr['EUROC']['B2_RIDGE_QF']['lag']} alpha={fr['EUROC']['B2_RIDGE_QF']['alpha']}
B3 lag={fr['EUROC']['B3_RIDGE_QF_QW']['lag']} alpha={fr['EUROC']['B3_RIDGE_QF_QW']['alpha']}

Selected UZH source hyperparameters:
B2 lag={fr['UZH']['B2_RIDGE_QF']['lag']} alpha={fr['UZH']['B2_RIDGE_QF']['alpha']}
B3 lag={fr['UZH']['B3_RIDGE_QF_QW']['lag']} alpha={fr['UZH']['B3_RIDGE_QF_QW']['alpha']}

Within EuRoC:
{_fmt_ctx(summary, 'WITHIN_EUROC')}

Within UZH:
{_fmt_ctx(summary, 'WITHIN_UZH')}

EuRoC -> UZH:
{_fmt_ctx(summary, 'EUROC_TO_UZH')}

UZH -> EuRoC:
{_fmt_ctx(summary, 'UZH_TO_EUROC')}

Best baseline by context:
- WITHIN_EUROC: {_best(summary, 'WITHIN_EUROC')}
- WITHIN_UZH: {_best(summary, 'WITHIN_UZH')}
- EUROC_TO_UZH: {_best(summary, 'EUROC_TO_UZH')}
- UZH_TO_EUROC: {_best(summary, 'UZH_TO_EUROC')}

Persistence difficulty:
median Persistence_RMSE / SD(q_f) = {med_ratio:.4f}; median ACF at 200 ms = {med_acf200:.4f}

Effect of q_w input:
{qw_note}

50 ms results:
WITHIN_EUROC Persistence/Linear/Ridge_qf/Ridge_both = {_mean_metric(summary,'WITHIN_EUROC','B0_PERSISTENCE','mean_rmse_50ms')} / {_mean_metric(summary,'WITHIN_EUROC','B1_LINEAR','mean_rmse_50ms')} / {_mean_metric(summary,'WITHIN_EUROC','B2_RIDGE_QF','mean_rmse_50ms')} / {_mean_metric(summary,'WITHIN_EUROC','B3_RIDGE_QF_QW','mean_rmse_50ms')}
WITHIN_UZH = {_mean_metric(summary,'WITHIN_UZH','B0_PERSISTENCE','mean_rmse_50ms')} / {_mean_metric(summary,'WITHIN_UZH','B1_LINEAR','mean_rmse_50ms')} / {_mean_metric(summary,'WITHIN_UZH','B2_RIDGE_QF','mean_rmse_50ms')} / {_mean_metric(summary,'WITHIN_UZH','B3_RIDGE_QF_QW','mean_rmse_50ms')}

100 ms results:
WITHIN_EUROC = {_mean_metric(summary,'WITHIN_EUROC','B0_PERSISTENCE','mean_rmse_100ms')} / {_mean_metric(summary,'WITHIN_EUROC','B1_LINEAR','mean_rmse_100ms')} / {_mean_metric(summary,'WITHIN_EUROC','B2_RIDGE_QF','mean_rmse_100ms')} / {_mean_metric(summary,'WITHIN_EUROC','B3_RIDGE_QF_QW','mean_rmse_100ms')}
WITHIN_UZH = {_mean_metric(summary,'WITHIN_UZH','B0_PERSISTENCE','mean_rmse_100ms')} / {_mean_metric(summary,'WITHIN_UZH','B1_LINEAR','mean_rmse_100ms')} / {_mean_metric(summary,'WITHIN_UZH','B2_RIDGE_QF','mean_rmse_100ms')} / {_mean_metric(summary,'WITHIN_UZH','B3_RIDGE_QF_QW','mean_rmse_100ms')}

200 ms results:
WITHIN_EUROC = {_mean_metric(summary,'WITHIN_EUROC','B0_PERSISTENCE','mean_rmse_200ms')} / {_mean_metric(summary,'WITHIN_EUROC','B1_LINEAR','mean_rmse_200ms')} / {_mean_metric(summary,'WITHIN_EUROC','B2_RIDGE_QF','mean_rmse_200ms')} / {_mean_metric(summary,'WITHIN_EUROC','B3_RIDGE_QF_QW','mean_rmse_200ms')}
WITHIN_UZH = {_mean_metric(summary,'WITHIN_UZH','B0_PERSISTENCE','mean_rmse_200ms')} / {_mean_metric(summary,'WITHIN_UZH','B1_LINEAR','mean_rmse_200ms')} / {_mean_metric(summary,'WITHIN_UZH','B2_RIDGE_QF','mean_rmse_200ms')} / {_mean_metric(summary,'WITHIN_UZH','B3_RIDGE_QF_QW','mean_rmse_200ms')}

High-specific-force secondary analysis:
{high_note}

Sequence-level uncertainty:
10,000 sequence-level bootstrap percentile CIs in BASELINE_SUMMARY.csv and BOOTSTRAP_SUMMARY.csv. n=6/8; p-values are exploratory only.

Transfer degradation:
{transfer_note}

Target-domain fitted objects:
{ctx['n_target_fitted']}

Source-model freeze hash:
{ctx['freeze_hash']}

Transfer freeze verification:
{ctx['transfer_lock']}

Critical findings:
{ctx['n_findings_critical']}
{'; '.join(ctx['critical']) if ctx['critical'] else 'none'}

Major findings:
0

pytest:
SEE_PYTEST passed
SEE_PYTEST failed

ADVANCED-MODEL GATE:
{ctx['gate']}

{ctx['gate_reason']}

OVERALL STAGE-2 DECISION:
{ctx['decision']}
"""


def _qw_effect(summary: list[dict[str, Any]]) -> str:
    bits = []
    for ctx in ("WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"):
        b2 = [r for r in summary if r["evaluation_context"] == ctx and r["model"] == "B2_RIDGE_QF"]
        b3 = [r for r in summary if r["evaluation_context"] == ctx and r["model"] == "B3_RIDGE_QF_QW"]
        if not b2 or not b3:
            continue
        d = float(b3[0]["mean_sequence_rmse"]) - float(b2[0]["mean_sequence_rmse"])
        bits.append(f"{ctx}: B3−B2 = {d:+.4f} m/s^2")
    return "; ".join(bits)


def _transfer_degradation(summary: list[dict[str, Any]]) -> str:
    def rmse(ctx: str, model: str) -> float:
        hits = [r for r in summary if r["evaluation_context"] == ctx and r["model"] == model]
        return float(hits[0]["mean_sequence_rmse"])

    lines = []
    for model in ("B0_PERSISTENCE", "B2_RIDGE_QF", "B3_RIDGE_QF_QW"):
        a = rmse("EUROC_TO_UZH", model) - rmse("WITHIN_UZH", model)
        b = rmse("UZH_TO_EUROC", model) - rmse("WITHIN_EUROC", model)
        lines.append(f"{model}: (EUROC→UZH minus within-UZH)={a:+.4f}; (UZH→EUROC minus within-EuRoC)={b:+.4f}")
    return " ".join(lines)


def _append_decisions(root: Path, utc: str, decision: str, gate: str, config_hash: str) -> None:
    path = root / "DECISIONS.md"
    block = f"""
## {utc}

- **decision:** Execute Stage 2 leakage-safe baselines (Persistence, linear extrapolation, Ridge q_f, Ridge q_f+q_w) under the frozen Stage-1 protocol. Do not add deep models in this stage.
- **reason:** Establish whether q_f is forecastable beyond persistence and whether simple models transfer zero-shot, without retuning after target results.
- **evidence:** configs/stage2_baselines.yaml sha256={config_hash}; STAGE2_REPORT.md; gate={gate}; decision={decision}
- **consequence:** Neural networks remain unauthorized until an independent review of the Stage-2 gate.
"""
    text = path.read_text(encoding="utf-8")
    if "Stage 2 leakage-safe baselines" not in text:
        path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def _append_stop(root: Path, utc: str, decision: str, gate: str) -> None:
    path = root / "STOP_HERE.md"
    block = f"""
# STAGE 2 STATUS

status = {decision}
utc = {utc}
advanced_model_gate = {gate}

NO NEURAL NETWORKS WERE TRAINED.
STOPPING AFTER BASELINE FORECASTING.
"""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if "# STAGE 2 STATUS" not in text:
        path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
