"""Rebuild manuscript-facing publication_current/ from frozen Stage-2/3 CSVs.

Does not train models, retune hyperparameters, or fit target-domain objects.
The primary Ridge comparator is hardcoded to the source-only freeze (B3).
This module never inspects target-domain RMSE to choose a Ridge variant.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from audit.stage0c import write_csv
from stage2.bootstrap import bootstrap_mean_ci, paired_sign_flip_p, win_loss_tie
from stage2.metrics import skill_score
from stage4.io import fnum, read_csv
from stage4.publication_figures import (
    figure1_schematic,
    figure2_primary_rmse,
    figure3_transfer_gap,
    figure4_horizon,
    figure5_advanced_vs_ridge,
)


def fmt3(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{float(mean):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def grab(rows: list[dict[str, Any]], context: str, model: str) -> dict[str, Any]:
    hits = [r for r in rows if r["context"] == context and r["model"] == model]
    if not hits:
        raise KeyError(f"missing {context} {model}")
    return hits[0]

CONTEXTS = ("WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC")
RIDGE_NAME = "Source-selected B3 Ridge"
PUB_MODELS = ("Persistence", RIDGE_NAME, "TCN", "GRU", "Transformer")
ADV_MODELS = ("TCN", "GRU", "Transformer")
BOOT_REPS = 10000
BOOT_SEED = 20260912

# Locked source-only freeze. Do not replace by inspecting target-domain error.
PRIMARY_RIDGE_ID = "B3_RIDGE_QF_QW"
PRIMARY_RIDGE_BY_CONTEXT = {
    "WITHIN_EUROC": PRIMARY_RIDGE_ID,
    "WITHIN_UZH": PRIMARY_RIDGE_ID,
    "EUROC_TO_UZH": PRIMARY_RIDGE_ID,
    "UZH_TO_EUROC": PRIMARY_RIDGE_ID,
}
SUPPLEMENTARY_RIDGE_ID = "B2_RIDGE_QF"

STAGE3_SEQ_FILES = {
    "WITHIN_EUROC": "WITHIN_EUROC_ADVANCED.csv",
    "WITHIN_UZH": "WITHIN_UZH_ADVANCED.csv",
    "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_ADVANCED.csv",
    "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_ADVANCED.csv",
}
STAGE2_SEQ_FILES = {
    "WITHIN_EUROC": "WITHIN_EUROC_SEQUENCE_RESULTS.csv",
    "WITHIN_UZH": "WITHIN_UZH_SEQUENCE_RESULTS.csv",
    "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv",
    "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv",
}
STAGE3_ADV_FILE_MODEL = {
    "B0_PERSISTENCE": "Persistence",
    "TCN": "TCN",
    "GRU": "GRU",
    "TRANSFORMER": "Transformer",
}

RIDGE_DETAILS = {
    "WITHIN_EUROC": {
        "inputs": "q_f + q_w",
        "lag": 25,
        "alpha": 1.0e-4,
        "fit_dataset": "EUROC",
        "selection_rule": "source-only nested LORO validation on EuRoC; never target RMSE",
    },
    "WITHIN_UZH": {
        "inputs": "q_f + q_w",
        "lag": 10,
        "alpha": 0.01,
        "fit_dataset": "UZH_FPV",
        "selection_rule": "source-only nested LORO validation on UZH-FPV; never target RMSE",
    },
    "EUROC_TO_UZH": {
        "inputs": "q_f + q_w",
        "lag": 25,
        "alpha": 1.0e-4,
        "fit_dataset": "EUROC",
        "selection_rule": "source-frozen EuRoC B3 applied zero-shot; never selected on UZH target RMSE",
    },
    "UZH_TO_EUROC": {
        "inputs": "q_f + q_w",
        "lag": 10,
        "alpha": 0.01,
        "fit_dataset": "UZH_FPV",
        "selection_rule": "source-frozen UZH B3 applied zero-shot; never selected on EuRoC target RMSE",
    },
}

B2_DETAILS = {
    "WITHIN_EUROC": {"inputs": "q_f", "lag": 25, "alpha": 1.0e-4, "fit_dataset": "EUROC"},
    "WITHIN_UZH": {"inputs": "q_f", "lag": 25, "alpha": 1.0, "fit_dataset": "UZH_FPV"},
    "EUROC_TO_UZH": {"inputs": "q_f", "lag": 25, "alpha": 1.0e-4, "fit_dataset": "EUROC"},
    "UZH_TO_EUROC": {"inputs": "q_f", "lag": 25, "alpha": 1.0, "fit_dataset": "UZH_FPV"},
}


def primary_ridge_id(context: str) -> str:
    """Return the source-selected primary Ridge ID for a context.

    The return value is the frozen source-only B3 identifier. This function
    does not read RMSE tables and must not be used to pick a lower target error.
    """
    if context not in PRIMARY_RIDGE_BY_CONTEXT:
        raise KeyError(context)
    return PRIMARY_RIDGE_BY_CONTEXT[context]


def load_publication_sequences(root: Path) -> list[dict[str, Any]]:
    """Persistence and advanced models from Stage-3; Ridge from Stage-2 B3 only."""
    rows: list[dict[str, Any]] = []
    for ctx, name in STAGE3_SEQ_FILES.items():
        for raw in read_csv(root / "results" / "stage3" / name):
            model = STAGE3_ADV_FILE_MODEL.get(raw["model"])
            if model is None:
                continue
            rec = dict(raw)
            rec["pub_model"] = model
            rec["evaluation_context"] = ctx
            rec["ridge_source"] = ""
            rows.append(rec)
    for ctx, name in STAGE2_SEQ_FILES.items():
        ridge_id = primary_ridge_id(ctx)
        for raw in read_csv(root / "results" / "stage2" / name):
            if raw["model"] != ridge_id:
                continue
            rec = dict(raw)
            rec["pub_model"] = RIDGE_NAME
            rec["evaluation_context"] = ctx
            rec["ridge_source"] = "stage2_sequence_file:" + name
            rows.append(rec)
    return rows


def sequence_values(rows: list[dict[str, Any]], context: str, pub_model: str, field: str) -> np.ndarray:
    vals = [
        fnum(r[field])
        for r in rows
        if r["evaluation_context"] == context and r.get("pub_model") == pub_model
    ]
    return np.asarray(vals, dtype=np.float64)


def sequence_ids(rows: list[dict[str, Any]], context: str, pub_model: str) -> list[str]:
    ids: list[str] = []
    for r in rows:
        if r["evaluation_context"] == context and r.get("pub_model") == pub_model:
            sid = str(r["sequence_id"])
            if sid not in ids:
                ids.append(sid)
    return ids


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
        if r["evaluation_context"] == context and r.get("pub_model") == RIDGE_NAME
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
        "sign_flip_p": paired_sign_flip_p(delta),
        "ridge_id": primary_ridge_id(context),
    }


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


def transfer_gaps(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = (("EUROC_TO_UZH", "WITHIN_UZH"), ("UZH_TO_EUROC", "WITHIN_EUROC"))
    for model in (RIDGE_NAME, "TCN", "GRU", "Transformer"):
        for direction, within in mapping:
            zs = grab(summaries, direction, model)["mean_seq_rmse"]
            wi = grab(summaries, within, model)["mean_seq_rmse"]
            gap = zs - wi
            rows.append(
                {
                    "direction": direction,
                    "model": model,
                    "ridge_id": PRIMARY_RIDGE_ID if model == RIDGE_NAME else "",
                    "zero_shot_rmse": zs,
                    "within_target_rmse": wi,
                    "transfer_gap": gap,
                    "relative_transfer_gap_pct": 100.0 * gap / wi if wi else float("nan"),
                    "notes": "descriptive; Persistence omitted because it has no trained source model; Ridge is source-selected B3",
                }
            )
    return rows


def primary_table_md(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Primary forecasting results (manuscript-facing, source-selected B3 Ridge)",
        "",
        "Equal-sequence mean multi-horizon RMSE of specific-force magnitude $q_f$ (m/s^2).",
        "Intervals are 95% percentile CIs from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8).",
        "B3 is the primary Ridge comparator in both zero-shot transfer directions because it was selected exclusively by source-domain validation. B2 is retained only as a supplementary sensitivity baseline.",
        "",
        "| Model | Within EuRoC | Within UZH | EuRoC → UZH | UZH → EuRoC |",
        "|---|---|---|---|---|",
    ]
    for model in PUB_MODELS:
        cells = [model]
        for ctx in CONTEXTS:
            r = grab(summaries, ctx, model)
            cells.append(fmt_ci(r["mean_seq_rmse"], r["ci_low"], r["ci_high"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "RMSE at 50 / 100 / 200 ms (equal-sequence means, m/s^2):",
        "",
        "| Model | Context | 50 ms | 100 ms | 200 ms |",
        "|---|---|---|---|---|",
    ]
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            r = grab(summaries, ctx, model)
            lines.append(f"| {model} | {ctx} | {fmt3(r['rmse_50ms'])} | {fmt3(r['rmse_100ms'])} | {fmt3(r['rmse_200ms'])} |")
    lines.append("")
    return "\n".join(lines)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


FIXED100_FILES = (
    "RIDGE_B3_FIXED_LAG100_SUMMARY.csv",
    "RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv",
    "RIDGE_B3_FIXED_LAG100_SEQUENCE_RESULTS.csv",
    "RIDGE_B3_FIXED_LAG100_COMPARISON.md",
)

RECOMMENDED_TABLE_ALIASES = (
    ("TABLE_PRIMARY_RESULTS.csv", "PRIMARY_RESULTS.csv"),
    ("TABLE_PERSISTENCE_RELATIVE_SKILL.csv", "PERSISTENCE_SKILL.csv"),
    ("TABLE_TRANSFER_GAP.csv", "TRANSFER_GAPS.csv"),
    ("TABLE_ADVANCED_VS_RIDGE.csv", "ADVANCED_VS_RIDGE.csv"),
    ("HORIZON_PUBLICATION_DATA.csv", "HORIZON_RMSE_10_TO_200MS.csv"),
)

FILTER_ORDER_CORRECTION_PARAGRAPH = (
    "The Stage-1 filtering calculations and processed data were correct, but the "
    "filter order was misreported because metadata inferred order as twice the "
    "number of SOS rows. The 200-Hz EuRoC design is fifth order and is represented "
    "by three SOS rows, one containing a first-order denominator section. The "
    "500-Hz UZH-FPV design is sixth order. No filter coefficients, processed "
    "signals, trained models, predictions, or forecasting results were changed "
    "by this documentation correction."
)


def write_filter_specification_corrected(root: Path, dest: Path) -> None:
    """Publication-facing filter spec: same SOS coefficients, corrected order labels."""
    original = (root / "results" / "stage1" / "FILTER_SPECIFICATION.md").read_text(encoding="utf-8")
    text = original.replace(
        "# Causal anti-alias filter specification (Stage 1)",
        "# Causal anti-alias filter specification (publication-facing, order labels corrected)",
        1,
    )
    text = text.replace(
        "No prediction performance was used to choose these edges.\n",
        "No prediction performance was used to choose these edges.\n\n"
        + FILTER_ORDER_CORRECTION_PARAGRAPH
        + "\n\n"
        "Executed SOS coefficients are copied unchanged from the historical "
        "`results/stage1/FILTER_SPECIFICATION.md` audit file. That historical file, "
        "and `audit_original/results/stage1/`, retain the original (incorrect) order labels.\n",
        1,
    )
    text = text.replace(
        "- order: 6 (3 SOS sections)",
        "- order: 5 (3 SOS sections; one first-order denominator section with a2 = 0)",
        1,
    )
    _write_text(dest, text)


def write_resampling_audit_corrected(root: Path, dest: Path) -> None:
    """Same resampling audit rows; EuRoC filter_name order label 6 -> 5."""
    original = (root / "results" / "stage1" / "RESAMPLING_AUDIT.csv").read_text(encoding="utf-8")
    text = original.replace("ellip_sos_order6_fs200", "ellip_sos_order5_fs200")
    _write_text(dest, text)


def run_publication_current(root: Path) -> dict[str, Any]:
    out = root / "publication_current"
    fig_dir = out / "figures"
    fig_data = out / "figure_data"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_data.mkdir(parents=True, exist_ok=True)

    seq_rows = load_publication_sequences(root)
    summaries = [summarize_model(seq_rows, ctx, m) for ctx in CONTEXTS for m in PUB_MODELS]
    paired = [paired_vs_ridge(seq_rows, ctx, m) for ctx in CONTEXTS for m in ADV_MODELS]
    horizon_rows: list[dict[str, Any]] = []
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            horizon_rows.extend(horizon_curve(seq_rows, ctx, model))
    gap_rows = transfer_gaps(summaries)

    ridge_map_rows: list[dict[str, Any]] = []
    for ctx in CONTEXTS:
        d = RIDGE_DETAILS[ctx]
        ridge_map_rows.append(
            {
                "context": ctx,
                "primary_ridge_id": primary_ridge_id(ctx),
                "inputs": d["inputs"],
                "lag": d["lag"],
                "alpha": d["alpha"],
                "fit_dataset": d["fit_dataset"],
                "selection_rule": d["selection_rule"],
                "role": "primary Ridge comparator",
                "b2_status": "supplementary sensitivity only",
                "b2_ridge_id": SUPPLEMENTARY_RIDGE_ID,
                "notes": "B3 is the primary Ridge comparator in both zero-shot transfer directions because it was selected exclusively by source-domain validation. B2 is retained only as a supplementary sensitivity baseline.",
            }
        )
    write_csv(
        out / "RIDGE_COMPARATOR_MAP.csv",
        [
            "context",
            "primary_ridge_id",
            "inputs",
            "lag",
            "alpha",
            "fit_dataset",
            "selection_rule",
            "role",
            "b2_status",
            "b2_ridge_id",
            "notes",
        ],
        ridge_map_rows,
    )

    primary_csv_rows = []
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            r = grab(summaries, ctx, model)
            primary_csv_rows.append(
                {
                    "context": ctx,
                    "model": model,
                    "ridge_id": primary_ridge_id(ctx) if model == RIDGE_NAME else "",
                    "n_sequences": r["n_sequences"],
                    "mean_seq_rmse": r["mean_seq_rmse"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "formatted_rmse_ci": fmt_ci(r["mean_seq_rmse"], r["ci_low"], r["ci_high"]),
                    "rmse_50ms": r["rmse_50ms"],
                    "rmse_100ms": r["rmse_100ms"],
                    "rmse_200ms": r["rmse_200ms"],
                    "mean_mae": r["mean_mae"],
                    "skill_vs_persistence": r["skill_vs_persistence"],
                }
            )
    write_csv(
        out / "TABLE_PRIMARY_RESULTS.csv",
        [
            "context",
            "model",
            "ridge_id",
            "n_sequences",
            "mean_seq_rmse",
            "ci_low",
            "ci_high",
            "formatted_rmse_ci",
            "rmse_50ms",
            "rmse_100ms",
            "rmse_200ms",
            "mean_mae",
            "skill_vs_persistence",
        ],
        primary_csv_rows,
    )
    _write_text(out / "TABLE_PRIMARY_RESULTS.md", primary_table_md(summaries))

    skill_rows = []
    for r in primary_csv_rows:
        skill_rows.append(
            {
                "context": r["context"],
                "model": r["model"],
                "ridge_id": r["ridge_id"],
                "model_rmse": r["mean_seq_rmse"],
                "persistence_rmse": grab(summaries, r["context"], "Persistence")["mean_seq_rmse"],
                "skill_vs_persistence": r["skill_vs_persistence"],
                "skill_definition": "1 - model_RMSE / persistence_RMSE (equal-sequence means)",
            }
        )
    write_csv(
        out / "TABLE_PERSISTENCE_RELATIVE_SKILL.csv",
        ["context", "model", "ridge_id", "model_rmse", "persistence_rmse", "skill_vs_persistence", "skill_definition"],
        skill_rows,
    )
    write_csv(
        out / "TABLE_TRANSFER_GAP.csv",
        [
            "direction",
            "model",
            "ridge_id",
            "zero_shot_rmse",
            "within_target_rmse",
            "transfer_gap",
            "relative_transfer_gap_pct",
            "notes",
        ],
        gap_rows,
    )
    write_csv(
        out / "HORIZON_PUBLICATION_DATA.csv",
        ["context", "model", "horizon_step", "horizon_ms", "mean_sequence_rmse", "ci_low", "ci_high"],
        horizon_rows,
    )

    vs_rows = []
    boot_rows = []
    paired_seq = []
    for p in paired:
        vs_rows.append(
            {
                "context": p["context"],
                "model": p["model"],
                "ridge_id": p["ridge_id"],
                "ridge_rmse": p["ridge_mean"],
                "advanced_rmse": p["advanced_mean"],
                "mean_paired_delta": p["mean_paired_delta"],
                "median_paired_delta": p["median_paired_delta"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
                "wins": p["wins"],
                "losses": p["losses"],
                "ties": p["ties"],
                "relative_percent_change_from_ridge": p["relative_improvement_pct"],
                "delta_definition": "Delta = RMSE_advanced - RMSE_Ridge; negative = advanced lower error; Ridge = source-selected B3",
            }
        )
        boot_rows.append(
            {
                "context": p["context"],
                "model": p["model"],
                "ridge_id": p["ridge_id"],
                "n_sequences": p["n_sequences"],
                "mean_paired_delta": p["mean_paired_delta"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
                "bootstrap_replicates": BOOT_REPS,
                "bootstrap_seed": BOOT_SEED,
                "sign_flip_p_exploratory": p["sign_flip_p"],
            }
        )
        for sid, d in zip(p["sequence_ids"], p["delta"], strict=True):
            paired_seq.append(
                {
                    "context": p["context"],
                    "sequence_id": sid,
                    "model": p["model"],
                    "ridge_id": p["ridge_id"],
                    "delta": float(d),
                    "delta_definition": "RMSE_advanced - RMSE_B3_Ridge",
                }
            )
    write_csv(
        out / "TABLE_ADVANCED_VS_RIDGE.csv",
        [
            "context",
            "model",
            "ridge_id",
            "ridge_rmse",
            "advanced_rmse",
            "mean_paired_delta",
            "median_paired_delta",
            "ci_low",
            "ci_high",
            "wins",
            "losses",
            "ties",
            "relative_percent_change_from_ridge",
            "delta_definition",
        ],
        vs_rows,
    )
    write_csv(
        out / "TABLE_PAIRED_BOOTSTRAP.csv",
        [
            "context",
            "model",
            "ridge_id",
            "n_sequences",
            "mean_paired_delta",
            "ci_low",
            "ci_high",
            "bootstrap_replicates",
            "bootstrap_seed",
            "sign_flip_p_exploratory",
        ],
        boot_rows,
    )
    write_csv(
        out / "FLIGHTLEVEL_PAIRED_DIFFERENCES.csv",
        ["context", "sequence_id", "model", "ridge_id", "delta", "delta_definition"],
        paired_seq,
    )

    per_flight = []
    for r in seq_rows:
        if r.get("pub_model") not in PUB_MODELS:
            continue
        per_flight.append(
            {
                "context": r["evaluation_context"],
                "dataset": r["dataset"],
                "sequence_id": r["sequence_id"],
                "model": r["pub_model"],
                "ridge_id": primary_ridge_id(r["evaluation_context"]) if r["pub_model"] == RIDGE_NAME else "",
                "rmse_all_horizons": r["rmse_all_horizons"],
                "mae_all_horizons": r["mae_all_horizons"],
                "rmse_50ms": r["rmse_50ms"],
                "rmse_100ms": r["rmse_100ms"],
                "rmse_200ms": r["rmse_200ms"],
                "n_origins": r["n_origins"],
            }
        )
    write_csv(
        out / "Supplementary_PerFlight_Primary_RMSE.csv",
        [
            "context",
            "dataset",
            "sequence_id",
            "model",
            "ridge_id",
            "rmse_all_horizons",
            "mae_all_horizons",
            "rmse_50ms",
            "rmse_100ms",
            "rmse_200ms",
            "n_origins",
        ],
        per_flight,
    )
    write_csv(
        out / "Supplementary_FlightLevel_Paired_Differences.csv",
        ["context", "sequence_id", "model", "ridge_id", "delta", "delta_definition"],
        paired_seq,
    )
    write_csv(
        out / "Supplementary_Paired_Tests.csv",
        [
            "context",
            "model",
            "ridge_id",
            "n_sequences",
            "mean_paired_delta",
            "median_paired_delta",
            "ci_low",
            "ci_high",
            "wins",
            "losses",
            "ties",
            "sign_flip_p_exploratory",
            "delta_definition",
        ],
        [
            {
                "context": p["context"],
                "model": p["model"],
                "ridge_id": p["ridge_id"],
                "n_sequences": p["n_sequences"],
                "mean_paired_delta": p["mean_paired_delta"],
                "median_paired_delta": p["median_paired_delta"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
                "wins": p["wins"],
                "losses": p["losses"],
                "ties": p["ties"],
                "sign_flip_p_exploratory": p["sign_flip_p"],
                "delta_definition": "Delta = RMSE_advanced - RMSE_Ridge; Ridge = source-selected B3; sign-flip p is exploratory",
            }
            for p in paired
        ],
    )

    baseline = read_csv(root / "results" / "stage2" / "BASELINE_SUMMARY.csv")
    role_map = {
        "B0_PERSISTENCE": "primary Persistence baseline",
        "B1_LINEAR": "supplementary linear-extrapolation baseline",
        "B2_RIDGE_QF": "supplementary sensitivity Ridge (not the primary comparator)",
        "B3_RIDGE_QF_QW": "primary source-selected B3 Ridge comparator",
    }
    sens_rows = []
    for r in baseline:
        rec = dict(r)
        rec["role"] = role_map.get(r["model"], "frozen Stage-2 baseline")
        rec["source_file"] = "results/stage2/BASELINE_SUMMARY.csv"
        sens_rows.append(rec)
    write_csv(
        out / "Supplementary_Baseline_and_Ridge_Sensitivity.csv",
        list(sens_rows[0].keys()),
        sens_rows,
    )
    write_resampling_audit_corrected(root, out / "Supplementary_Resampling_Audit.csv")
    _copy(root / "results" / "stage2" / "HYPERPARAMETER_SELECTION.csv", out / "Supplementary_Ridge_Hyperparameter_Selection.csv")
    _copy(root / "results" / "stage3" / "SELECTED_MODELS.csv", out / "Supplementary_Selected_Advanced_Models.csv")

    fig2_rows = [
        {
            "context": r["context"],
            "model": r["model"],
            "mean_seq_rmse": r["mean_seq_rmse"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
        }
        for r in summaries
    ]
    write_csv(fig_data / "FIG2_primary_rmse.csv", ["context", "model", "mean_seq_rmse", "ci_low", "ci_high"], fig2_rows)
    write_csv(
        fig_data / "FIG3_transfer_gap.csv",
        ["direction", "model", "transfer_gap", "relative_transfer_gap_pct"],
        [{"direction": r["direction"], "model": r["model"], "transfer_gap": r["transfer_gap"], "relative_transfer_gap_pct": r["relative_transfer_gap_pct"]} for r in gap_rows],
    )
    write_csv(
        fig_data / "FIG4_horizon_curves.csv",
        ["context", "model", "horizon_step", "horizon_ms", "mean_sequence_rmse", "ci_low", "ci_high"],
        horizon_rows,
    )
    write_csv(fig_data / "FIG5_paired_delta.csv", ["context", "sequence_id", "model", "ridge_id", "delta", "delta_definition"], paired_seq)

    figure1_schematic(fig_dir / "Fig01_Protocol_Schematic")
    figure2_primary_rmse(summaries, fig_dir / "Fig02_Primary_RMSE")
    figure3_transfer_gap(gap_rows, fig_dir / "Fig03_Transfer_Gap")
    figure4_horizon(horizon_rows, fig_dir / "Fig04_Horizon_RMSE")
    figure5_advanced_vs_ridge(paired_seq, paired, fig_dir / "Fig05_Advanced_vs_Ridge")
    _copy(Path(__file__).with_name("publication_figures.py"), fig_dir / "source_figures.py")

    uzh_ridge = grab(summaries, "UZH_TO_EUROC", RIDGE_NAME)
    write_filter_specification_corrected(root, out / "FILTER_SPECIFICATION_CORRECTED.md")
    write_resampling_audit_corrected(root, out / "RESAMPLING_AUDIT_CORRECTED.csv")
    for src_name, alias in RECOMMENDED_TABLE_ALIASES:
        _copy(out / src_name, out / alias)
    sens = root / "results" / "sensitivity"
    for name in FIXED100_FILES:
        src = sens / name
        if not src.exists():
            raise FileNotFoundError(f"missing fixed-100-lag sensitivity file: {src}")
        _copy(src, out / name)

    _write_text(
        out / "README.md",
        "\n".join(
            [
                "# publication_current",
                "",
                "Authoritative manuscript-facing tables and figures for this package.",
                "",
                "B3 is the primary Ridge comparator in both zero-shot transfer directions because it was selected exclusively by source-domain validation. B2 is retained only as a supplementary sensitivity baseline.",
                "",
                "These files are recomputed from frozen `results/stage2/` and `results/stage3/` CSVs. No model was retrained.",
                "Ridge is always `B3_RIDGE_QF_QW`. Target-domain RMSE is never used to choose the comparator.",
                "",
                f"UZH_TO_EUROC primary Ridge mean RMSE = {uzh_ridge['mean_seq_rmse']:.12f}",
                f"50/100/200 ms = {uzh_ridge['rmse_50ms']:.12f} / {uzh_ridge['rmse_100ms']:.12f} / {uzh_ridge['rmse_200ms']:.12f}",
                "",
                "Filter-order labels in `FILTER_SPECIFICATION_CORRECTED.md` and `RESAMPLING_AUDIT_CORRECTED.csv` are publication-facing: EuRoC/200 Hz = order 5, UZH-FPV/500 Hz = order 6. Historical Stage-1 files keep the original (incorrect) order-6 labels for both designs. SOS coefficients are unchanged.",
                "",
                "Fixed 100-lag Ridge B3 CSVs here are a **post hoc supplementary** equal-history sensitivity. They do not replace the primary source-selected-lag B3 comparator.",
                "",
                "Recommended filenames (`PRIMARY_RESULTS.csv`, `PERSISTENCE_SKILL.csv`, `TRANSFER_GAPS.csv`, `ADVANCED_VS_RIDGE.csv`, `HORIZON_RMSE_10_TO_200MS.csv`) are copies of the `TABLE_*` / `HORIZON_PUBLICATION_DATA.csv` files.",
                "",
                "Historical Stage-4 outputs under `results/stage4/` retain the pre-correction (descriptive best-frozen Ridge, UZH→EuRoC B2) audit.",
                "Figure 6 (high $q_f$) is not part of the current main manuscript and is not regenerated here.",
                "",
            ]
        ),
    )
    _write_text(
        out / "FIGURE_CAPTIONS.md",
        "\n".join(
            [
                "# Figure captions (manuscript-facing)",
                "",
                "## Figure 1. Protocol schematic",
                "Locked measurement and evaluation protocol: invariant scalars, causal 100 Hz streams, source-only fit, within-domain LORO, and zero-shot transfer.",
                "",
                "## Figure 2. Primary RMSE",
                "Equal-sequence mean multi-horizon RMSE of $q_f$ (m/s^2) for Persistence, the source-selected B3 Ridge comparator, TCN, GRU, and Transformer. Error bars are 95% percentile intervals from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8).",
                "",
                "## Figure 3. Transfer gap",
                "Transfer gap = zero-shot target RMSE − corresponding within-target RMSE for the source-selected B3 Ridge comparator and the three advanced models. Persistence is omitted because it has no trained source model.",
                "",
                "## Figure 4. Horizon RMSE",
                "Equal-sequence RMSE versus forecast horizon for all 20 steps from 10 ms to 200 ms.",
                "",
                "## Figure 5. Advanced versus Ridge",
                "Per-flight paired $\\Delta$ = RMSE_advanced − RMSE_B3 Ridge (m/s^2), with the equal-sequence mean and 95% sequence-bootstrap CI shown to the right of each panel. Negative values mean the advanced model has lower sequence RMSE than source-selected B3 Ridge.",
                "",
            ]
        ),
    )

    supp_root = root / "supplementary"
    for name in (
        "Supplementary_PerFlight_Primary_RMSE.csv",
        "Supplementary_FlightLevel_Paired_Differences.csv",
        "Supplementary_Paired_Tests.csv",
        "Supplementary_Baseline_and_Ridge_Sensitivity.csv",
        "Supplementary_Resampling_Audit.csv",
        "Supplementary_Ridge_Hyperparameter_Selection.csv",
        "Supplementary_Selected_Advanced_Models.csv",
    ):
        _copy(out / name, supp_root / name)
    for name in FIXED100_FILES:
        if name.endswith(".csv"):
            _copy(out / name, supp_root / name)

    return {
        "summaries": summaries,
        "paired": paired,
        "uzh_to_euroc_ridge_mean": uzh_ridge["mean_seq_rmse"],
        "uzh_to_euroc_ridge_50": uzh_ridge["rmse_50ms"],
        "uzh_to_euroc_ridge_100": uzh_ridge["rmse_100ms"],
        "uzh_to_euroc_ridge_200": uzh_ridge["rmse_200ms"],
        "out": out,
    }
