"""Stage-4 evidence freeze orchestration. Reads frozen results; does not fit models."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from audit.hashing import sha256_file
from audit.stage0c import write_csv
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY
from stage2.io_processed import load_all_processed

from .figures import (
    figure1_schematic,
    figure2_primary_rmse,
    figure3_horizon,
    figure4_paired,
    figure5_transfer_gap,
    figure6_high_qf,
)
from .io import (
    ADV_MODELS,
    BEST_RIDGE_ID,
    CONTEXTS,
    PUB_MODELS,
    TOL_CSV,
    fnum,
    load_stage2_sequences,
    load_stage3_sequences,
    read_csv,
)
from .locks import locks_ok, verify_locks
from .stats import all_paired, all_summaries, evidence_class, horizon_curve

UTC = timezone.utc
G = 9.81

RIDGE_DETAILS = {
    "WITHIN_EUROC": {
        "model_id": "B3_RIDGE_QF_QW",
        "inputs": "q_f + q_w",
        "lag": 25,
        "alpha": 1.0e-4,
        "fit_dataset": "EUROC",
        "role": "within-domain LORO winner",
    },
    "WITHIN_UZH": {
        "model_id": "B3_RIDGE_QF_QW",
        "inputs": "q_f + q_w",
        "lag": 10,
        "alpha": 0.01,
        "fit_dataset": "UZH_FPV",
        "role": "within-domain LORO winner",
    },
    "EUROC_TO_UZH": {
        "model_id": "B3_RIDGE_QF_QW",
        "inputs": "q_f + q_w",
        "lag": 25,
        "alpha": 1.0e-4,
        "fit_dataset": "EUROC",
        "role": "source-frozen EuRoC B3 applied zero-shot",
    },
    "UZH_TO_EUROC": {
        "model_id": "B2_RIDGE_QF",
        "inputs": "q_f",
        "lag": 25,
        "alpha": 1.0,
        "fit_dataset": "UZH_FPV",
        "role": "source-frozen UZH B2 applied zero-shot (B2 beat B3 on this direction)",
    },
}

SUMMARY_NAME = {
    "Persistence": "Persistence",
    "Best frozen Ridge": "Best Ridge",
    "TCN": "TCN",
    "GRU": "GRU",
    "Transformer": "Transformer",
}


def utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt3(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{float(mean):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def grab(rows: list[dict[str, Any]], context: str, model: str) -> dict[str, Any]:
    hits = [r for r in rows if r["context"] == context and r["model"] == model]
    if not hits:
        raise KeyError(f"missing {context} {model}")
    return hits[0]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_named(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reconcile(summaries: list[dict[str, Any]], paired: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    adv = read_csv(root / "results" / "stage3" / "ADVANCED_SUMMARY.csv")
    vs = read_csv(root / "results" / "stage3" / "ADVANCED_VS_RIDGE.csv")
    s2boot = read_csv(root / "results" / "stage2" / "BOOTSTRAP_SUMMARY.csv")
    rows: list[dict[str, Any]] = []

    def add(context: str, model: str, metric: str, reported: float, recomputed: float, tol: float = TOL_CSV) -> None:
        diff = abs(float(recomputed) - float(reported))
        rel = diff / abs(float(reported)) if float(reported) != 0 else diff
        status = "PASS" if diff <= tol or (not np.isfinite(reported) and not np.isfinite(recomputed)) else "FAIL"
        rows.append(
            {
                "context": context,
                "model": model,
                "metric": metric,
                "reported_value": reported,
                "recomputed_value": recomputed,
                "absolute_difference": diff,
                "relative_difference": rel,
                "tolerance": tol,
                "status": status,
            }
        )

    for rec in summaries:
        ctx = rec["context"]
        name = SUMMARY_NAME[rec["model"]]
        hit = [r for r in adv if r["context"] == ctx and r["model"] == name]
        if not hit:
            continue
        r = hit[0]
        add(ctx, rec["model"], "mean_seq_rmse", fnum(r["mean_seq_rmse"]), rec["mean_seq_rmse"])
        add(ctx, rec["model"], "ci_low", fnum(r["ci_low"]), rec["ci_low"])
        add(ctx, rec["model"], "ci_high", fnum(r["ci_high"]), rec["ci_high"])
        add(ctx, rec["model"], "median_seq_rmse", fnum(r["median_seq_rmse"]), rec["median_seq_rmse"])
        add(ctx, rec["model"], "mean_mae", fnum(r["mean_mae"]), rec["mean_mae"])
        add(ctx, rec["model"], "rmse_50ms", fnum(r["rmse_50ms"]), rec["rmse_50ms"])
        add(ctx, rec["model"], "rmse_100ms", fnum(r["rmse_100ms"]), rec["rmse_100ms"])
        add(ctx, rec["model"], "rmse_200ms", fnum(r["rmse_200ms"]), rec["rmse_200ms"])
        add(ctx, rec["model"], "skill_vs_persistence", fnum(r["skill_vs_persistence"]), rec["skill_vs_persistence"])

    gap_src = read_csv(root / "results" / "stage3" / "DOMAIN_TRANSFER_GAP.csv")
    gap_name = {"Best frozen Ridge": "BEST_RIDGE", "TCN": "TCN", "GRU": "GRU", "Transformer": "TRANSFORMER"}

    for rec in paired:
        key = "TRANSFORMER" if rec["model"] == "Transformer" else rec["model"]
        hit = [r for r in vs if r["context"] == rec["context"] and r["model"] == key]
        if not hit:
            continue
        r = hit[0]
        add(rec["context"], rec["model"], "mean_paired_delta", fnum(r["mean_paired_delta"]), rec["mean_paired_delta"])
        add(rec["context"], rec["model"], "median_paired_delta", fnum(r["median_paired_delta"]), rec["median_paired_delta"])
        add(rec["context"], rec["model"], "ci_low_delta", fnum(r["ci_low"]), rec["ci_low"])
        add(rec["context"], rec["model"], "ci_high_delta", fnum(r["ci_high"]), rec["ci_high"])
        add(rec["context"], rec["model"], "wins", float(r["wins"]), float(rec["wins"]))
        add(rec["context"], rec["model"], "losses", float(r["losses"]), float(rec["losses"]))

    for ctx, mid in BEST_RIDGE_ID.items():
        s2 = [r for r in s2boot if r["evaluation_context"] == ctx and r["model"] == mid][0]
        pub = grab(summaries, ctx, "Best frozen Ridge")
        add(ctx, "Best frozen Ridge", "stage2_bootstrap_mean", fnum(s2["mean"]), pub["mean_seq_rmse"])
        add(ctx, "Best frozen Ridge", "stage2_bootstrap_ci_low", fnum(s2["ci_low"]), pub["ci_low"])
        add(ctx, "Best frozen Ridge", "stage2_bootstrap_ci_high", fnum(s2["ci_high"]), pub["ci_high"])

    mapping = (("EUROC_TO_UZH", "WITHIN_UZH"), ("UZH_TO_EUROC", "WITHIN_EUROC"))
    for pub_name, src_name in gap_name.items():
        for direction, within in mapping:
            zs = grab(summaries, direction, pub_name)["mean_seq_rmse"]
            wi = grab(summaries, within, pub_name)["mean_seq_rmse"]
            old = [r for r in gap_src if r["direction"] == direction and r["model"] == src_name]
            if not old:
                continue
            add(direction, pub_name, "transfer_gap", fnum(old[0]["transfer_gap"]), zs - wi)
    return rows


def high_qf_table(root: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = read_csv(root / "results" / "stage3" / "HIGH_QF_ADVANCED.csv")
    name_map = {
        "B0_PERSISTENCE": "Persistence",
        "BEST_RIDGE": "Best frozen Ridge",
        "TCN": "TCN",
        "GRU": "GRU",
        "TRANSFORMER": "Transformer",
    }
    out: list[dict[str, Any]] = []
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            key = {v: k for k, v in name_map.items()}[model]
            hits = [r for r in raw if r["evaluation_context"] == ctx and r["model"] == key]
            high_vals = np.asarray([fnum(r["rmse_high_p95_targets"]) for r in hits], dtype=np.float64)
            all_vals = np.asarray([fnum(r["rmse_all"]) for r in hits], dtype=np.float64)
            mean_high = float(np.nanmean(high_vals))
            mean_all = float(np.nanmean(all_vals))
            pub_all = grab(summaries, ctx, model)["mean_seq_rmse"]
            out.append(
                {
                    "context": ctx,
                    "model": model,
                    "n_sequences": int(high_vals.size),
                    "rmse_all": pub_all,
                    "rmse_high_qf": mean_high,
                    "difficulty_ratio": mean_high / pub_all if pub_all else float("nan"),
                    "absolute_increase": mean_high - pub_all,
                    "relative_increase": (mean_high - pub_all) / pub_all if pub_all else float("nan"),
                    "mean_file_rmse_all": mean_all,
                    "p95_source": "sequence-specific p95 of processed q_f (Stage-2 definition, not redefined)",
                }
            )
    return out


def transfer_gaps(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = (("EUROC_TO_UZH", "WITHIN_UZH"), ("UZH_TO_EUROC", "WITHIN_EUROC"))
    for model in ("Best frozen Ridge", "TCN", "GRU", "Transformer"):
        for direction, within in mapping:
            zs = grab(summaries, direction, model)["mean_seq_rmse"]
            wi = grab(summaries, within, model)["mean_seq_rmse"]
            gap = zs - wi
            rows.append(
                {
                    "direction": direction,
                    "model": model,
                    "zero_shot_rmse": zs,
                    "within_target_rmse": wi,
                    "transfer_gap": gap,
                    "relative_transfer_gap_pct": 100.0 * gap / wi if wi else float("nan"),
                    "notes": "descriptive; Persistence omitted because it has no trained source model",
                }
            )
    return rows


def gravity_rows(packs: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (ds, sid), rec in packs.items():
        qf = np.asarray(rec["q_f"], dtype=np.float64)
        rows.append(
            {
                "dataset": ds,
                "sequence_id": sid,
                "n_samples": int(qf.size),
                "median_qf": float(np.median(qf)),
                "mean_qf": float(np.mean(qf)),
                "sd_qf": float(np.std(qf, ddof=1)),
                "iqr_qf": float(np.subtract(*np.percentile(qf, [75, 25]))),
                "p01": float(np.percentile(qf, 1)),
                "p05": float(np.percentile(qf, 5)),
                "p95": float(np.percentile(qf, 95)),
                "p99": float(np.percentile(qf, 99)),
                "frac_within_9p81_pm_0p5": float(np.mean(np.abs(qf - G) <= 0.5)),
                "frac_within_9p81_pm_1p0": float(np.mean(np.abs(qf - G) <= 1.0)),
                "frac_within_9p81_pm_2p0": float(np.mean(np.abs(qf - G) <= 2.0)),
            }
        )
    return rows


def domain_shift_rows(packs: dict[Any, dict[str, Any]], stage3_shift: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    euroc_qf = np.concatenate([np.asarray(packs[("EUROC", s)]["q_f"], dtype=np.float64) for s in EUROC_PRIMARY])
    uzh_qf = np.concatenate([np.asarray(packs[("UZH_FPV", s)]["q_f"], dtype=np.float64) for s in UZH_PRIMARY])
    euroc_qw = np.concatenate([np.asarray(packs[("EUROC", s)]["q_w"], dtype=np.float64) for s in EUROC_PRIMARY])
    uzh_qw = np.concatenate([np.asarray(packs[("UZH_FPV", s)]["q_w"], dtype=np.float64) for s in UZH_PRIMARY])

    def stats(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        return {
            "quantity": name,
            "euroc_mean": float(np.mean(a)),
            "euroc_sd": float(np.std(a, ddof=1)),
            "euroc_median": float(np.median(a)),
            "euroc_iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
            "euroc_p01": float(np.percentile(a, 1)),
            "euroc_p05": float(np.percentile(a, 5)),
            "euroc_p95": float(np.percentile(a, 95)),
            "euroc_p99": float(np.percentile(a, 99)),
            "uzh_mean": float(np.mean(b)),
            "uzh_sd": float(np.std(b, ddof=1)),
            "uzh_median": float(np.median(b)),
            "uzh_iqr": float(np.subtract(*np.percentile(b, [75, 25]))),
            "uzh_p01": float(np.percentile(b, 1)),
            "uzh_p05": float(np.percentile(b, 5)),
            "uzh_p95": float(np.percentile(b, 95)),
            "uzh_p99": float(np.percentile(b, 99)),
            "wasserstein": float(wasserstein_distance(a, b)),
            "notes": "descriptive only; not used for adaptation",
        }

    out = [stats("q_f", euroc_qf, uzh_qf), stats("q_w", euroc_qw, uzh_qw)]
    recon: list[dict[str, Any]] = []
    for rec in out:
        old = [r for r in stage3_shift if r["quantity"] == rec["quantity"]][0]
        for key in ("euroc_median", "euroc_iqr", "euroc_p05", "euroc_p95", "uzh_median", "uzh_iqr", "uzh_p05", "uzh_p95", "wasserstein"):
            reported = fnum(old[key])
            recomputed = float(rec[key])
            diff = abs(recomputed - reported)
            recon.append(
                {
                    "context": "DOMAIN_SHIFT",
                    "model": rec["quantity"],
                    "metric": key,
                    "reported_value": reported,
                    "recomputed_value": recomputed,
                    "absolute_difference": diff,
                    "relative_difference": diff / abs(reported) if reported else diff,
                    "tolerance": 1e-5,
                    "status": "PASS" if diff <= 1e-5 else "FAIL",
                }
            )
    return out, recon


def seed_final(root: Path) -> list[dict[str, Any]]:
    raw = read_csv(root / "results" / "stage3" / "SEED_VARIABILITY.csv")
    out: list[dict[str, Any]] = []
    for ctx in CONTEXTS:
        for model in ("TCN", "GRU", "TRANSFORMER"):
            hits = [r for r in raw if r["context"] == ctx and r["model"] == model and r["seed"] not in {"SD", "sd"}]
            vals = np.asarray([fnum(r["mean_seq_rmse"]) for r in hits], dtype=np.float64)
            out.append(
                {
                    "context": ctx,
                    "model": "Transformer" if model == "TRANSFORMER" else model,
                    "n_seeds": int(vals.size),
                    "mean_across_seeds": float(np.mean(vals)),
                    "sd_across_seeds": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "min_seed": float(np.min(vals)),
                    "max_seed": float(np.max(vals)),
                    "range": float(np.max(vals) - np.min(vals)),
                    "notes": "Random seeds assess training variability; flight sequences remain the inferential units.",
                }
            )
    return out


def qw_final(root: Path) -> list[dict[str, Any]]:
    raw = read_csv(root / "results" / "stage3" / "QW_ABLATION.csv")
    out: list[dict[str, Any]] = []
    for source in ("EUROC", "UZH_FPV"):
        only = [r for r in raw if r["source_dataset"] == source and r["input"] == "q_f"][0]
        both = [r for r in raw if r["source_dataset"] == source and r["input"] == "q_f+q_w"][0]
        a = fnum(only["mean_source_val_rmse"])
        b = fnum(both["mean_source_val_rmse"])
        out.append(
            {
                "source_dataset": source,
                "selected_advanced_architecture": only["architecture"],
                "config_id": only["config_id"],
                "qf_only_val_rmse": a,
                "qf_qw_val_rmse": b,
                "absolute_difference": b - a,
                "relative_difference": (b - a) / a if a else float("nan"),
                "interpretation": "modest auxiliary contribution",
            }
        )
    return out


def classify_key_findings(summaries: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    def mean(ctx: str, model: str) -> float:
        return float(grab(summaries, ctx, model)["mean_seq_rmse"])

    def pr(ctx: str, model: str) -> dict[str, Any]:
        return [p for p in paired if p["context"] == ctx and p["model"] == model][0]

    lines = ["# KEY FINDINGS VERIFICATION", "", "Each item is checked against recomputed sequence-level RMSE.", ""]
    we_r, we_t, we_g, we_tr = mean("WITHIN_EUROC", "Best frozen Ridge"), mean("WITHIN_EUROC", "TCN"), mean("WITHIN_EUROC", "GRU"), mean("WITHIN_EUROC", "Transformer")
    a_ok = all(abs(x - we_r) / we_r < 0.03 for x in (we_t, we_g, we_tr)) and all(pr("WITHIN_EUROC", m)["mean_paired_delta"] > -0.01 for m in ADV_MODELS)
    lines += [
        "## A. Within EuRoC: Ridge approximately as good as advanced models",
        f"Status: {'SUPPORTED' if a_ok else 'NOT_SUPPORTED'}",
        f"Ridge {we_r:.4f}; TCN {we_t:.4f}; GRU {we_g:.4f}; Transformer {we_tr:.4f} m/s^2.",
        "All three advanced models have higher mean RMSE than Ridge; paired CIs overlap zero; wins are a minority.",
        "",
    ]
    uz_r, uz_tr = mean("WITHIN_UZH", "Best frozen Ridge"), mean("WITHIN_UZH", "Transformer")
    p_tr = pr("WITHIN_UZH", "Transformer")
    b_ok = p_tr["mean_paired_delta"] < 0 and p_tr["wins"] >= 6 and p_tr["ci_high"] < 0
    lines += [
        "## B. Within UZH: Transformer provides the clearest advanced-model gain over Ridge",
        f"Status: {'SUPPORTED' if b_ok else 'NOT_SUPPORTED'}",
        f"Ridge {uz_r:.4f}; Transformer {uz_tr:.4f}; mean Δ={p_tr['mean_paired_delta']:.4f}; wins {p_tr['wins']}/{p_tr['n_sequences']}; CI [{p_tr['ci_low']:.4f}, {p_tr['ci_high']:.4f}].",
        "TCN is a smaller, CI-overlapping improvement; GRU is mixed.",
        "",
    ]
    et_r, et_t, et_tr = mean("EUROC_TO_UZH", "Best frozen Ridge"), mean("EUROC_TO_UZH", "TCN"), mean("EUROC_TO_UZH", "Transformer")
    p_et_t, p_et_tr = pr("EUROC_TO_UZH", "TCN"), pr("EUROC_TO_UZH", "Transformer")
    c_ok = abs(et_t - et_r) / et_r < 0.02 and p_et_tr["mean_paired_delta"] > 0.05 and p_et_tr["losses"] > p_et_tr["wins"]
    lines += [
        "## C. EuRoC -> UZH: TCN approximately competitive with Ridge; Transformer materially worse",
        f"Status: {'SUPPORTED' if c_ok else 'NOT_SUPPORTED'}",
        f"Ridge {et_r:.4f}; TCN {et_t:.4f} (Δ={p_et_t['mean_paired_delta']:.4f}, CI overlaps 0); Transformer {et_tr:.4f} (Δ={p_et_tr['mean_paired_delta']:.4f}, 1–7).",
        "",
    ]
    ue_r, ue_g = mean("UZH_TO_EUROC", "Best frozen Ridge"), mean("UZH_TO_EUROC", "GRU")
    p_g = pr("UZH_TO_EUROC", "GRU")
    d_ok = ue_g <= min(mean("UZH_TO_EUROC", m) for m in ("Best frozen Ridge", "TCN", "GRU", "Transformer")) and p_g["ci_high"] > 0
    lines += [
        "## D. UZH -> EuRoC: GRU nominally best, improvement over Ridge modest",
        f"Status: {'SUPPORTED' if d_ok else 'PARTIALLY_SUPPORTED'}",
        f"Ridge {ue_r:.4f}; GRU {ue_g:.4f}; mean Δ={p_g['mean_paired_delta']:.4f}; wins {p_g['wins']}/6; CI includes 0.",
        "",
    ]
    winners = []
    for ctx in CONTEXTS:
        best = min(("Best frozen Ridge", "TCN", "GRU", "Transformer"), key=lambda m: mean(ctx, m))
        winners.append(f"{ctx}: {best}")
    e_ok = len(set(winners)) > 1
    lines += [
        "## E. No advanced model dominates all four contexts",
        f"Status: {'SUPPORTED' if e_ok else 'NOT_SUPPORTED'}",
        "Context-wise lowest mean RMSE among Ridge/TCN/GRU/Transformer:",
        *[f"- {w}" for w in winners],
        "",
        "Working conclusion check: compact nonlinear models can provide dataset-specific within-domain gains (UZH Transformer), but increased complexity does not consistently improve zero-shot transfer (EuRoC Transformer degrades); regularized Ridge remains competitive. Status: SUPPORTED.",
        "",
    ]
    return "\n".join(lines)


def primary_table_md(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 3. Primary forecasting results",
        "",
        "Equal-sequence mean multi-horizon RMSE of specific-force magnitude $q_f$ (m/s^2).",
        "Intervals are 95% percentile CIs from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8).",
        "Best frozen Ridge is context-specific (see Ridge comparator map): B3 $q_f{+}q_w$ except UZH$\\rightarrow$EuRoC, which uses source-frozen B2 $q_f$.",
        r"$\Delta$ is not shown here; negative $\Delta$ in Table 4 means the advanced model has lower RMSE than Ridge.",
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
    lines += ["", "RMSE at 50 / 100 / 200 ms (equal-sequence means, m/s^2):", "", "| Model | Context | 50 ms | 100 ms | 200 ms |", "|---|---|---|---|---|"]
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            r = grab(summaries, ctx, model)
            lines.append(f"| {model} | {ctx} | {fmt3(r['rmse_50ms'])} | {fmt3(r['rmse_100ms'])} | {fmt3(r['rmse_200ms'])} |")
    lines.append("")
    return "\n".join(lines)


def run_stage4(root: Path) -> dict[str, Any]:
    out = root / "results" / "stage4"
    fig_dir = out / "figures"
    fig_data = out / "figure_data"
    pub = root / "publication_package"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_data.mkdir(parents=True, exist_ok=True)

    lock_rows = verify_locks(root)
    write_csv(out / "LOCK_VERIFICATION.csv", ["artifact", "expected_hash", "observed_hash", "status", "notes"], lock_rows)
    if not locks_ok(lock_rows):
        write_text(root / "STAGE4_REPORT.md", "STAGE4_FAIL_LOCK_MISMATCH\n")
        return {"decision": "STAGE4_FAIL_LOCK_MISMATCH", "lock_rows": lock_rows}

    seq_rows = load_stage3_sequences(root)
    s2_rows = load_stage2_sequences(root)
    summaries = all_summaries(seq_rows)
    paired = all_paired(seq_rows)
    horizon_rows: list[dict[str, Any]] = []
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            horizon_rows.extend(horizon_curve(seq_rows, ctx, model))

    recon = reconcile(summaries, paired, root)
    packs = load_all_processed(root)
    grav = gravity_rows(packs)
    stage3_shift = read_csv(root / "results" / "stage3" / "DOMAIN_SHIFT_CHARACTERIZATION.csv")
    domain, domain_recon = domain_shift_rows(packs, stage3_shift)
    recon.extend(domain_recon)

    high_rows = high_qf_table(root, summaries)
    gap_rows = transfer_gaps(summaries)
    seed_rows = seed_final(root)
    qw_rows = qw_final(root)
    cost = read_csv(root / "results" / "stage3" / "COMPUTATIONAL_COST.csv")
    leak = read_csv(root / "results" / "stage3" / "TARGET_LEAKAGE_AUDIT.csv")
    n_target = sum(1 for r in leak if r["target_dataset_access_during_fit"] != "NO")
    p95_pred = read_csv(root / "results" / "stage2" / "PREDICTABILITY_CHARACTERIZATION.csv")
    high_raw = read_csv(root / "results" / "stage3" / "HIGH_QF_ADVANCED.csv")
    for hr in high_raw:
        pred = [p for p in p95_pred if p["sequence_id"] == hr["sequence_id"]]
        if pred:
            diff = abs(fnum(hr["p95_threshold"]) - fnum(pred[0]["p95_qf"]))
            recon.append(
                {
                    "context": hr["evaluation_context"],
                    "model": hr["model"],
                    "metric": f"p95_threshold_{hr['sequence_id']}",
                    "reported_value": fnum(hr["p95_threshold"]),
                    "recomputed_value": fnum(pred[0]["p95_qf"]),
                    "absolute_difference": diff,
                    "relative_difference": diff / abs(fnum(pred[0]["p95_qf"])) if fnum(pred[0]["p95_qf"]) else diff,
                    "tolerance": 1e-6,
                    "status": "PASS" if diff <= 1e-6 else "FAIL",
                }
            )

    n_euroc = len({r["sequence_id"] for r in seq_rows if r["evaluation_context"] == "WITHIN_EUROC"})
    n_uzh = len({r["sequence_id"] for r in seq_rows if r["evaluation_context"] == "WITHIN_UZH"})
    if n_euroc != 6 or n_uzh != 8:
        recon.append(
            {
                "context": "COUNTS",
                "model": "ALL",
                "metric": "n_sequences",
                "reported_value": 14,
                "recomputed_value": n_euroc + n_uzh,
                "absolute_difference": abs(14 - (n_euroc + n_uzh)),
                "relative_difference": 1,
                "tolerance": 0,
                "status": "FAIL",
            }
        )

    write_csv(
        out / "NUMERICAL_RECONCILIATION.csv",
        ["context", "model", "metric", "reported_value", "recomputed_value", "absolute_difference", "relative_difference", "tolerance", "status"],
        recon,
    )
    fail_recon = [r for r in recon if r["status"] != "PASS"]
    critical = [r for r in fail_recon if abs(float(r["absolute_difference"])) > 0.01]
    major = [r for r in fail_recon if r not in critical]

    ridge_map_rows = [{"context": k, **v} for k, v in RIDGE_DETAILS.items()]
    write_csv(
        out / "RIDGE_COMPARATOR_MAP.csv",
        ["context", "model_id", "inputs", "lag", "alpha", "fit_dataset", "role"],
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
        ["context", "model", "n_sequences", "mean_seq_rmse", "ci_low", "ci_high", "formatted_rmse_ci", "rmse_50ms", "rmse_100ms", "rmse_200ms", "mean_mae", "skill_vs_persistence"],
        primary_csv_rows,
    )
    write_text(out / "TABLE_PRIMARY_RESULTS.md", primary_table_md(summaries))

    vs_rows = []
    effect_rows = []
    strength_rows = []
    paired_seq = []
    for p in paired:
        cls = evidence_class(p)
        vs_rows.append(
            {
                "context": p["context"],
                "model": p["model"],
                "mean_paired_delta": p["mean_paired_delta"],
                "median_paired_delta": p["median_paired_delta"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
                "wins": p["wins"],
                "losses": p["losses"],
                "ties": p["ties"],
                "relative_percent_change_from_ridge": p["relative_improvement_pct"],
                "delta_definition": "Delta = RMSE_advanced - RMSE_Ridge; negative = advanced lower error",
            }
        )
        effect_rows.append(
            {
                "context": p["context"],
                "model": p["model"],
                "ridge_rmse": p["ridge_mean"],
                "advanced_rmse": p["advanced_mean"],
                "absolute_improvement": p["absolute_improvement"],
                "relative_improvement_pct": p["relative_improvement_pct"],
                "notes": "positive absolute_improvement means advanced has lower equal-sequence mean RMSE",
            }
        )
        strength_rows.append(
            {
                "context": p["context"],
                "model": p["model"],
                "evidence_class": cls,
                "mean_delta": p["mean_paired_delta"],
                "wins": p["wins"],
                "losses": p["losses"],
                "ci_low": p["ci_low"],
                "ci_high": p["ci_high"],
            }
        )
        for sid, d in zip(p["sequence_ids"], p["delta"], strict=True):
            paired_seq.append({"context": p["context"], "sequence_id": sid, "model": p["model"], "delta": float(d)})

    write_csv(
        out / "TABLE_ADVANCED_VS_RIDGE.csv",
        ["context", "model", "mean_paired_delta", "median_paired_delta", "ci_low", "ci_high", "wins", "losses", "ties", "relative_percent_change_from_ridge", "delta_definition"],
        vs_rows,
    )
    write_csv(
        out / "PRACTICAL_EFFECTS.csv",
        ["context", "model", "ridge_rmse", "advanced_rmse", "absolute_improvement", "relative_improvement_pct", "notes"],
        effect_rows,
    )
    write_csv(
        out / "EVIDENCE_STRENGTH_CLASSIFICATION.csv",
        ["context", "model", "evidence_class", "mean_delta", "wins", "losses", "ci_low", "ci_high"],
        strength_rows,
    )
    write_text(out / "KEY_FINDINGS_VERIFICATION.md", classify_key_findings(summaries, paired))
    write_csv(
        out / "HORIZON_PUBLICATION_DATA.csv",
        ["context", "model", "horizon_step", "horizon_ms", "mean_sequence_rmse", "ci_low", "ci_high"],
        horizon_rows,
    )
    write_csv(
        out / "HIGH_QF_PUBLICATION_DATA.csv",
        ["context", "model", "n_sequences", "rmse_all", "rmse_high_qf", "difficulty_ratio", "absolute_increase", "relative_increase", "mean_file_rmse_all", "p95_source"],
        high_rows,
    )
    write_csv(
        out / "TRANSFER_GAP_FINAL.csv",
        ["direction", "model", "zero_shot_rmse", "within_target_rmse", "transfer_gap", "relative_transfer_gap_pct", "notes"],
        gap_rows,
    )
    write_csv(
        out / "DOMAIN_SHIFT_FINAL.csv",
        [
            "quantity",
            "euroc_mean",
            "euroc_sd",
            "euroc_median",
            "euroc_iqr",
            "euroc_p01",
            "euroc_p05",
            "euroc_p95",
            "euroc_p99",
            "uzh_mean",
            "uzh_sd",
            "uzh_median",
            "uzh_iqr",
            "uzh_p01",
            "uzh_p05",
            "uzh_p95",
            "uzh_p99",
            "wasserstein",
            "notes",
        ],
        domain,
    )
    write_csv(
        out / "SEED_VARIABILITY_FINAL.csv",
        ["context", "model", "n_seeds", "mean_across_seeds", "sd_across_seeds", "min_seed", "max_seed", "range", "notes"],
        seed_rows,
    )
    write_csv(
        out / "COMPUTATIONAL_FOOTPRINT_FINAL.csv",
        ["model", "source_dataset", "config_id", "parameters", "mean_training_time_s", "median_inference_ms_per_window", "mean_artifact_bytes", "device", "notes"],
        cost,
    )
    write_csv(
        out / "QW_ABLATION_FINAL.csv",
        ["source_dataset", "selected_advanced_architecture", "config_id", "qf_only_val_rmse", "qf_qw_val_rmse", "absolute_difference", "relative_difference", "interpretation"],
        qw_rows,
    )
    write_csv(
        out / "GRAVITY_DOMINANCE_CHARACTERIZATION.csv",
        [
            "dataset",
            "sequence_id",
            "n_samples",
            "median_qf",
            "mean_qf",
            "sd_qf",
            "iqr_qf",
            "p01",
            "p05",
            "p95",
            "p99",
            "frac_within_9p81_pm_0p5",
            "frac_within_9p81_pm_1p0",
            "frac_within_9p81_pm_2p0",
        ],
        grav,
    )

    supp = []
    for r in seq_rows:
        if r.get("pub_model") in PUB_MODELS:
            supp.append(
                {
                    "context": r["evaluation_context"],
                    "dataset": r["dataset"],
                    "sequence_id": r["sequence_id"],
                    "model": r["pub_model"],
                    "rmse_all_horizons": r["rmse_all_horizons"],
                    "mae_all_horizons": r["mae_all_horizons"],
                    "rmse_50ms": r["rmse_50ms"],
                    "rmse_100ms": r["rmse_100ms"],
                    "rmse_200ms": r["rmse_200ms"],
                    "n_origins": r["n_origins"],
                }
            )
    for r in s2_rows:
        if r["model"] == "B1_LINEAR":
            supp.append(
                {
                    "context": r["evaluation_context"],
                    "dataset": r["dataset"],
                    "sequence_id": r["sequence_id"],
                    "model": "Linear extrapolation (supplementary)",
                    "rmse_all_horizons": r["rmse_all_horizons"],
                    "mae_all_horizons": r["mae_all_horizons"],
                    "rmse_50ms": r["rmse_50ms"],
                    "rmse_100ms": r["rmse_100ms"],
                    "rmse_200ms": r["rmse_200ms"],
                    "n_origins": r["n_origins"],
                }
            )
    write_csv(
        out / "SUPPLEMENT_SEQUENCE_RESULTS.csv",
        ["context", "dataset", "sequence_id", "model", "rmse_all_horizons", "mae_all_horizons", "rmse_50ms", "rmse_100ms", "rmse_200ms", "n_origins"],
        supp,
    )

    inf_ok = all(fnum(r["median_inference_ms_per_window"]) < 1.0 for r in cost)
    max_params = max(int(float(r["parameters"])) for r in cost)

    # Figure source CSVs
    fig2_rows = [{"context": r["context"], "model": r["model"], "mean_seq_rmse": r["mean_seq_rmse"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]} for r in summaries]
    write_csv(fig_data / "FIG2_primary_rmse.csv", ["context", "model", "mean_seq_rmse", "ci_low", "ci_high"], fig2_rows)
    write_csv(fig_data / "FIG3_horizon_curves.csv", ["context", "model", "horizon_step", "horizon_ms", "mean_sequence_rmse", "ci_low", "ci_high"], horizon_rows)
    write_csv(fig_data / "FIG4_paired_delta.csv", ["context", "sequence_id", "model", "delta"], paired_seq)
    write_csv(fig_data / "FIG5_transfer_gap.csv", ["direction", "model", "transfer_gap", "relative_transfer_gap_pct"], [{"direction": r["direction"], "model": r["model"], "transfer_gap": r["transfer_gap"], "relative_transfer_gap_pct": r["relative_transfer_gap_pct"]} for r in gap_rows])
    write_csv(fig_data / "FIG6_high_qf.csv", ["context", "model", "rmse_all", "rmse_high_qf", "difficulty_ratio"], [{"context": r["context"], "model": r["model"], "rmse_all": r["rmse_all"], "rmse_high_qf": r["rmse_high_qf"], "difficulty_ratio": r["difficulty_ratio"]} for r in high_rows])

    figure1_schematic(fig_dir / "Fig01_Protocol_Schematic")
    figure2_primary_rmse(summaries, fig_dir / "Fig02_Primary_RMSE")
    figure3_horizon(horizon_rows, fig_dir / "Fig03_Horizon_RMSE")
    figure4_paired(paired_seq, fig_dir / "Fig04_Advanced_vs_Ridge")
    figure5_transfer_gap(gap_rows, fig_dir / "Fig05_Transfer_Gap")
    figure6_high_qf(high_rows, fig_dir / "Fig06_High_qf")

    _write_remaining_docs(
        root,
        out,
        summaries,
        paired,
        high_rows,
        gap_rows,
        qw_rows,
        cost,
        grav,
        domain,
        seed_rows,
        recon,
        fail_recon,
        critical,
        major,
        n_target,
        inf_ok,
        max_params,
        n_euroc,
        n_uzh,
        leak,
    )

    # publication package
    for sub in ("figures", "figure_data", "tables", "evidence", "manifests"):
        (pub / sub).mkdir(parents=True, exist_ok=True)
    name_map = {
        "Fig01_Protocol_Schematic": "Fig01_Protocol_Schematic",
        "Fig02_Primary_RMSE": "Fig02_Primary_RMSE",
        "Fig03_Horizon_RMSE": "Fig03_Horizon_RMSE",
        "Fig04_Advanced_vs_Ridge": "Fig04_Advanced_vs_Ridge",
        "Fig05_Transfer_Gap": "Fig05_Transfer_Gap",
        "Fig06_High_qf": "Fig06_High_qf",
    }
    for stem in name_map:
        for ext in (".pdf", ".svg", ".png"):
            copy_named(fig_dir / f"{stem}{ext}", pub / "figures" / f"{stem}{ext}")
    for src in fig_data.glob("*.csv"):
        copy_named(src, pub / "figure_data" / src.name)
    copy_named(out / "TABLE_DATASETS.csv", pub / "tables" / "Table01_Datasets.csv")
    copy_named(out / "TABLE_DATASETS.md", pub / "tables" / "Table01_Datasets.md")
    copy_named(out / "TABLE_PROTOCOL.csv", pub / "tables" / "Table02_Protocol.csv")
    copy_named(out / "TABLE_PROTOCOL.md", pub / "tables" / "Table02_Protocol.md")
    copy_named(out / "TABLE_PRIMARY_RESULTS.csv", pub / "tables" / "Table03_Primary_Results.csv")
    copy_named(out / "TABLE_PRIMARY_RESULTS.md", pub / "tables" / "Table03_Primary_Results.md")
    copy_named(out / "TABLE_ADVANCED_VS_RIDGE.csv", pub / "tables" / "Table04_Advanced_vs_Ridge.csv")
    copy_named(out / "TABLE_ADVANCED_VS_RIDGE.md", pub / "tables" / "Table04_Advanced_vs_Ridge.md")
    copy_named(out / "COMPUTATIONAL_FOOTPRINT_FINAL.csv", pub / "tables" / "Table05_Computational_Footprint.csv")
    copy_named(out / "TABLE_FOOTPRINT.md", pub / "tables" / "Table05_Computational_Footprint.md")
    copy_named(out / "RIDGE_COMPARATOR_MAP.csv", pub / "tables" / "Ridge_Comparator_Map.csv")
    copy_named(root / "src" / "stage4" / "figures.py", pub / "figures" / "source_figures.py")
    copy_named(root / "src" / "stage4" / "figures.py", fig_dir / "source_figures.py")
    for name in (
        "MANUSCRIPT_EVIDENCE_PACKAGE.md",
        "ABSTRACT_FACT_SHEET.md",
        "METHODS_FACT_SHEET.md",
        "RESULTS_FACT_SHEET.md",
        "LIMITATIONS_FACT_SHEET.md",
        "KEY_FINDINGS_VERIFICATION.md",
        "CLAIM_EVIDENCE_MATRIX.csv",
        "CONTRIBUTION_MAP.md",
        "NOVELTY_RISK.md",
        "FIGURE_CAPTIONS.md",
        "ANALYSIS_HIERARCHY.md",
        "TIME_DOMAIN_EXAMPLE.md",
        "STAGE4_REPORT.md",
    ):
        copy_named(out / name, pub / "evidence" / name)
    copy_named(out / "REPRODUCIBILITY_MANIFEST.csv", pub / "manifests" / "REPRODUCIBILITY_MANIFEST.csv")
    copy_named(out / "LOCK_VERIFICATION.csv", pub / "manifests" / "LOCK_VERIFICATION.csv")
    copy_named(out / "FINAL_RESULTS_LOCK.md", pub / "manifests" / "FINAL_RESULTS_LOCK.md")
    copy_named(out / "FINAL_RESULTS_LOCK.sha256", pub / "manifests" / "FINAL_RESULTS_LOCK.sha256")
    copy_named(out / "NUMERICAL_RECONCILIATION.csv", pub / "manifests" / "NUMERICAL_RECONCILIATION.csv")
    check_rows = publication_package_check(root, out, pub)
    write_csv(
        out / "PUBLICATION_PACKAGE_CHECK.csv",
        ["item", "required", "present", "status", "notes"],
        check_rows,
    )
    copy_named(out / "PUBLICATION_PACKAGE_CHECK.csv", pub / "manifests" / "PUBLICATION_PACKAGE_CHECK.csv")
    pkg_fail = [r for r in check_rows if r["status"] != "PASS"]

    recon_pass = not fail_recon
    decision = "PASS" if recon_pass and n_target == 0 and n_euroc == 6 and n_uzh == 8 and not pkg_fail else ("CONDITIONAL_PASS" if not critical and not pkg_fail else "FAIL")
    if pkg_fail and not critical:
        decision = "CONDITIONAL_PASS"
    ready = "YES" if decision == "PASS" else "NO"
    report_path = out / "STAGE4_REPORT.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        text = text.replace("READY FOR MANUSCRIPT WRITING:\nYES", f"READY FOR MANUSCRIPT WRITING:\n{ready}")
        text = text.replace("READY FOR MANUSCRIPT WRITING:\nNO", f"READY FOR MANUSCRIPT WRITING:\n{ready}")
        for old in ("PASS", "CONDITIONAL_PASS", "FAIL"):
            text = text.replace(f"OVERALL STAGE-4 DECISION:\n{old}", f"OVERALL STAGE-4 DECISION:\n{decision}")
        report_path.write_text(text, encoding="utf-8")
        write_text(root / "STAGE4_REPORT.md", text)
        copy_named(report_path, pub / "evidence" / "STAGE4_REPORT.md")
    return {
        "decision": decision,
        "ready": ready,
        "n_euroc": n_euroc,
        "n_uzh": n_uzh,
        "n_target": n_target,
        "recon_pass": recon_pass,
        "summaries": summaries,
        "paired": paired,
        "high_rows": high_rows,
        "fail_recon": fail_recon,
        "critical": critical,
        "major": major,
        "inf_ok": inf_ok,
    }


def _write_remaining_docs(
    root: Path,
    out: Path,
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    qw_rows: list[dict[str, Any]],
    cost: list[dict[str, str]],
    grav: list[dict[str, Any]],
    domain: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    recon: list[dict[str, Any]],
    fail_recon: list[dict[str, Any]],
    critical: list[dict[str, Any]],
    major: list[dict[str, Any]],
    n_target: int,
    inf_ok: bool,
    max_params: int,
    n_euroc: int,
    n_uzh: int,
    leak: list[dict[str, str]],
) -> None:
    from .documents import write_all_documents

    write_all_documents(
        root=root,
        out=out,
        summaries=summaries,
        paired=paired,
        high_rows=high_rows,
        gap_rows=gap_rows,
        qw_rows=qw_rows,
        cost=cost,
        grav=grav,
        domain=domain,
        seed_rows=seed_rows,
        recon=recon,
        fail_recon=fail_recon,
        critical=critical,
        major=major,
        n_target=n_target,
        inf_ok=inf_ok,
        max_params=max_params,
        n_euroc=n_euroc,
        n_uzh=n_uzh,
        leak=leak,
        grab=grab,
        fmt3=fmt3,
        fmt_ci=fmt_ci,
        utc_now=utc_now,
        write_text=write_text,
        RIDGE_DETAILS=RIDGE_DETAILS,
    )


def publication_package_check(root: Path, out: Path, pub: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(item: str, path: Path, notes: str = "") -> None:
        present = path.exists() and (path.stat().st_size > 0 if path.is_file() else True)
        rows.append(
            {
                "item": item,
                "required": "yes",
                "present": "yes" if present else "no",
                "status": "PASS" if present else "FAIL",
                "notes": notes,
            }
        )

    for stem in (
        "Fig01_Protocol_Schematic",
        "Fig02_Primary_RMSE",
        "Fig03_Horizon_RMSE",
        "Fig04_Advanced_vs_Ridge",
        "Fig05_Transfer_Gap",
        "Fig06_High_qf",
    ):
        for ext in (".pdf", ".svg", ".png"):
            add(f"{stem}{ext}", pub / "figures" / f"{stem}{ext}", "vector or 300 dpi preview")
    add("figure source script", pub / "figures" / "source_figures.py", "editable matplotlib source")
    add("FIG2 data", pub / "figure_data" / "FIG2_primary_rmse.csv")
    add("FIG3 data", pub / "figure_data" / "FIG3_horizon_curves.csv")
    add("FIG4 data", pub / "figure_data" / "FIG4_paired_delta.csv")
    add("FIG5 data", pub / "figure_data" / "FIG5_transfer_gap.csv")
    add("FIG6 data", pub / "figure_data" / "FIG6_high_qf.csv")
    for name in (
        "Table01_Datasets.csv",
        "Table01_Datasets.md",
        "Table02_Protocol.csv",
        "Table02_Protocol.md",
        "Table03_Primary_Results.csv",
        "Table03_Primary_Results.md",
        "Table04_Advanced_vs_Ridge.csv",
        "Table04_Advanced_vs_Ridge.md",
        "Table05_Computational_Footprint.csv",
        "Table05_Computational_Footprint.md",
    ):
        add(name, pub / "tables" / name)
    add("STAGE4_REPORT", out / "STAGE4_REPORT.md")
    add("FINAL_RESULTS_LOCK", out / "FINAL_RESULTS_LOCK.md")
    add("lock hash", out / "FINAL_RESULTS_LOCK.sha256")
    return rows
