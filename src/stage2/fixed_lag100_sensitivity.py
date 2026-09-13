"""Fixed 100-lag Ridge B3 sensitivity. Does not retrain neural models.

Lag is locked at 100 samples (the same 1.0 s history as TCN/GRU/Transformer).
Only alpha is selected, from the original Stage-2 grid, with source-only
group-aware validation. Target-domain recordings never enter alpha selection
or scaler fitting.

Writes exclusively under results/sensitivity/. Primary Stage-2/3 files are
not modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from audit.hashing import sha256_file
from audit.stage0c import write_csv
from stage1.folds import EUROC_ROOM, UZH_GROUP
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY
from stage2.bootstrap import bootstrap_mean_ci
from stage2.io_processed import load_all_processed
from stage2.metrics import sequence_result_row
from stage2.models import fit_ridge, ridge_predict
from stage2.protocol import verify_protocol_lock
from stage2.selection import select_ridge_config
from stage2.windows import SequenceWindows, build_sequence_windows

FIXED_LAG = 100
MODEL_ID = "B3_RIDGE_QF_QW_FIXED_LAG100"
PRIMARY_RIDGE_ID = "B3_RIDGE_QF_QW"
CONTEXTS = ("WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC")
CTX_LABEL = {
    "WITHIN_EUROC": "Within EuRoC",
    "WITHIN_UZH": "Within UZH-FPV",
    "EUROC_TO_UZH": "EuRoC -> UZH-FPV",
    "UZH_TO_EUROC": "UZH-FPV -> EuRoC",
}
STAGE2_SEQ_FILES = {
    "WITHIN_EUROC": "WITHIN_EUROC_SEQUENCE_RESULTS.csv",
    "WITHIN_UZH": "WITHIN_UZH_SEQUENCE_RESULTS.csv",
    "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv",
    "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv",
}
PROTECTED_PRIMARY = (
    "configs/stage2_baselines.yaml",
    "configs/stage3_advanced.yaml",
    "results/stage2/SOURCE_MODELS_FROZEN.md",
    "results/stage2/BOOTSTRAP_SUMMARY.csv",
    "results/stage2/BASELINE_SUMMARY.csv",
    "results/stage2/WITHIN_EUROC_SEQUENCE_RESULTS.csv",
    "results/stage2/WITHIN_UZH_SEQUENCE_RESULTS.csv",
    "results/stage2/TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv",
    "results/stage2/TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv",
    "results/stage2/HYPERPARAMETER_SELECTION.csv",
    "results/stage3/WITHIN_EUROC_ADVANCED.csv",
    "results/stage3/WITHIN_UZH_ADVANCED.csv",
    "results/stage3/TRANSFER_EUROC_TO_UZH_ADVANCED.csv",
    "results/stage3/TRANSFER_UZH_TO_EUROC_ADVANCED.csv",
    "results/stage3/SELECTED_MODELS.csv",
    "results/stage3/STAGE3_SOURCE_MODELS_FROZEN.md",
)

SEQ_COLS = (
    [
        "evaluation_context",
        "dataset",
        "sequence_id",
        "model",
        "lag",
        "selected_alpha",
        "n_origins",
        "rmse_all_horizons",
        "mae_all_horizons",
        "rmse_50ms",
        "rmse_100ms",
        "rmse_200ms",
        "skill",
    ]
    + [f"rmse_h{i:02d}" for i in range(1, 21)]
    + ["fold_id", "notes"]
)
SEL_COLS = [
    "evaluation_context",
    "phase",
    "fold_id",
    "dataset",
    "model",
    "lag",
    "selected_alpha",
    "selection_score_rmse",
    "n_tied",
    "train_sequences",
    "held_out_sequence",
    "candidate_alphas",
    "notes",
]
SUM_COLS = [
    "evaluation_context",
    "model",
    "lag",
    "n_sequences",
    "mean_sequence_rmse",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "mean_rmse_50ms",
    "mean_rmse_100ms",
    "mean_rmse_200ms",
]


def snapshot_primary_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in PROTECTED_PRIMARY:
        path = root / rel
        out[rel] = sha256_file(path) if path.exists() else "MISSING"
    return out


def assert_primary_unchanged(root: Path, before: dict[str, str]) -> None:
    after = snapshot_primary_hashes(root)
    changed = [rel for rel, digest in before.items() if after.get(rel) != digest]
    if changed:
        raise RuntimeError("primary manuscript files were modified: " + ", ".join(changed))


def _load_packs(root: Path, cfg: dict[str, Any]) -> tuple[dict[str, SequenceWindows], dict[str, SequenceWindows]]:
    raw_map = load_all_processed(root)
    euroc: dict[str, SequenceWindows] = {}
    uzh: dict[str, SequenceWindows] = {}
    for (ds, seq), rec in raw_map.items():
        pack = build_sequence_windows(
            rec,
            lookback_samples=int(cfg["lookback_samples"]),
            horizon_samples=int(cfg["horizon_samples"]),
            stride_samples=int(cfg["stride_samples"]),
            high_percentile=float(cfg["high_qf_percentile"]),
        )
        if pack.hist_qf.shape[1] != FIXED_LAG:
            raise RuntimeError(f"{seq} history length {pack.hist_qf.shape[1]} != {FIXED_LAG}")
        if ds == "EUROC":
            euroc[seq] = pack
        else:
            uzh[seq] = pack
    return euroc, uzh


def _select_alpha(
    seq_ids: list[str],
    packs: dict[str, SequenceWindows],
    groups: dict[str, str],
    *,
    dataset: str,
    alphas: list[float],
    tie: float,
    context: str,
) -> dict[str, Any]:
    fit_log: list[dict[str, str]] = []
    chosen = select_ridge_config(
        seq_ids,
        packs,
        groups,
        dataset=dataset,
        model_name=MODEL_ID,
        lags=[FIXED_LAG],
        alphas=alphas,
        use_qw=True,
        tie_relative_tol=tie,
        fit_log=fit_log,
        context=context,
    )
    if int(chosen["lag"]) != FIXED_LAG:
        raise RuntimeError(f"lag drifted to {chosen['lag']}")
    if float(chosen["alpha"]) not in {float(a) for a in alphas}:
        raise RuntimeError(f"alpha {chosen['alpha']} not on the predeclared grid")
    target_tokens = ("indoor_", "outdoor_") if dataset == "EUROC" else ("V1_", "V2_")
    for row in fit_log:
        joined = row["fit_sequences"]
        if any(tok in joined for tok in target_tokens):
            raise RuntimeError(f"target sequences entered selection fit: {joined}")
        if dataset == "EUROC" and row["fit_dataset"] != "EUROC":
            raise RuntimeError("EuRoC selection used a non-EuRoC fit dataset")
        if dataset == "UZH_FPV" and row["fit_dataset"] != "UZH_FPV":
            raise RuntimeError("UZH selection used a non-UZH fit dataset")
    return chosen


def _eval_seq(
    seq: SequenceWindows,
    fitted,
    *,
    context: str,
    fold_id: str,
    alpha: float,
    notes: str,
) -> dict[str, Any]:
    pred = ridge_predict(fitted, seq)
    row = sequence_result_row(
        evaluation_context=context,
        dataset=seq.dataset,
        sequence_id=seq.sequence_id,
        model=MODEL_ID,
        pred=pred,
        actual=seq.future_qf,
        persistence_rmse=None,
        n_origins=seq.n_windows,
        extra={"fold_id": fold_id, "notes": notes},
    )
    row["lag"] = FIXED_LAG
    row["selected_alpha"] = float(alpha)
    return row


def _run_loro(
    *,
    dataset: str,
    seq_ids: tuple[str, ...],
    groups: dict[str, str],
    packs: dict[str, SequenceWindows],
    context: str,
    alphas: list[float],
    tie: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seq_rows: list[dict[str, Any]] = []
    sel_rows: list[dict[str, Any]] = []
    for i, held in enumerate(seq_ids, start=1):
        train_ids = [s for s in seq_ids if s != held]
        fold_id = f"{dataset}_LORO_{i:02d}_{held}"
        print(f"[lag100] {context} {fold_id}", flush=True)
        chosen = _select_alpha(
            train_ids,
            packs,
            groups,
            dataset=dataset,
            alphas=alphas,
            tie=tie,
            context=context,
        )
        sel_rows.append(
            {
                "evaluation_context": context,
                "phase": "WITHIN_NESTED_SELECTION",
                "fold_id": fold_id,
                "dataset": dataset,
                "model": MODEL_ID,
                "lag": FIXED_LAG,
                "selected_alpha": chosen["alpha"],
                "selection_score_rmse": chosen["score"],
                "n_tied": chosen["n_tied"],
                "train_sequences": ",".join(train_ids),
                "held_out_sequence": held,
                "candidate_alphas": ",".join(str(a) for a in alphas),
                "notes": "lag fixed at 100; alpha selected on source training recordings only; held-out excluded",
            }
        )
        fitted = fit_ridge(
            [packs[s] for s in train_ids],
            lag=FIXED_LAG,
            alpha=float(chosen["alpha"]),
            use_qw=True,
            dataset=dataset,
            model_name=MODEL_ID,
        )
        if held in fitted.train_sequences:
            raise RuntimeError(f"held-out {held} leaked into the LORO fit")
        seq_rows.append(
            _eval_seq(
                packs[held],
                fitted,
                context=context,
                fold_id=fold_id,
                alpha=float(chosen["alpha"]),
                notes="fixed lag=100; nested source-only alpha; held-out never trained",
            )
        )
    return seq_rows, sel_rows


def _summarize(rows: list[dict[str, Any]], n_reps: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for context in CONTEXTS:
        mrows = [r for r in rows if r["evaluation_context"] == context]
        rmses = np.array([float(r["rmse_all_horizons"]) for r in mrows], dtype=np.float64)
        boot = bootstrap_mean_ci(rmses, n_reps=n_reps, seed=seed)
        out.append(
            {
                "evaluation_context": context,
                "model": MODEL_ID,
                "lag": FIXED_LAG,
                "n_sequences": len(mrows),
                "mean_sequence_rmse": boot["mean"],
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "mean_rmse_50ms": float(np.mean([float(r["rmse_50ms"]) for r in mrows])),
                "mean_rmse_100ms": float(np.mean([float(r["rmse_100ms"]) for r in mrows])),
                "mean_rmse_200ms": float(np.mean([float(r["rmse_200ms"]) for r in mrows])),
            }
        )
    return out


def _primary_summary(root: Path, n_reps: int, seed: int) -> dict[str, dict[str, float]]:
    import csv

    out: dict[str, dict[str, float]] = {}
    for context, name in STAGE2_SEQ_FILES.items():
        with (root / "results" / "stage2" / name).open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["model"] == PRIMARY_RIDGE_ID]
        rmses = np.array([float(r["rmse_all_horizons"]) for r in rows], dtype=np.float64)
        boot = bootstrap_mean_ci(rmses, n_reps=n_reps, seed=seed)
        out[context] = {
            "n_sequences": float(len(rows)),
            "mean_sequence_rmse": boot["mean"],
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "mean_rmse_50ms": float(np.mean([float(r["rmse_50ms"]) for r in rows])),
            "mean_rmse_100ms": float(np.mean([float(r["rmse_100ms"]) for r in rows])),
            "mean_rmse_200ms": float(np.mean([float(r["rmse_200ms"]) for r in rows])),
        }
    return out


def _comparison_md(
    sens: list[dict[str, Any]],
    primary: dict[str, dict[str, float]],
    source_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Fixed 100-lag Ridge B3 sensitivity",
        "",
        "This is a **supplementary equal-history benchmark**. It is not a primary result.",
        "",
        "## Protocol",
        "",
        "- Model: Ridge B3 with inputs `[q_f, q_w]`.",
        "- History length **fixed at lag = 100 samples (1.0 s)**, the same lookback used by TCN / GRU / Transformer.",
        "- Only `alpha` is selected, from the original predeclared grid `[1e-6, 1e-4, 1e-2, 1, 100]`.",
        "- Within-domain LORO: lag stays 100; alpha is chosen with source-only group-aware validation on the training recordings of that fold. The held-out sequence is excluded.",
        "- Zero-shot transfer: alpha is chosen on the source dataset only, the Ridge and scaler are frozen, and the target is evaluated unchanged.",
        "- Scaler, equal-recording weights, prediction origins, 20-step horizon, sequence-level RMSE, and 10,000 sequence-bootstrap CIs (seed 20260912) match the primary study.",
        "- **No TCN, GRU, or Transformer was retrained.**",
        "- **Target-domain data were not used to choose alpha or any other fitted object.**",
        "- **Primary manuscript result files were not changed.** Comparison uses frozen Stage-2 `B3_RIDGE_QF_QW`.",
        "",
        "Delta = RMSE_fixed100 − RMSE_primary_B3. Positive delta means the forced 100-lag history is worse.",
        "",
        "## Source-only selected alpha (lag locked at 100)",
        "",
        "| Context | Phase | Fold | Selected alpha | Source val RMSE |",
        "|---|---|---|---|---|",
    ]
    for r in source_rows:
        if r["phase"] != "SOURCE_SELECTION_FOR_TRANSFER" and "LORO" not in str(r["fold_id"]):
            continue
        if r["phase"] == "SOURCE_SELECTION_FOR_TRANSFER":
            lines.append(
                f"| {r['evaluation_context']} | source freeze | {r['fold_id']} | {r['selected_alpha']} | {float(r['selection_score_rmse']):.6f} |"
            )
    lines += ["", "LORO fold alphas are in `RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv`.", ""]
    lines += [
        "## Equal-sequence mean RMSE (m/s^2)",
        "",
        "| Context | Primary B3 RMSE [95% CI] | Fixed lag-100 B3 RMSE [95% CI] | Delta (fixed100 − primary) |",
        "|---|---|---|---|",
    ]
    for context in CONTEXTS:
        p = primary[context]
        s = [r for r in sens if r["evaluation_context"] == context][0]
        delta = float(s["mean_sequence_rmse"]) - float(p["mean_sequence_rmse"])
        sign = "worse" if delta > 0 else ("better" if delta < 0 else "tied")
        lines.append(
            "| {lab} | {pm:.6f} [{plo:.6f}, {phi:.6f}] | {sm:.6f} [{slo:.6f}, {shi:.6f}] | {d:+.6f} ({sign}) |".format(
                lab=CTX_LABEL[context],
                pm=p["mean_sequence_rmse"],
                plo=p["bootstrap_ci_low"],
                phi=p["bootstrap_ci_high"],
                sm=s["mean_sequence_rmse"],
                slo=s["bootstrap_ci_low"],
                shi=s["bootstrap_ci_high"],
                d=delta,
                sign=sign,
            )
        )
    lines += [
        "",
        "50 / 100 / 200 ms equal-sequence means (m/s^2):",
        "",
        "| Context | Primary 50/100/200 | Fixed100 50/100/200 |",
        "|---|---|---|",
    ]
    for context in CONTEXTS:
        p = primary[context]
        s = [r for r in sens if r["evaluation_context"] == context][0]
        lines.append(
            f"| {CTX_LABEL[context]} | {p['mean_rmse_50ms']:.6f} / {p['mean_rmse_100ms']:.6f} / {p['mean_rmse_200ms']:.6f} | {s['mean_rmse_50ms']:.6f} / {s['mean_rmse_100ms']:.6f} / {s['mean_rmse_200ms']:.6f} |"
        )
    lines += [
        "",
        "## Explicit statements",
        "",
        "1. No neural model was retrained.",
        "2. Target data were not used for tuning.",
        "3. Primary manuscript results were not changed.",
        "",
    ]
    return "\n".join(lines)


def run_fixed_lag100_sensitivity(root: Path) -> dict[str, Any]:
    lock = verify_protocol_lock(root)
    if lock["match"] != "PASS":
        raise RuntimeError(f"protocol lock mismatch: {lock}")
    before = snapshot_primary_hashes(root)
    cfg = yaml.safe_load((root / "configs" / "stage2_baselines.yaml").read_text(encoding="utf-8"))
    if int(cfg["lookback_samples"]) != FIXED_LAG:
        raise RuntimeError("lookback_samples is not 100")
    if FIXED_LAG not in [int(x) for x in cfg["lags"]]:
        raise RuntimeError("lag 100 is not on the original predeclared lag grid")
    alphas = [float(a) for a in cfg["alphas"]]
    expected_alphas = [1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0]
    if [float(a) for a in alphas] != expected_alphas:
        raise RuntimeError(f"alpha grid drifted: {alphas}")
    n_reps = int(cfg["bootstrap_reps"])
    seed = int(cfg["bootstrap_seed"])
    tie = float(cfg["tie_relative_tol"])

    print("[lag100] loading processed sequences (no neural training)", flush=True)
    euroc_packs, uzh_packs = _load_packs(root, cfg)

    seq_rows: list[dict[str, Any]] = []
    sel_rows: list[dict[str, Any]] = []

    a, b = _run_loro(
        dataset="EUROC",
        seq_ids=EUROC_PRIMARY,
        groups=EUROC_ROOM,
        packs=euroc_packs,
        context="WITHIN_EUROC",
        alphas=alphas,
        tie=tie,
    )
    seq_rows.extend(a)
    sel_rows.extend(b)
    a, b = _run_loro(
        dataset="UZH_FPV",
        seq_ids=UZH_PRIMARY,
        groups=UZH_GROUP,
        packs=uzh_packs,
        context="WITHIN_UZH",
        alphas=alphas,
        tie=tie,
    )
    seq_rows.extend(a)
    sel_rows.extend(b)

    frozen: dict[str, dict[str, Any]] = {}
    for dataset, seq_ids, groups, packs, tag, sel_ctx in (
        ("EUROC", list(EUROC_PRIMARY), EUROC_ROOM, euroc_packs, "EUROC", "SOURCE_SELECTION_EUROC"),
        ("UZH_FPV", list(UZH_PRIMARY), UZH_GROUP, uzh_packs, "UZH", "SOURCE_SELECTION_UZH"),
    ):
        print(f"[lag100] source-only alpha freeze on {dataset}", flush=True)
        chosen = _select_alpha(
            seq_ids,
            packs,
            groups,
            dataset=dataset,
            alphas=alphas,
            tie=tie,
            context=sel_ctx,
        )
        frozen[tag] = chosen
        sel_rows.append(
            {
                "evaluation_context": sel_ctx,
                "phase": "SOURCE_SELECTION_FOR_TRANSFER",
                "fold_id": f"{tag}_SOURCE_ALL",
                "dataset": dataset,
                "model": MODEL_ID,
                "lag": FIXED_LAG,
                "selected_alpha": chosen["alpha"],
                "selection_score_rmse": chosen["score"],
                "n_tied": chosen["n_tied"],
                "train_sequences": ",".join(seq_ids),
                "held_out_sequence": "",
                "candidate_alphas": ",".join(str(a) for a in alphas),
                "notes": "source-only group-aware alpha selection; lag fixed at 100; no target data",
            }
        )

    for src_tag, src_ds, src_ids, src_packs, tgt_ds, tgt_ids, tgt_packs, ctx in (
        ("EUROC", "EUROC", list(EUROC_PRIMARY), euroc_packs, "UZH_FPV", list(UZH_PRIMARY), uzh_packs, "EUROC_TO_UZH"),
        ("UZH", "UZH_FPV", list(UZH_PRIMARY), uzh_packs, "EUROC", list(EUROC_PRIMARY), euroc_packs, "UZH_TO_EUROC"),
    ):
        print(f"[lag100] zero-shot {ctx} (frozen source Ridge, lag=100)", flush=True)
        ch = frozen[src_tag]
        fitted = fit_ridge(
            [src_packs[s] for s in src_ids],
            lag=FIXED_LAG,
            alpha=float(ch["alpha"]),
            use_qw=True,
            dataset=src_ds,
            model_name=MODEL_ID,
        )
        if tgt_ds == "UZH_FPV" and any(s.startswith("indoor_") or s.startswith("outdoor_") for s in fitted.train_sequences):
            raise RuntimeError("UZH target leaked into EuRoC->UZH source fit")
        if tgt_ds == "EUROC" and any(s.startswith("V1_") or s.startswith("V2_") for s in fitted.train_sequences):
            raise RuntimeError("EuRoC target leaked into UZH->EuRoC source fit")
        for sid in tgt_ids:
            seq_rows.append(
                _eval_seq(
                    tgt_packs[sid],
                    fitted,
                    context=ctx,
                    fold_id=f"TRANSFER_{ctx}",
                    alpha=float(ch["alpha"]),
                    notes="zero-shot; source-frozen lag=100 Ridge B3; target unused for fit or alpha",
                )
            )

    summary = _summarize(seq_rows, n_reps, seed)
    primary = _primary_summary(root, n_reps, seed)

    out = root / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "RIDGE_B3_FIXED_LAG100_SEQUENCE_RESULTS.csv", SEQ_COLS, seq_rows)
    write_csv(out / "RIDGE_B3_FIXED_LAG100_SUMMARY.csv", SUM_COLS, summary)
    write_csv(out / "RIDGE_B3_FIXED_LAG100_SOURCE_SELECTION.csv", SEL_COLS, sel_rows)
    (out / "RIDGE_B3_FIXED_LAG100_COMPARISON.md").write_text(
        _comparison_md(summary, primary, sel_rows),
        encoding="utf-8",
    )
    assert_primary_unchanged(root, before)
    print("[lag100] wrote results/sensitivity/; primary files unchanged", flush=True)
    print("NO NEURAL MODEL WAS RETRAINED.", flush=True)
    print("TARGET DATA WERE NOT USED FOR TUNING.", flush=True)
    print("PRIMARY MANUSCRIPT RESULTS WERE NOT CHANGED.", flush=True)
    return {"summary": summary, "primary": primary, "n_sequence_rows": len(seq_rows), "out": out}
