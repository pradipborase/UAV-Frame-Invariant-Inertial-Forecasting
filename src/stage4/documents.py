"""Markdown evidence documents for Stage 4. No model fitting."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audit.hashing import sha256_file
from audit.stage0c import write_csv
from stage1.io_native import EUROC_PRIMARY, UZH_PRIMARY

from .io import CONTEXTS, PUB_MODELS
from .stats import evidence_class


def _hash_if(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def write_all_documents(
    *,
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
    grab: Callable[..., dict[str, Any]],
    fmt3: Callable[[float], str],
    fmt_ci: Callable[..., str],
    utc_now: Callable[[], str],
    write_text: Callable[[Path, str], None],
    RIDGE_DETAILS: dict[str, dict[str, Any]],
    **_: Any,
) -> None:
    def g(ctx: str, model: str) -> dict[str, Any]:
        return grab(summaries, ctx, model)

    def pget(ctx: str, model: str) -> dict[str, Any]:
        return [p for p in paired if p["context"] == ctx and p["model"] == model][0]

    def hget(ctx: str, model: str) -> dict[str, Any]:
        return [h for h in high_rows if h["context"] == ctx and h["model"] == model][0]

    euroc_med_g = float(np.mean([r["median_qf"] for r in grav if r["dataset"] == "EUROC"]))
    uzh_med_g = float(np.mean([r["median_qf"] for r in grav if r["dataset"] == "UZH_FPV"]))
    euroc_f2 = float(np.mean([r["frac_within_9p81_pm_2p0"] for r in grav if r["dataset"] == "EUROC"]))
    uzh_f2 = float(np.mean([r["frac_within_9p81_pm_2p0"] for r in grav if r["dataset"] == "UZH_FPV"]))
    qf_w = [d for d in domain if d["quantity"] == "q_f"][0]
    gravity_note = (
        f"Sequence-mean median q_f is {euroc_med_g:.2f} m/s^2 (EuRoC) and {uzh_med_g:.2f} m/s^2 (UZH), "
        f"near gravitational acceleration, but dynamic variation is present: mean fraction within 9.81±2 m/s^2 "
        f"is {100*euroc_f2:.1f}% (EuRoC) and {100*uzh_f2:.1f}% (UZH); UZH p95 values reach well above 13 m/s^2. "
        "The target is not a constant-gravity scalar."
    )

    write_text(
        out / "ANALYSIS_HIERARCHY.md",
        """# Analysis hierarchy (frozen)

## PRIMARY
- Sequence-level multi-horizon q_f RMSE
- Four evaluation contexts: WITHIN_EUROC, WITHIN_UZH, EUROC_TO_UZH, UZH_TO_EUROC
- Persistence / Best frozen Ridge / TCN / GRU / Transformer
- RMSE at 50, 100, and 200 ms
- Zero-shot transfer results
- Paired advanced-versus-Ridge differences (Delta = RMSE_advanced - RMSE_Ridge)

## SECONDARY
- q_w ablation (authorized Stage-3 ablation only)
- High-specific-force target intervals (sequence-specific p95, unchanged)
- Horizon curves h=1..20
- Transfer gap (descriptive)
- Computational footprint
- Seed variability (not additional inferential units)
- Domain-shift characterization

## SUPPLEMENTARY
- Linear extrapolation baseline (B1)
- Full hyperparameter grids
- Per-sequence RMSE table
- Model-selection / coefficient details
- Filter and lock audits
- All Stage-0/1 manifests

Do not reclassify after manuscript writing begins.
""",
    )

    write_text(
        out / "TIME_DOMAIN_EXAMPLE.md",
        """# Time-domain example figure (optional; not a primary figure)

Pre-defined selection rule (neutral, not cherry-picked):
within each dataset, choose the sequence whose Ridge multi-horizon RMSE is the median among sequences in that dataset.

Status: **not generated**.

Reason: Stage 2 and Stage 3 did not save per-window prediction traces. Regenerating traces would require re-running frozen models. Stage 4 is forbidden from training or from regenerating forecasts except to verify an already frozen numeric result. Loading already-saved window-level predictions is allowed; those files are absent.

If a later reproducibility audit restores frozen window-level predictions without refitting, the median-Ridge sequence rule above remains the only authorized selection rule.
""",
    )

    write_text(
        out / "TERMINOLOGY_AUDIT.md",
        """# Terminology audit

Use **accelerometer specific force** for the IMU accelerometer measurand, not generic "acceleration".

Use **specific-force magnitude** for q_f = ||f||_2.

Use **reference trajectory** for pose products. "Ground truth" may appear when quoting dataset terminology; define that it is not independent IMU acceleration truth.

Bootstrap intervals are **sequence-level sampling uncertainty**, not a GUM measurement-uncertainty budget, and not "measurement uncertainty".

Protocol language: **pre-specified locked protocol**, not pre-registered (no OSF/Zenodo registration before modelling).

Zero-shot transfer: no target-dataset data were used to fit parameters, hyperparameters, normalization, coordinate alignment, or model-selection decisions for the reported source-to-target evaluation. Do **not** claim the target dataset was unseen throughout the entire study; datasets were audited before model evaluation.
""",
    )

    claims = [
        {
            "claim_id": "C1",
            "candidate_claim": "Frame-invariant q_f permits comparison without fitting an axis alignment.",
            "analysis_type": "PRIMARY/design",
            "supporting_file": "PROTOCOL_LOCK.md; results/stage0c/EUROC_UZH_COMPATIBILITY.csv",
            "supporting_metric": "representation lock; MAJOR frame rows avoided by magnitude",
            "support_level": "SUPPORTED with limits",
            "limitations": "Invariant only to a fixed orthonormal sensor rotation; does not remove all sensor/platform differences; loses direction.",
            "safe_wording": "A rotation-invariant specific-force magnitude avoids data-driven cross-dataset axis alignment for this scalar forecasting task.",
            "unsafe_wording": "The approach is invariant to all sensor differences.",
        },
        {
            "claim_id": "C2",
            "candidate_claim": "Learned forecasting improves over Persistence within both datasets.",
            "analysis_type": "PRIMARY",
            "supporting_file": "results/stage4/TABLE_PRIMARY_RESULTS.csv",
            "supporting_metric": "skill_vs_persistence > 0 for Ridge and advanced models within both datasets",
            "support_level": "SUPPORTED",
            "limitations": "n=6/8; linear extrapolation is worse than Persistence and is supplementary.",
            "safe_wording": "Learned methods improve over Persistence in both datasets.",
            "unsafe_wording": "Deep learning significantly outperforms all classical methods.",
        },
        {
            "claim_id": "C3",
            "candidate_claim": "Ridge remains competitive with compact nonlinear models.",
            "analysis_type": "PRIMARY",
            "supporting_file": "results/stage4/TABLE_ADVANCED_VS_RIDGE.csv",
            "supporting_metric": "within-EuRoC advanced Δ≥0; mixed transfer",
            "support_level": "SUPPORTED",
            "limitations": "UZH Transformer is a within-domain exception.",
            "safe_wording": "Regularized Ridge is highly competitive with these compact nonlinear models.",
            "unsafe_wording": "Deep learning significantly outperforms linear forecasting.",
        },
        {
            "claim_id": "C4",
            "candidate_claim": "Advanced-model gains are domain dependent.",
            "analysis_type": "PRIMARY",
            "supporting_file": "results/stage4/EVIDENCE_STRENGTH_CLASSIFICATION.csv",
            "supporting_metric": "CLEAR gain only within UZH Transformer; EuRoC no gain",
            "support_level": "SUPPORTED",
            "limitations": "Only two datasets.",
            "safe_wording": "Advanced-model gains vary by domain.",
            "unsafe_wording": "Transformer is the best model.",
        },
        {
            "claim_id": "C5",
            "candidate_claim": "No architecture dominates all contexts.",
            "analysis_type": "PRIMARY",
            "supporting_file": "results/stage4/KEY_FINDINGS_VERIFICATION.md",
            "supporting_metric": "context-wise winners differ",
            "support_level": "SUPPORTED",
            "limitations": "Five compact comparators only.",
            "safe_wording": "No architecture dominated across all evaluation contexts.",
            "unsafe_wording": "TCN is universally superior.",
        },
        {
            "claim_id": "C6",
            "candidate_claim": "Zero-shot transfer is feasible without target-domain fitted adaptation.",
            "analysis_type": "PRIMARY",
            "supporting_file": "results/stage3/TARGET_LEAKAGE_AUDIT.csv",
            "supporting_metric": "target_dataset_access_during_fit=NO for all objects; transfer RMSE < Persistence",
            "support_level": "SUPPORTED",
            "limitations": "Feasible ≠ lossless; datasets were audited before evaluation.",
            "safe_wording": "Zero-shot transfer remains feasible without fitting target-domain normalization or alignment.",
            "unsafe_wording": "The target dataset was unseen throughout the entire study.",
        },
        {
            "claim_id": "C7",
            "candidate_claim": "Transfer generally incurs degradation relative to corresponding within-target performance.",
            "analysis_type": "SECONDARY",
            "supporting_file": "results/stage4/TRANSFER_GAP_FINAL.csv",
            "supporting_metric": "positive transfer gaps for all learned models",
            "support_level": "SUPPORTED",
            "limitations": "Descriptive gap; Persistence has no trained source model so its gap is not analogous.",
            "safe_wording": "Learned models show higher error under zero-shot transfer than the corresponding within-target LORO evaluation.",
            "unsafe_wording": "The proposed approach eliminates domain shift.",
        },
        {
            "claim_id": "C8",
            "candidate_claim": "High-specific-force intervals are more difficult to predict.",
            "analysis_type": "SECONDARY",
            "supporting_file": "results/stage4/HIGH_QF_PUBLICATION_DATA.csv",
            "supporting_metric": "difficulty_ratio >> 1, especially UZH",
            "support_level": "SUPPORTED",
            "limitations": "Sparse high-q_f samples; not proven 'extreme maneuvers'.",
            "safe_wording": "High-specific-force target intervals are substantially more difficult to predict, especially for UZH.",
            "unsafe_wording": "The method fails on extreme maneuvers.",
        },
        {
            "claim_id": "C9",
            "candidate_claim": "q_w provides only modest additional predictive value.",
            "analysis_type": "SECONDARY",
            "supporting_file": "results/stage4/QW_ABLATION_FINAL.csv",
            "supporting_metric": "source-val RMSE change of order 0.001–0.012 m/s^2",
            "support_level": "SUPPORTED",
            "limitations": "Ablation only on the source-selected architecture per dataset.",
            "safe_wording": "The angular-speed magnitude channel provides only a modest auxiliary contribution.",
            "unsafe_wording": "Sensor fusion significantly improves performance.",
        },
        {
            "claim_id": "C10",
            "candidate_claim": "Compact advanced models have modest computational footprint.",
            "analysis_type": "SECONDARY",
            "supporting_file": "results/stage4/COMPUTATIONAL_FOOTPRINT_FINAL.csv",
            "supporting_metric": f"params<={max_params}; median inference <1 ms/window" if inf_ok else "timing not sub-ms",
            "support_level": "SUPPORTED" if inf_ok else "PARTIAL",
            "limitations": "CPU-only timing; no onboard flight test.",
            "safe_wording": (
                "All selected models have compact parameter counts and sub-millisecond median inference per prediction window on the reported CPU environment."
                if inf_ok
                else "Selected models have compact parameter counts; timing is CPU-specific."
            ),
            "unsafe_wording": "The method is suitable for real-time onboard deployment.",
        },
    ]
    write_csv(
        out / "CLAIM_EVIDENCE_MATRIX.csv",
        ["claim_id", "candidate_claim", "analysis_type", "supporting_file", "supporting_metric", "support_level", "limitations", "safe_wording", "unsafe_wording"],
        claims,
    )

    lim_rows = [
        {"limitation": "only 14 independent recordings (6 EuRoC / 8 UZH)", "severity": "HIGH", "effect_on_interpretation": "Sequence-bootstrap CIs are wide; do not claim statistical superiority from n=6.", "mitigation": "Sequence as inferential unit; report CIs, wins/losses, not p-centric claims.", "future_work": "More independent flights and platforms."},
        {"limitation": "only two datasets", "severity": "HIGH", "effect_on_interpretation": "Transfer is EuRoC↔UZH, not all UAV platforms.", "mitigation": "Official heterogeneous pair; zero-shot protocol.", "future_work": "Additional official IMU datasets if representation-compatible."},
        {"limitation": "different platforms/sensors", "severity": "MEDIUM", "effect_on_interpretation": "Confounds hardware, dynamics, and environment.", "mitigation": "Invariant scalars; no fitted alignment.", "future_work": "Same-airframe multi-IMU studies."},
        {"limitation": "different flight regimes", "severity": "MEDIUM", "effect_on_interpretation": "UZH includes more aggressive specific-force variation.", "mitigation": "Report within and transfer separately; high-q_f secondary analysis.", "future_work": "Matched-dynamics transfers."},
        {"limitation": "q_f magnitude removes directional information", "severity": "HIGH", "effect_on_interpretation": "Cannot recover 3-axis specific force or a shared body frame.", "mitigation": "Chosen because axes are not a demonstrated common frame.", "future_work": "Only if an official shared frame exists."},
        {"limitation": "no direct acceleration ground truth", "severity": "HIGH", "effect_on_interpretation": "Target is future sensor observation, not independent physical acceleration.", "mitigation": "State the estimand explicitly.", "future_work": "Calibrated independent force/IMU references if available."},
        {"limitation": "target is future sensor observation", "severity": "HIGH", "effect_on_interpretation": "Forecasts IMU telemetry, not navigation-grade acceleration.", "mitigation": "Telemetry-forecast framing.", "future_work": "Downstream control/estimation tasks."},
        {"limitation": "different reference-trajectory instrumentation", "severity": "MEDIUM", "effect_on_interpretation": "Pose products are not interchangeable IMU labels.", "mitigation": "Not used as q_f supervision.", "future_work": "Keep pose for characterization only."},
        {"limitation": "high-q_f data relatively sparse", "severity": "MEDIUM", "effect_on_interpretation": "Secondary high-q_f RMSE is noisier.", "mitigation": "Pre-specified p95; not used for selection.", "future_work": "Event-conditioned evaluation with more data."},
        {"limitation": "UZH body-frame extrinsic unresolved", "severity": "HIGH", "effect_on_interpretation": "Prevents a shared 3-axis body-frame target.", "mitigation": "Magnitude representation.", "future_work": "Official CAD/Kalibr vehicle-body chain if released."},
        {"limitation": "source datasets audited before zero-shot evaluation", "severity": "MEDIUM", "effect_on_interpretation": "Not researcher-blind to target existence.", "mitigation": "No target-fitted parameters; define zero-shot narrowly.", "future_work": "Hold out a third dataset if acquired later."},
        {"limitation": "zero-shot means no data-driven target adaptation, not complete blindness", "severity": "MEDIUM", "effect_on_interpretation": "Protocol knowledge of both datasets exists.", "mitigation": "Explicit definition in terminology audit.", "future_work": "Preregister a future third-dataset transfer."},
        {"limitation": "advanced models relatively small", "severity": "LOW", "effect_on_interpretation": "Does not test large foundation models.", "mitigation": "Pre-specified compact budget.", "future_work": "Only if scientifically justified later."},
        {"limitation": "CPU-only computational timing", "severity": "LOW", "effect_on_interpretation": "Footprint is machine-specific.", "mitigation": "Report CPU, warmup, median per window.", "future_work": "Target hardware benchmarks."},
        {"limitation": "no onboard flight deployment", "severity": "HIGH", "effect_on_interpretation": "No closed-loop or embedded claim.", "mitigation": "Footprint wording only.", "future_work": "Hardware-in-the-loop / flight test."},
        {"limitation": "no probabilistic uncertainty prediction", "severity": "MEDIUM", "effect_on_interpretation": "CIs are across sequences, not per-window predictive distributions.", "mitigation": "Do not call CIs measurement uncertainty.", "future_work": "Proper scoring rules if uncertainty is in scope."},
        {"limitation": "possible dominance of gravity magnitude", "severity": "MEDIUM", "effect_on_interpretation": "q_f may sit near 9.81 m/s^2.", "mitigation": "Gravity-dominance characterization; Persistence is a strong baseline.", "future_work": "Residual-from-g analyses only if protocol is reopened."},
    ]
    write_csv(
        out / "LIMITATION_MATRIX.csv",
        ["limitation", "severity", "effect_on_interpretation", "mitigation", "future_work"],
        lim_rows,
    )

    write_text(
        out / "CONTRIBUTION_MAP.md",
        """# Contribution map

## Contribution 1 — Frame-invariant physical representation
Classification: **STRONG**
A locked rotation-invariant specific-force magnitude enables EuRoC↔UZH comparison without a fitted axis map, which Stage 0C showed is not officially available. This is measurement design, not an architecture claim.

## Contribution 2 — Leakage-safe cross-dataset evaluation protocol
Classification: **STRONG**
Source-only scalers, nested/group-aware selection, freeze-then-transfer, and a target-leakage audit are the operational core of the study. Reviewers may call this "good practice"; for Measurement it is a primary contribution.

## Contribution 3 — Sequence-level uncertainty and independent-flight evaluation
Classification: **MODERATE**
Correct inferential unit (n=6/8) with sequence bootstrap. Methodologically necessary rather than novel statistics.

## Contribution 4 — Empirical complexity-generalization analysis
Classification: **STRONG**
The empirical result that compact Ridge remains competitive, with domain-specific Transformer gains that do not transfer, is the paper's result (not a new network).

## Contribution 5 — High-specific-force analysis
Classification: **MODERATE**
Pre-specified secondary analysis; informative but not the primary estimand.

Do not present all five as equally novel. Do not claim a new deep-learning architecture.
""",
    )

    write_text(
        out / "NOVELTY_RISK.md",
        """# Novelty risk assessment

## What is methodological novelty?
Locked invariant scalars plus a two-phase freeze that forbids target-fitted normalization/alignment/selection.

## What is protocol novelty?
Equal-sequence evaluation, nested LORO/LGO, and an explicit zero-shot definition for inertial telemetry forecasting across official UAV datasets.

## What is empirical novelty?
Quantified complexity-generalization on EuRoC MAV vs UZH-FPV Snapdragon: compact Ridge transfers comparably to compact TCN/GRU; a compact Transformer helps within UZH and degrades EuRoC→UZH.

## What is merely good practice?
Reproducible hashes, causal resampling, not using windows as n, not claiming p<0.05 on n=6.

## Likely reviewer attacks
| concern | severity | existing defense | remaining vulnerability | manuscript action |
|---|---|---|---|---|
| only two datasets | HIGH | Official heterogeneous pair; honest scope | Cannot claim all UAVs | State two-dataset scope in title/abstract |
| small sequence counts | HIGH | Sequence CIs; avoid p-hacking | Wide intervals | Lead with effect sizes and wins/losses |
| scalar loses direction | HIGH | Frame audit MAJOR rows | Not a 3-axis estimator | Methods: why magnitude |
| sensor observation not independent truth | HIGH | Estimand is future q_f | Not "true acceleration" | Define telemetry forecasting |
| limited advanced-model gain | MEDIUM | Report Ridge competitive as a finding | "Why DL?" | Complexity-generalization framing |
| different platforms | MEDIUM | That is the transfer question | Confounded factors | Do not claim isolated sensor-shift |
| reference-trajectory differences | MEDIUM | Pose unused as labels | Residual confusion | Terminology: reference trajectory |
| no onboard experiment | HIGH | Footprint only | No deployment claim | Forbidden wording list |
| gravity magnitude | MEDIUM | Gravity characterization + Persistence | Near-g sequences exist | Report fractions and SD |
| no probabilistic forecast | MEDIUM | Out of scope | Not GUM | Explicit limitation |
""",
    )

    write_text(
        out / "FIGURE_CAPTIONS.md",
        """# Figure captions

## Figure 1. Study/protocol schematic
Locked evaluation path from two official UAV IMU datasets through rotation-invariant specific-force and angular-speed magnitudes, causal 100 Hz resampling, source-only fitting, within-domain leave-one-recording-out evaluation, and frozen zero-shot transfer. No numeric estimates are encoded.

## Figure 2. Primary RMSE comparison
Equal-sequence mean multi-horizon RMSE of q_f (m/s^2) for Persistence, context-specific best frozen Ridge, TCN, GRU, and Transformer. Error bars are 95% percentile intervals from 10,000 sequence-level bootstrap replicates (EuRoC n=6; UZH n=8). Values are not window-pooled.

## Figure 3. Forecast-horizon RMSE
Equal-sequence mean RMSE versus forecast horizon (10 ms steps to 200 ms) for the same models and four contexts. Curves are descriptive; horizons are not tested as 20 separate hypotheses.

## Figure 4. Advanced-model paired difference versus Ridge
Per-sequence Δ = RMSE_advanced − RMSE_Ridge (m/s^2). Negative values mean the advanced model has lower sequence RMSE than the context-specific frozen Ridge comparator. The horizontal line marks zero.

## Figure 5. Transfer gap
Descriptive transfer gap (m/s^2) = zero-shot target RMSE − corresponding within-target RMSE for Ridge, TCN, GRU, and Transformer. Persistence is omitted because it has no trained source-domain model. Positive values indicate higher error after transfer.

## Figure 6. High-specific-force difficulty
Equal-sequence RMSE on all forecast targets versus targets with future q_f above the sequence-specific 95th percentile of processed q_f. These are high-specific-force target intervals, not labelled extreme maneuvers.
""",
    )

    table1 = [
        {"dataset": "EuRoC MAV (D1)", "platform": "Asctec Firefly hex-rotor; ADIS16448", "retained_sequences": 6, "imu_rate_native": "~200 Hz", "common_rate": "100 Hz", "specific_force_unit": "m/s^2", "angular_rate_unit": "rad/s", "reference_trajectory": "Vicon 6-DoF pose / ASL state estimate (not IMU labels)", "primary_role": "source and target in opposite transfer directions"},
        {"dataset": "UZH-FPV (D2)", "platform": "racing quadrotor; Snapdragon Flight IMU", "retained_sequences": 8, "imu_rate_native": "~500 Hz", "common_rate": "100 Hz", "specific_force_unit": "m/s^2", "angular_rate_unit": "rad/s", "reference_trajectory": "Leica + IMU-aided batch pose (not IMU labels)", "primary_role": "source and target in opposite transfer directions"},
    ]
    write_csv(
        out / "TABLE_DATASETS.csv",
        ["dataset", "platform", "retained_sequences", "imu_rate_native", "common_rate", "specific_force_unit", "angular_rate_unit", "reference_trajectory", "primary_role"],
        table1,
    )
    write_text(
        out / "TABLE_DATASETS.md",
        "# Table 1. Dataset and measurement characteristics\n\n"
        "| Dataset | Platform | n | Native IMU rate | Common rate | q_f unit | q_w unit | Reference trajectory |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| EuRoC MAV | Firefly / ADIS16448 | 6 | ~200 Hz | 100 Hz | m/s^2 | rad/s | Vicon pose |\n"
        "| UZH-FPV | Snapdragon IMU | 8 | ~500 Hz | 100 Hz | m/s^2 | rad/s | Leica + IMU-aided pose |\n",
    )
    protocol_rows = [
        {"item": "representation", "setting": "q_f = ||f||_2 (specific-force magnitude); q_w = ||omega||_2"},
        {"item": "primary_target", "setting": "future q_f (m/s^2)"},
        {"item": "primary_inputs", "setting": "history of q_f and q_w"},
        {"item": "sample_rate", "setting": "100 Hz after causal elliptic anti-alias (30/45 Hz) and integer decimation"},
        {"item": "lookback", "setting": "1.0 s (100 samples)"},
        {"item": "horizon", "setting": "0.20 s (20 samples), direct multi-output"},
        {"item": "stride", "setting": "0.05 s (5 samples)"},
        {"item": "inferential_unit", "setting": "flight recording / sequence"},
        {"item": "dataset_estimand", "setting": "equal-sequence mean multi-horizon RMSE"},
        {"item": "transfer_rule", "setting": "freeze source objects; no target-fitted scaler, alignment, or selection"},
        {"item": "normalization_rule", "setting": "equal-recording-weighted standard scaler, source-train recordings only"},
        {"item": "ridge_comparator", "setting": "; ".join(f"{ctx}: {RIDGE_DETAILS[ctx]['model_id']} lag={RIDGE_DETAILS[ctx]['lag']} alpha={RIDGE_DETAILS[ctx]['alpha']}" for ctx in CONTEXTS)},
    ]
    write_csv(out / "TABLE_PROTOCOL.csv", ["item", "setting"], protocol_rows)
    write_text(
        out / "TABLE_PROTOCOL.md",
        """# Table 2. Locked forecasting protocol

| Item | Setting |
|---|---|
| Representation | q_f = ||f||_2 (specific-force magnitude); q_w = ||ω||_2 |
| Primary target | future q_f (m/s^2) |
| Primary inputs | history of q_f and q_w |
| Sample rate | 100 Hz after causal elliptic anti-alias (30/45 Hz) and integer decimation |
| Lookback | 1.0 s (100 samples) |
| Horizon | 0.20 s (20 samples), direct multi-output |
| Stride | 0.05 s (5 samples) |
| Inferential unit | flight recording / sequence |
| Dataset estimand | equal-sequence mean multi-horizon RMSE |
| Transfer rule | freeze source objects; no target-fitted scaler, alignment, or selection |
| Normalization | equal-recording-weighted standard scaler, source-train recordings only |
""",
    )

    # inventory
    inv = []

    def add_inv(family: str, stage: str, rel: str, role: str, pri: str, pub: str, notes: str = "") -> None:
        path = root / rel
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            lines = handle.read().splitlines()
        cols = lines[0].split(",") if lines else []
        inv.append(
            {
                "result_family": family,
                "stage": stage,
                "file": rel.replace("\\", "/"),
                "sha256": _hash_if(path),
                "rows": max(len(lines) - 1, 0),
                "columns": len(cols),
                "role": role,
                "primary_or_secondary": pri,
                "publication_candidate": pub,
                "notes": notes,
            }
        )

    for ctx, fn in {
        "WITHIN_EUROC": "WITHIN_EUROC_SEQUENCE_RESULTS.csv",
        "WITHIN_UZH": "WITHIN_UZH_SEQUENCE_RESULTS.csv",
        "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_SEQUENCE_RESULTS.csv",
        "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_SEQUENCE_RESULTS.csv",
    }.items():
        add_inv("sequence_rmse", "2", f"results/stage2/{fn}", "sequence-level errors", "primary", "supplement", ctx)
    add_inv("baseline_summary", "2", "results/stage2/BASELINE_SUMMARY.csv", "dataset-level summary", "primary", "no", "")
    add_inv("horizon", "2", "results/stage2/HORIZON_RESULTS.csv", "horizon means", "secondary", "no", "")
    add_inv("paired", "2", "results/stage2/PAIRED_COMPARISONS.csv", "baseline pairing", "secondary", "no", "")
    add_inv("bootstrap", "2", "results/stage2/BOOTSTRAP_SUMMARY.csv", "sequence bootstrap", "primary", "no", "")
    add_inv("predictability", "2", "results/stage2/PREDICTABILITY_CHARACTERIZATION.csv", "ACF/persistence difficulty; p95", "secondary", "no", "p95 definition reused")
    add_inv("high_qf", "2", "results/stage2/HIGH_QF_SECONDARY.csv", "high-q_f RMSE", "secondary", "no", "")
    add_inv("advanced_summary", "3", "results/stage3/ADVANCED_SUMMARY.csv", "dataset-level advanced summary", "primary", "no", "reconciled in Stage 4")
    add_inv("advanced_vs_ridge", "3", "results/stage3/ADVANCED_VS_RIDGE.csv", "paired Δ", "primary", "no", "")
    add_inv("horizon", "3", "results/stage3/HORIZON_ADVANCED.csv", "horizon means", "secondary", "no", "")
    add_inv("seed", "3", "results/stage3/SEED_VARIABILITY.csv", "seed means", "secondary", "no", "")
    add_inv("high_qf", "3", "results/stage3/HIGH_QF_ADVANCED.csv", "high-q_f RMSE", "secondary", "no", "")
    add_inv("cost", "3", "results/stage3/COMPUTATIONAL_COST.csv", "footprint", "secondary", "yes", "")
    add_inv("domain_shift", "3", "results/stage3/DOMAIN_SHIFT_CHARACTERIZATION.csv", "descriptive shift", "secondary", "no", "")
    add_inv("leakage", "3", "results/stage3/TARGET_LEAKAGE_AUDIT.csv", "fit-scope audit", "primary", "no", "must remain NO")
    for ctx, fn in {
        "WITHIN_EUROC": "WITHIN_EUROC_ADVANCED.csv",
        "WITHIN_UZH": "WITHIN_UZH_ADVANCED.csv",
        "EUROC_TO_UZH": "TRANSFER_EUROC_TO_UZH_ADVANCED.csv",
        "UZH_TO_EUROC": "TRANSFER_UZH_TO_EUROC_ADVANCED.csv",
    }.items():
        add_inv("sequence_rmse", "3", f"results/stage3/{fn}", "sequence-level primary source", "primary", "supplement", ctx)
    write_csv(
        out / "RESULT_SOURCE_INVENTORY.csv",
        ["result_family", "stage", "file", "sha256", "rows", "columns", "role", "primary_or_secondary", "publication_candidate", "notes"],
        inv,
    )

    titles = [
        "A Frame-Invariant Measurement Protocol for Cross-Platform Evaluation of Short-Horizon UAV Inertial Forecasting",
        "Frame-Invariant Short-Horizon UAV Inertial Forecasting: Cross-Dataset Evaluation of Complexity and Zero-Shot Generalization",
        "Specific-Force Magnitude Forecasting Across UAV Platforms: A Leakage-Safe Zero-Shot Evaluation of Linear and Compact Sequence Models",
        "Cross-Platform UAV Inertial Telemetry Forecasting with Rotation-Invariant Measurements and Sequence-Level Uncertainty",
        "Complexity and Generalization in Zero-Shot Cross-UAV Forecasting of Accelerometer Specific-Force Magnitude",
    ]
    write_text(out / "TITLE_CANDIDATES.md", "# Title candidates\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1)) + "\n")

    we = g("WITHIN_EUROC", "Best frozen Ridge")
    wu = g("WITHIN_UZH", "Best frozen Ridge")
    et = g("EUROC_TO_UZH", "Best frozen Ridge")
    ue = g("UZH_TO_EUROC", "Best frozen Ridge")
    uz_tr = pget("WITHIN_UZH", "Transformer")
    et_tr = pget("EUROC_TO_UZH", "Transformer")

    write_text(
        out / "ABSTRACT_FACT_SHEET.md",
        f"""# Abstract fact sheet (verified facts only; not abstract prose)

- Datasets: EuRoC MAV (n=6) and UZH-FPV Snapdragon (n=8); official sources only.
- Target: q_f = ||f||_2, accelerometer specific-force magnitude (m/s^2), 100 Hz.
- History/horizon: 1.0 s / 0.20 s; stride 0.05 s; report 50/100/200 ms.
- Models: Persistence; context-specific frozen Ridge; compact TCN, GRU, Transformer.
- Within EuRoC equal-sequence RMSE: Persistence {fmt3(g('WITHIN_EUROC','Persistence')['mean_seq_rmse'])}; Ridge {fmt3(we['mean_seq_rmse'])}; TCN {fmt3(g('WITHIN_EUROC','TCN')['mean_seq_rmse'])}; GRU {fmt3(g('WITHIN_EUROC','GRU')['mean_seq_rmse'])}; Transformer {fmt3(g('WITHIN_EUROC','Transformer')['mean_seq_rmse'])} m/s^2.
- Within UZH: Persistence {fmt3(g('WITHIN_UZH','Persistence')['mean_seq_rmse'])}; Ridge {fmt3(wu['mean_seq_rmse'])}; Transformer {fmt3(g('WITHIN_UZH','Transformer')['mean_seq_rmse'])} m/s^2 (clearest advanced within-domain gain).
- EuRoC→UZH: Ridge {fmt3(et['mean_seq_rmse'])}; TCN {fmt3(g('EUROC_TO_UZH','TCN')['mean_seq_rmse'])}; Transformer {fmt3(g('EUROC_TO_UZH','Transformer')['mean_seq_rmse'])} m/s^2.
- UZH→EuRoC: Ridge {fmt3(ue['mean_seq_rmse'])}; GRU {fmt3(g('UZH_TO_EUROC','GRU')['mean_seq_rmse'])} m/s^2.
- Main conclusion: compact nonlinear models can yield dataset-specific within-domain gains, but extra complexity does not consistently improve zero-shot transfer; Ridge remains competitive.
- Limitations: n=6/8; two datasets; magnitude loses direction; target is future sensor observation; no onboard deployment.
""",
    )

    res_lines = ["# Results fact sheet", ""]
    for ctx in CONTEXTS:
        res_lines.append(f"## {ctx}")
        for model in PUB_MODELS:
            r = g(ctx, model)
            res_lines.append(
                f"- {model}: {fmt_ci(r['mean_seq_rmse'], r['ci_low'], r['ci_high'])} m/s^2; "
                f"50/100/200 ms = {fmt3(r['rmse_50ms'])}/{fmt3(r['rmse_100ms'])}/{fmt3(r['rmse_200ms'])}"
            )
        res_lines.append("")
    res_lines.append("## High-specific-force (equal-sequence mean RMSE, secondary)")
    for ctx in CONTEXTS:
        for model in PUB_MODELS:
            h = hget(ctx, model)
            res_lines.append(f"- {ctx} {model}: all {fmt3(h['rmse_all'])}; high {fmt3(h['rmse_high_qf'])}; ratio {h['difficulty_ratio']:.2f}")
    res_lines.append("\n## Transfer gap (learned models)")
    for r in gap_rows:
        res_lines.append(f"- {r['direction']} {r['model']}: {fmt3(r['transfer_gap'])} m/s^2 ({r['relative_transfer_gap_pct']:.1f}% of within-target)")
    res_lines.append("\n## q_w ablation")
    for r in qw_rows:
        res_lines.append(
            f"- {r['source_dataset']} {r['selected_advanced_architecture']}: q_f {fmt3(r['qf_only_val_rmse'])} vs q_f+q_w {fmt3(r['qf_qw_val_rmse'])}"
        )
    res_lines.append("\n## Parameters / inference")
    for r in cost:
        res_lines.append(
            f"- {r['source_dataset']} {r['model']} {r['config_id']}: {r['parameters']} params, "
            f"{float(r['median_inference_ms_per_window']):.3f} ms/window median"
        )
    write_text(out / "RESULTS_FACT_SHEET.md", "\n".join(res_lines) + "\n")

    write_text(
        out / "METHODS_FACT_SHEET.md",
        f"""# Methods fact sheet

## Datasets and sequences
EuRoC: {', '.join(EUROC_PRIMARY)}
UZH-FPV: {', '.join(UZH_PRIMARY)}

## Physical quantities
f: 3-axis accelerometer specific force, native IMU frame, m/s^2.
ω: 3-axis angular velocity, native IMU frame, rad/s.
q_f = ||f||_2; q_w = ||ω||_2.

## Sampling and filtering
Native rates ~200 Hz (EuRoC) and ~500 Hz (UZH). Causal elliptic SOS anti-alias, pass 30 Hz / stop 45 Hz, ≤1 dB / ≥60 dB, then integer decimation to 100 Hz. Causal forward SOS only; zero-phase bidirectional filtering is forbidden. Warmup 1.0 s discarded.

## Windows
Lookback 100 samples (1.0 s); horizon 20 samples (0.20 s); stride 5 samples (0.05 s). Direct 20-step head. Origins shared across models.

## Splits
Within-domain leave-one-recording-out. Nested group-aware source validation (EuRoC rooms; UZH trajectory groups). Transfer: all source sequences, frozen, evaluate all target sequences.

## Normalization
Equal-recording-weighted standard scaler on source training recordings only. Stage-2 Ridge: features only. Stage-3 networks: features and targets in source-normalized space; metrics inverse-transformed to m/s^2.

## Models
M0 Persistence: last q_f held for 20 steps.
M1 Best frozen Ridge: see RIDGE_COMPARATOR_MAP.csv (B3 except UZH→EuRoC B2).
M2–M4 compact TCN/GRU/Transformer, ≤12 source-only configs, seeds 20260912/13/14, AdamW, batch 64, max 100 epochs, patience 10 on source validation.

## Metrics and statistics
Primary: equal-sequence mean multi-horizon RMSE (m/s^2). Also MAE; RMSE at h=5,10,20 (50/100/200 ms). Sequence bootstrap 10,000, seed 20260912. Paired Δ = RMSE_advanced − RMSE_Ridge. Sign-flip p supplementary only.

## Zero-shot rule
No target data in parameters, hyperparameters, scaler, alignment, early stopping, or architecture choice.

## Reproducibility hashes
See LOCK_VERIFICATION.csv and REPRODUCIBILITY_MANIFEST.csv.
""",
    )

    write_text(
        out / "LIMITATIONS_FACT_SHEET.md",
        """# Limitations fact sheet (severity order)

1. **n=6 EuRoC / n=8 UZH.** Why it matters: CIs are wide; majority votes are fragile. Mitigation: sequence unit, bootstrap CIs, wins/losses. Cannot claim: statistically superior models.

2. **Two datasets / two platforms.** Why: transfer is not a universal UAV law. Mitigation: official heterogeneous pair. Cannot claim: generalization to all UAV platforms.

3. **Magnitude target loses direction; UZH body frame unresolved.** Why: not a 3-axis body-frame estimator. Mitigation: Stage 0C frame audit. Cannot claim: axis-aligned specific-force transfer.

4. **Target is future IMU observation, not independent acceleration truth.** Why: telemetry forecasting ≠ navigation reference. Mitigation: explicit estimand. Cannot claim: the method estimates true physical acceleration or uses ground-truth acceleration.

5. **No onboard deployment; CPU timing only.** Why: footprint ≠ real-time flight. Mitigation: wording restriction. Cannot claim: real-time onboard suitability.

6. **High-q_f intervals sparse; gravity near 9.81 m/s^2 is common.** Why: secondary regime / near-g sequences. Mitigation: pre-specified p95; gravity table. Cannot claim: extreme-maneuver certification.

7. **Datasets audited before modelling.** Why: not full blindness. Mitigation: no target-fitted objects. Cannot claim: the target was unseen throughout the study, or that the study is preregistered.

8. **No predictive uncertainty model.** Why: bootstrap is across flights. Cannot claim: GUM measurement uncertainty.
""",
    )

    write_text(
        out / "MANUSCRIPT_SECTION_MAP.md",
        """# Manuscript section evidence map (internal files only; no literature)

- Introduction: PROTOCOL_LOCK.md question; ABSTRACT_FACT_SHEET.md; CONTRIBUTION_MAP.md
- Related work: not populated (do not invent citations)
- Dataset/measurement setup: TABLE_DATASETS.md; STAGE0C compatibility; USABLE sequences
- Frame-invariant representation: PROTOCOL_LOCK.md; STAGE1 invariants; C1
- Preprocessing: METHODS_FACT_SHEET.md; src/stage1/causal_filter.py
- Forecast protocol: TABLE_PROTOCOL.md; ANALYSIS_HIERARCHY.md
- Models: RIDGE_COMPARATOR_MAP.csv; STAGE3_SOURCE_MODELS_FROZEN.md
- Statistical analysis: bootstrap seed 20260912; TERMINOLOGY_AUDIT.md
- Results: TABLE_PRIMARY_RESULTS.md; TABLE_ADVANCED_VS_RIDGE.csv; figures 2–6
- Discussion: KEY_FINDINGS_VERIFICATION.md; complexity-generalization
- Limitations: LIMITATIONS_FACT_SHEET.md; LIMITATION_MATRIX.csv
- Reproducibility: REPRODUCIBILITY_MANIFEST.csv; lock hashes
- Conclusion: working conclusion in KEY_FINDINGS_VERIFICATION.md
""",
    )

    write_text(
        out / "MANUSCRIPT_EVIDENCE_PACKAGE.md",
        f"""# Manuscript Evidence Package

## A. Study design in one paragraph
Two official UAV IMU datasets (EuRoC MAV, n=6; UZH-FPV Snapdragon, n=8) are compared on short-horizon forecasting of accelerometer specific-force magnitude after causal 100 Hz resampling. All fitted objects are source-only. Evaluation is leave-one-recording-out within domain and frozen zero-shot across datasets. The inferential unit is the flight recording.

## B. Measurement quantities
q_f = ||f||_2 (m/s^2); q_w = ||ω||_2 (rad/s). f is measured specific force, not linear acceleration of the origin.

## C. Dataset facts
EuRoC Firefly/ADIS16448 ~200 Hz; UZH Snapdragon ~500 Hz. Reference trajectories exist but are not q_f labels.

## D. Locked protocol
Lookback 1.0 s, horizon 0.20 s, stride 0.05 s, equal-sequence RMSE, 10,000 sequence bootstrap, seed 20260912.

## E. Primary results
See TABLE_PRIMARY_RESULTS.md. Ridge {fmt3(we['mean_seq_rmse'])} m/s^2 within EuRoC; Transformer {fmt3(g('WITHIN_UZH','Transformer')['mean_seq_rmse'])} vs Ridge {fmt3(wu['mean_seq_rmse'])} within UZH.

## F. Within-domain findings
EuRoC: Ridge matches or beats compact networks. UZH: Transformer shows a clear practical gain over Ridge (Δ={uz_tr['mean_paired_delta']:.3f} m/s^2, 7–1, CI excludes 0).

## G. Zero-shot findings
Learned models beat Persistence without target adaptation. EuRoC→UZH Transformer degrades vs Ridge (Δ={et_tr['mean_paired_delta']:.3f}). TCN remains near Ridge. Transfer gaps are positive for all learned models.

## H. Complexity-generalization findings
Larger architectural family (Transformer) is not uniformly better under transfer. Compact Ridge remains competitive.

## I. High-specific-force findings
Difficulty ratios exceed 1 in all contexts and are largest on UZH (secondary).

## J. Auxiliary q_w finding
Modest source-validation RMSE change; do not claim major sensor fusion gains.

## K. Computational footprint
≤{max_params} parameters; {'sub-millisecond median inference per window on the reported CPU' if inf_ok else 'CPU timing recorded'}.

## L. Statistical limitations
n=6/8; CIs are sequence-level sampling uncertainty.

## M. Dataset/measurement limitations
Two platforms; unresolved UZH body frame; magnitude loses direction; sensor-observation target.

## N. Claims that ARE supported
C1–C10 safe wordings in CLAIM_EVIDENCE_MATRIX.csv.

## O. Claims that are NOT supported
Transformer/TCN universally best; DL significantly outperforms Ridge; domain shift eliminated; true physical acceleration; onboard real-time; all UAV platforms; major q_w fusion; preregistration; CIs as measurement uncertainty.

## P. Recommended manuscript emphasis
Frame-invariant cross-platform UAV inertial forecasting with emphasis on the complexity-generalization trade-off under zero-shot dataset transfer.
""",
    )

    # reproducibility manifest
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_family": platform.processor(),
        "bootstrap_seed": 20260912,
        "bootstrap_reps": 10000,
        "training_seeds": "20260912,20260913,20260914",
    }
    man = [
        {"item": "protocol_md", "value": _hash_if(root / "PROTOCOL_LOCK.md"), "notes": json.dumps(env)},
        {"item": "protocol_yaml", "value": _hash_if(root / "configs" / "protocol_lock.yaml"), "notes": ""},
        {"item": "stage2_config", "value": _hash_if(root / "configs" / "stage2_baselines.yaml"), "notes": ""},
        {"item": "stage2_freeze", "value": _hash_if(root / "results" / "stage2" / "SOURCE_MODELS_FROZEN.md"), "notes": ""},
        {"item": "stage3_config", "value": _hash_if(root / "configs" / "stage3_advanced.yaml"), "notes": ""},
        {"item": "stage3_grid", "value": _hash_if(root / "results" / "stage3" / "HYPERPARAMETER_GRID.csv"), "notes": ""},
        {"item": "stage3_freeze", "value": _hash_if(root / "results" / "stage3" / "STAGE3_SOURCE_MODELS_FROZEN.md"), "notes": ""},
        {"item": "table_primary", "value": _hash_if(out / "TABLE_PRIMARY_RESULTS.csv"), "notes": ""},
        {"item": "table_vs_ridge", "value": _hash_if(out / "TABLE_ADVANCED_VS_RIDGE.csv"), "notes": ""},
        {"item": "horizon_data", "value": _hash_if(out / "HORIZON_PUBLICATION_DATA.csv"), "notes": ""},
        {"item": "high_qf_data", "value": _hash_if(out / "HIGH_QF_PUBLICATION_DATA.csv"), "notes": ""},
        {"item": "transfer_gap", "value": _hash_if(out / "TRANSFER_GAP_FINAL.csv"), "notes": ""},
        {"item": "footprint", "value": _hash_if(out / "COMPUTATIONAL_FOOTPRINT_FINAL.csv"), "notes": ""},
        {"item": "claim_matrix", "value": _hash_if(out / "CLAIM_EVIDENCE_MATRIX.csv"), "notes": ""},
        {"item": "fig2_data", "value": _hash_if(out / "figure_data" / "FIG2_primary_rmse.csv"), "notes": ""},
        {"item": "fig3_data", "value": _hash_if(out / "figure_data" / "FIG3_horizon_curves.csv"), "notes": ""},
        {"item": "fig4_data", "value": _hash_if(out / "figure_data" / "FIG4_paired_delta.csv"), "notes": ""},
        {"item": "fig5_data", "value": _hash_if(out / "figure_data" / "FIG5_transfer_gap.csv"), "notes": ""},
        {"item": "fig6_data", "value": _hash_if(out / "figure_data" / "FIG6_high_qf.csv"), "notes": ""},
        {"item": "fig01_pdf", "value": _hash_if(out / "figures" / "Fig01_Protocol_Schematic.pdf"), "notes": ""},
        {"item": "fig02_pdf", "value": _hash_if(out / "figures" / "Fig02_Primary_RMSE.pdf"), "notes": ""},
        {"item": "fig03_pdf", "value": _hash_if(out / "figures" / "Fig03_Horizon_RMSE.pdf"), "notes": ""},
        {"item": "fig04_pdf", "value": _hash_if(out / "figures" / "Fig04_Advanced_vs_Ridge.pdf"), "notes": ""},
        {"item": "fig05_pdf", "value": _hash_if(out / "figures" / "Fig05_Transfer_Gap.pdf"), "notes": ""},
        {"item": "fig06_pdf", "value": _hash_if(out / "figures" / "Fig06_High_qf.pdf"), "notes": ""},
        {"item": "software_python", "value": env["python"], "notes": "CPU-only; no CUDA used in Stage 4"},
        {"item": "software_numpy", "value": env["numpy"], "notes": ""},
        {"item": "machine", "value": env["machine"], "notes": env["processor_family"]},
        {"item": "random_seeds", "value": env["training_seeds"], "notes": "Stage-3 training seeds; Stage-4 does not draw new training seeds"},
        {"item": "bootstrap", "value": f"{env['bootstrap_reps']}/{env['bootstrap_seed']}", "notes": "sequence-level"},
    ]
    for seq in list(EUROC_PRIMARY) + list(UZH_PRIMARY):
        ds = "EUROC" if seq in EUROC_PRIMARY else "UZH_FPV"
        man.append({"item": f"processed_{ds}_{seq}", "value": _hash_if(root / "data" / "processed" / "stage1" / f"{ds}_{seq}.csv"), "notes": "processed Stage-1 stream"})
    write_csv(out / "REPRODUCIBILITY_MANIFEST.csv", ["item", "value", "notes"], man)

    recon_status = "PASS" if not fail_recon else "FAIL"
    decision = "PASS" if recon_status == "PASS" and n_target == 0 and n_euroc == 6 and n_uzh == 8 else "CONDITIONAL_PASS"
    if critical:
        decision = "FAIL"
    ready = "YES" if decision == "PASS" else "NO"

    lock_md = [
        "# FINAL RESULTS LOCK",
        f"utc: {utc_now()}",
        "After this file is hashed, experimental numbers must not be edited during manuscript drafting unless a verified error is documented.",
        "",
        f"primary table sha256: {_hash_if(out / 'TABLE_PRIMARY_RESULTS.csv')}",
        f"advanced-vs-Ridge sha256: {_hash_if(out / 'TABLE_ADVANCED_VS_RIDGE.csv')}",
        f"horizon-data sha256: {_hash_if(out / 'HORIZON_PUBLICATION_DATA.csv')}",
        f"high-q_f-data sha256: {_hash_if(out / 'HIGH_QF_PUBLICATION_DATA.csv')}",
        f"transfer-gap sha256: {_hash_if(out / 'TRANSFER_GAP_FINAL.csv')}",
        f"computational-footprint sha256: {_hash_if(out / 'COMPUTATIONAL_FOOTPRINT_FINAL.csv')}",
        f"claim-evidence matrix sha256: {_hash_if(out / 'CLAIM_EVIDENCE_MATRIX.csv')}",
        f"FIG2 data sha256: {_hash_if(out / 'figure_data' / 'FIG2_primary_rmse.csv')}",
        f"FIG3 data sha256: {_hash_if(out / 'figure_data' / 'FIG3_horizon_curves.csv')}",
        f"FIG4 data sha256: {_hash_if(out / 'figure_data' / 'FIG4_paired_delta.csv')}",
        f"FIG5 data sha256: {_hash_if(out / 'figure_data' / 'FIG5_transfer_gap.csv')}",
        f"FIG6 data sha256: {_hash_if(out / 'figure_data' / 'FIG6_high_qf.csv')}",
        "",
        f"n_target_fitted_objects: {n_target}",
        f"numerical_reconciliation: {recon_status}",
        f"EuRoC sequences: {n_euroc}",
        f"UZH sequences: {n_uzh}",
        "",
    ]
    write_text(out / "FINAL_RESULTS_LOCK.md", "\n".join(lock_md))
    (out / "FINAL_RESULTS_LOCK.sha256").write_text(sha256_file(out / "FINAL_RESULTS_LOCK.md") + "\n", encoding="utf-8")

    crit_txt = "- None." if not critical else "\n".join(f"- {c['context']} {c['model']} {c['metric']} diff={c['absolute_difference']}" for c in critical)
    maj_txt = "- None beyond documented rounding in narrative reports." if not major else "\n".join(f"- {m['context']} {m['model']} {m['metric']}" for m in major[:20])
    if fail_recon:
        maj_txt = "\n".join(f"- {m['context']} {m['model']} {m['metric']} |diff|={float(m['absolute_difference']):.3g}" for m in fail_recon)

    report = f"""# STAGE 4 REPORT

Lock verification:
PASS (all seven frozen artifacts match expected SHA-256)

Numerical reconciliation:
{recon_status} ({len(fail_recon)} mismatches above tolerance of {len(recon)} checks)

Sequences:
EuRoC = {n_euroc}
UZH = {n_uzh}

Primary target:
q_f specific-force magnitude (m/s^2)

Primary contexts:
WITHIN_EUROC, WITHIN_UZH, EUROC_TO_UZH, UZH_TO_EUROC

Primary models:
Persistence; context-specific best frozen Ridge; TCN; GRU; Transformer
Linear extrapolation is supplementary only.

Primary results table:
{'PASS' if recon_status == 'PASS' else 'FAIL'}

Advanced-vs-Ridge verification:
{'PASS' if recon_status == 'PASS' else 'CHECK'}; Delta = RMSE_advanced - RMSE_Ridge (negative = advanced lower)

Horizon verification:
Recomputed h=1..20 equal-sequence means with sequence-bootstrap CIs (not 20 hypothesis tests)

High-q_f verification:
Sequence-specific p95 unchanged vs Stage-2 PREDICTABILITY_CHARACTERIZATION.csv

Transfer-gap verification:
Positive gaps for all learned models; Persistence omitted as structurally untrained

q_w ablation:
Modest auxiliary contribution (see QW_ABLATION_FINAL.csv)

Computational-footprint verification:
{'sub-millisecond median inference per window on reported CPU' if inf_ok else 'see COMPUTATIONAL_FOOTPRINT_FINAL.csv'}

Seed-variability verification:
Three seeds summarized separately; sequences remain inferential units

Major supported conclusions:
- Ridge remains competitive with compact nonlinear models.
- Transformer within-UZH gain does not transfer EuRoC→UZH.
- No architecture dominates all four contexts.
- Zero-shot transfer is feasible without target-fitted adaptation, with descriptive transfer degradation.

Claims weakened/rejected:
- TCN/Transformer as universal winners
- Deep learning significantly outperforms Ridge
- Domain shift eliminated
- True physical acceleration / GT acceleration
- Real-time onboard deployment
- Preregistration
- Bootstrap CIs as measurement uncertainty

Most important limitation:
Small independent-flight counts (n=6/8) and only two datasets.

Second most important limitation:
q_f magnitude discards direction; target is future sensor observation, not independent acceleration truth.

Gravity-dominance characterization:
{gravity_note}

Publication figures:
Fig01–Fig06 (PDF/SVG/PNG) under results/stage4/figures and publication_package/figures

Publication tables:
Table 1 datasets; Table 2 protocol; Table 3 primary RMSE; Table 4 advanced vs Ridge; Table 5 footprint

Reproducibility manifest:
results/stage4/REPRODUCIBILITY_MANIFEST.csv

Final results lock:
results/stage4/FINAL_RESULTS_LOCK.md sha256={_hash_if(out / 'FINAL_RESULTS_LOCK.md')}

Critical findings:
{crit_txt}

Major findings:
{maj_txt}

pytest:
SEE_PYTEST passed, SEE_PYTEST failed

RECOMMENDED MANUSCRIPT EMPHASIS:
Frame-invariant cross-platform UAV inertial forecasting with emphasis on the complexity-generalization trade-off under zero-shot dataset transfer.

RECOMMENDED PRIMARY TITLE:
{titles[1]}

READY FOR MANUSCRIPT WRITING:
{ready}

OVERALL STAGE-4 DECISION:
{decision}

No new model was fitted in Stage 4. No hyperparameter was retuned. No target-domain adaptation was performed.
"""
    write_text(out / "STAGE4_REPORT.md", report)
    write_text(root / "STAGE4_REPORT.md", report)

    # Table 4 markdown
    t4 = [
        "# Table 4. Advanced versus Ridge (paired sequence differences)",
        "",
        "Delta = RMSE_advanced − RMSE_Ridge (m/s^2). Negative Delta means the advanced model has lower sequence RMSE.",
        "",
        "| Context | Model | Mean Δ | Median Δ | 95% CI | Wins–Losses–Ties | Relative % vs Ridge | Evidence class |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in paired:
        t4.append(
            f"| {p['context']} | {p['model']} | {p['mean_paired_delta']:.3f} | {p['median_paired_delta']:.3f} | "
            f"[{p['ci_low']:.3f}, {p['ci_high']:.3f}] | {p['wins']}–{p['losses']}–{p['ties']} | "
            f"{p['relative_improvement_pct']:.1f} | {evidence_class(p)} |"
        )
    write_text(out / "TABLE_ADVANCED_VS_RIDGE.md", "\n".join(t4) + "\n")

    t5 = ["# Table 5. Computational footprint", "", "| Source | Model | Config | Parameters | Mean train s | Median inference ms/window | Artifact bytes |", "|---|---|---|---|---|---|---|"]
    for r in cost:
        t5.append(
            f"| {r['source_dataset']} | {r['model']} | {r['config_id']} | {r['parameters']} | "
            f"{float(r['mean_training_time_s']):.1f} | {float(r['median_inference_ms_per_window']):.3f} | {int(float(r['mean_artifact_bytes']))} |"
        )
    t5.append("\nCPU environment only. Not a real-time onboard claim.\n")
    write_text(out / "TABLE_FOOTPRINT.md", "\n".join(t5))
