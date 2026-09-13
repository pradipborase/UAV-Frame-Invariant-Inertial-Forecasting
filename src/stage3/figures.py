"""Publication figures for Stage 3. Y-axes start at zero. CSV sources stored beside PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["B0_PERSISTENCE", "BEST_RIDGE", "TCN", "GRU", "TRANSFORMER"]
MODEL_LABEL = {
    "B0_PERSISTENCE": "Persistence",
    "BEST_RIDGE": "Ridge",
    "TCN": "TCN",
    "GRU": "GRU",
    "TRANSFORMER": "Transformer",
}
COLORS = {
    "B0_PERSISTENCE": "#4d4d4d",
    "BEST_RIDGE": "#d62728",
    "TCN": "#1f77b4",
    "GRU": "#2ca02c",
    "TRANSFORMER": "#9467bd",
}
CONTEXTS = ["WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"]
CTX_TITLE = {
    "WITHIN_EUROC": "Within EuRoC",
    "WITHIN_UZH": "Within UZH-FPV",
    "EUROC_TO_UZH": r"EuRoC $\rightarrow$ UZH",
    "UZH_TO_EUROC": r"UZH $\rightarrow$ EuRoC",
}


def _save(fig: plt.Figure, pdf: Path, png: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_a_rmse_ci(rows: list[dict[str, Any]], *, pdf: Path, png: Path, csv_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))
    csv_lines = ["context,model,mean_seq_rmse,ci_low,ci_high"]
    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        means, lo, hi = [], [], []
        for model in MODEL_ORDER:
            hit = [r for r in rows if r["context"] == ctx and r["model"] == model]
            if not hit:
                means.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            r = hit[0]
            means.append(float(r["mean_seq_rmse"]))
            lo.append(float(r["ci_low"]))
            hi.append(float(r["ci_high"]))
            csv_lines.append(f"{ctx},{model},{r['mean_seq_rmse']},{r['ci_low']},{r['ci_high']}")
        yerr = np.vstack([np.asarray(means) - np.asarray(lo), np.asarray(hi) - np.asarray(means)])
        colors = [COLORS[m] for m in MODEL_ORDER]
        ax.bar(x, means, color=colors, edgecolor="none", yerr=yerr, capsize=3, error_kw={"lw": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=25, ha="right", fontsize=8)
        ax.set_title(CTX_TITLE[ctx])
        ax.set_ylabel(r"Equal-sequence RMSE (m/s$^2$)")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def figure_b_horizon(rows: list[dict[str, Any]], *, pdf: Path, png: Path, csv_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    csv_lines = ["evaluation_context,model,horizon_index,horizon_ms,equal_sequence_mean_rmse"]
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        for model in MODEL_ORDER:
            pts = [r for r in rows if r["evaluation_context"] == ctx and r["model"] == model]
            pts = sorted(pts, key=lambda r: int(r["horizon_index"]))
            if not pts:
                continue
            h = [int(r["horizon_ms"]) for r in pts]
            y = [float(r["equal_sequence_mean_rmse"]) for r in pts]
            ax.plot(h, y, color=COLORS[model], label=MODEL_LABEL[model], lw=1.6)
            for r in pts:
                csv_lines.append(
                    f"{ctx},{model},{r['horizon_index']},{r['horizon_ms']},{r['equal_sequence_mean_rmse']}"
                )
        ax.set_title(CTX_TITLE[ctx])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Horizon (ms)")
        ax.set_ylabel(r"Equal-sequence RMSE (m/s$^2$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def figure_c_paired(rows: list[dict[str, Any]], *, pdf: Path, png: Path, csv_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    csv_lines = ["evaluation_context,sequence_id,model,delta_vs_ridge"]
    adv = ["TCN", "GRU", "TRANSFORMER"]
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        sub = [r for r in rows if r["evaluation_context"] == ctx]
        seqs: list[str] = []
        for r in sub:
            if r["sequence_id"] not in seqs:
                seqs.append(r["sequence_id"])
        x = np.arange(len(seqs), dtype=np.float64)
        width = 0.24
        for i, model in enumerate(adv):
            vals = []
            for s in seqs:
                hit = [r for r in sub if r["sequence_id"] == s and r["model"] == model]
                vals.append(float(hit[0]["delta_vs_ridge"]) if hit else np.nan)
                if hit:
                    csv_lines.append(f"{ctx},{s},{model},{hit[0]['delta_vs_ridge']}")
            ax.bar(x + (i - 1) * width, vals, width=width, label=MODEL_LABEL[model], color=COLORS[model], edgecolor="none")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_snapdragon", "") for s in seqs], rotation=40, ha="right", fontsize=7)
        ax.set_title(CTX_TITLE[ctx])
        ax.set_ylabel(r"RMSE$_{\mathrm{adv}}-$RMSE$_{\mathrm{Ridge}}$ (m/s$^2$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def figure_d_transfer_gap(rows: list[dict[str, Any]], *, pdf: Path, png: Path, csv_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    models = ["BEST_RIDGE", "TCN", "GRU", "TRANSFORMER"]
    directions = ["EUROC_TO_UZH", "UZH_TO_EUROC"]
    dir_lab = {"EUROC_TO_UZH": r"EuRoC$\rightarrow$UZH", "UZH_TO_EUROC": r"UZH$\rightarrow$EuRoC"}
    x = np.arange(len(models), dtype=np.float64)
    width = 0.36
    csv_lines = ["direction,model,transfer_gap"]
    for i, d in enumerate(directions):
        vals = []
        for m in models:
            hit = [r for r in rows if r["direction"] == d and r["model"] == m]
            vals.append(float(hit[0]["transfer_gap"]) if hit else np.nan)
            if hit:
                csv_lines.append(f"{d},{m},{hit[0]['transfer_gap']}")
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=dir_lab[d], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel(r"Transfer gap (m/s$^2$)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def figure_e_high_qf(rows: list[dict[str, Any]], *, pdf: Path, png: Path, csv_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))
    csv_lines = ["evaluation_context,model,mean_high_qf_rmse"]
    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        means = []
        for model in MODEL_ORDER:
            hit = [r for r in rows if r["evaluation_context"] == ctx and r["model"] == model]
            if not hit:
                means.append(np.nan)
                continue
            vals = [float(r["rmse_high_p95_targets"]) for r in hit if r["rmse_high_p95_targets"] != ""]
            means.append(float(np.mean(vals)) if vals else np.nan)
            if vals:
                csv_lines.append(f"{ctx},{model},{np.mean(vals)}")
        ax.bar(x, means, color=[COLORS[m] for m in MODEL_ORDER], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=25, ha="right", fontsize=8)
        ax.set_title(CTX_TITLE[ctx])
        ax.set_ylabel(r"High-$q_f$ RMSE (m/s$^2$)")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def figure_f_footprint(rows: list[dict[str, Any]], *, pdf: Path, png: Path, csv_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    csv_lines = ["model,parameters,mean_within_rmse"]
    for r in rows:
        ax.scatter(
            float(r["parameters"]),
            float(r["mean_within_rmse"]),
            color=COLORS.get(r["model"], "#333333"),
            s=55,
            label=MODEL_LABEL.get(r["model"], r["model"]),
        )
        csv_lines.append(f"{r['model']},{r['parameters']},{r['mean_within_rmse']}")
        ax.annotate(MODEL_LABEL.get(r["model"], r["model"]), (float(r["parameters"]), float(r["mean_within_rmse"])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(r"Mean within-domain RMSE (m/s$^2$)")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)
