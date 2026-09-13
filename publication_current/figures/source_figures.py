"""Manuscript-facing figures for publication_current. No model fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

RIDGE = "Source-selected B3 Ridge"
MODEL_ORDER = ["Persistence", RIDGE, "TCN", "GRU", "Transformer"]
SHORT = {
    "Persistence": "Pers.",
    RIDGE: "Ridge (B3)",
    "TCN": "TCN",
    "GRU": "GRU",
    "Transformer": "Trans.",
}
MARKERS = {
    "Persistence": "o",
    RIDGE: "s",
    "TCN": "^",
    "GRU": "D",
    "Transformer": "v",
}
COLORS = {
    "Persistence": "#4d4d4d",
    RIDGE: "#000000",
    "TCN": "#2c7bb6",
    "GRU": "#31a354",
    "Transformer": "#756bb1",
}
CONTEXTS = ["WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"]
CTX_TITLE = {
    "WITHIN_EUROC": "Within EuRoC (n=6)",
    "WITHIN_UZH": "Within UZH-FPV (n=8)",
    "EUROC_TO_UZH": r"EuRoC $\rightarrow$ UZH (n=8)",
    "UZH_TO_EUROC": r"UZH $\rightarrow$ EuRoC (n=6)",
}


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1_schematic(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.0, edgecolor="black", facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    box(0.2, 3.2, 2.2, 1.1, "D1 EuRoC MAV\nFirefly / ADIS16448\n6 Vicon recordings\nnative ~200 Hz")
    box(0.2, 0.3, 2.2, 1.1, "D2 UZH-FPV\nSnapdragon IMU\n8 recordings\nnative ~500 Hz")
    box(2.8, 1.7, 2.4, 1.4, "Invariant scalars\n$q_f=\\|f\\|_2$ (m/s$^2$)\n$q_w=\\|\\omega\\|_2$ (rad/s)\nno fitted axis alignment")
    box(5.5, 1.7, 2.4, 1.4, "Causal 100 Hz streams\nelliptic anti-alias\n1.0 s lookback\n0.20 s / 20-step head")
    box(8.3, 3.15, 2.6, 1.15, "Source-only fit\nscaler + model selection\nno target adaptation")
    box(8.3, 1.7, 2.6, 1.15, "Within-domain LORO\nsequence = unit")
    box(8.3, 0.25, 2.6, 1.15, "Zero-shot transfer\nEuRoC$\\leftrightarrow$UZH\nfrozen source objects")
    ax.add_patch(FancyArrowPatch((2.4, 3.6), (2.8, 2.6), arrowstyle="-|>", mutation_scale=10, color="black", lw=0.9))
    ax.add_patch(FancyArrowPatch((2.4, 0.9), (2.8, 2.1), arrowstyle="-|>", mutation_scale=10, color="black", lw=0.9))
    ax.add_patch(FancyArrowPatch((5.2, 2.4), (5.5, 2.4), arrowstyle="-|>", mutation_scale=10, color="black", lw=0.9))
    ax.add_patch(FancyArrowPatch((7.9, 2.4), (8.3, 3.6), arrowstyle="-|>", mutation_scale=10, color="black", lw=0.9))
    ax.add_patch(FancyArrowPatch((7.9, 2.4), (8.3, 2.25), arrowstyle="-|>", mutation_scale=10, color="black", lw=0.9))
    ax.add_patch(FancyArrowPatch((7.9, 2.4), (8.3, 0.85), arrowstyle="-|>", mutation_scale=10, color="black", lw=0.9))
    ax.set_title("Locked measurement and evaluation protocol", fontsize=11, pad=6)
    _save(fig, path)


def figure2_primary_rmse(summaries: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharey=False)
    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        means, lo, hi = [], [], []
        for model in MODEL_ORDER:
            hit = [r for r in summaries if r["context"] == ctx and r["model"] == model][0]
            means.append(hit["mean_seq_rmse"])
            lo.append(hit["ci_low"])
            hi.append(hit["ci_high"])
        yerr = np.vstack([np.asarray(means) - np.asarray(lo), np.asarray(hi) - np.asarray(means)])
        for i, model in enumerate(MODEL_ORDER):
            ax.errorbar(
                x[i],
                means[i],
                yerr=np.array([[yerr[0, i]], [yerr[1, i]]]),
                fmt=MARKERS[model],
                color=COLORS[model],
                ecolor="black",
                elinewidth=0.8,
                capsize=3,
                markersize=7,
                label=SHORT[model] if ctx == "WITHIN_EUROC" else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], fontsize=8)
        ax.set_title(CTX_TITLE[ctx], fontsize=9)
        ax.set_ylabel(r"Equal-sequence RMSE (m/s$^2$)")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def figure3_transfer_gap(gap_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    models = [RIDGE, "TCN", "GRU", "Transformer"]
    directions = ["EUROC_TO_UZH", "UZH_TO_EUROC"]
    dir_lab = {"EUROC_TO_UZH": r"EuRoC$\rightarrow$UZH", "UZH_TO_EUROC": r"UZH$\rightarrow$EuRoC"}
    x = np.arange(len(models), dtype=np.float64)
    width = 0.36
    hatches = ["", "//"]
    for i, d in enumerate(directions):
        vals = []
        for m in models:
            hit = [r for r in gap_rows if r["direction"] == d and r["model"] == m][0]
            vals.append(float(hit["transfer_gap"]))
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=dir_lab[d],
            color=["#d9d9d9", "#6baed6"][i],
            edgecolor="black",
            linewidth=0.6,
            hatch=hatches[i],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Ridge (B3)", "TCN", "GRU", "Transformer"])
    ax.set_ylabel(r"Transfer gap (m/s$^2$)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def figure4_horizon(horizon_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True)
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        for model in MODEL_ORDER:
            pts = [r for r in horizon_rows if r["context"] == ctx and r["model"] == model]
            pts = sorted(pts, key=lambda r: int(r["horizon_step"]))
            assert len(pts) == 20, f"{ctx} {model} has {len(pts)} horizons, expected 20"
            h = [int(r["horizon_ms"]) for r in pts]
            y = [float(r["mean_sequence_rmse"]) for r in pts]
            ax.plot(h, y, color=COLORS[model], marker=MARKERS[model], ms=3.5, lw=1.3, label=SHORT[model] if ctx == "WITHIN_EUROC" else None)
        ax.set_title(CTX_TITLE[ctx], fontsize=9)
        ax.set_xlabel("Horizon (ms)")
        ax.set_ylabel(r"Equal-sequence RMSE (m/s$^2$)")
        ax.set_xlim(10, 200)
        ax.set_xticks([10, 50, 100, 150, 200])
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle=":", linewidth=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def figure5_advanced_vs_ridge(
    paired_seq_rows: list[dict[str, Any]],
    paired_summaries: list[dict[str, Any]],
    path: Path,
) -> None:
    """Per-flight paired Δ plus equal-sequence mean and 95% sequence-bootstrap CI."""
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    adv = ["TCN", "GRU", "Transformer"]
    for ax, ctx in zip(axes.ravel(), CONTEXTS, strict=True):
        sub = [r for r in paired_seq_rows if r["context"] == ctx]
        seqs: list[str] = []
        for r in sub:
            if r["sequence_id"] not in seqs:
                seqs.append(r["sequence_id"])
        x = np.arange(len(seqs), dtype=np.float64)
        mean_x = float(len(seqs)) + 0.85
        for i, model in enumerate(adv):
            vals = []
            for s in seqs:
                hit = [r for r in sub if r["sequence_id"] == s and r["model"] == model]
                vals.append(float(hit[0]["delta"]) if hit else np.nan)
            ax.scatter(
                x + (i - 1) * 0.12,
                vals,
                marker=MARKERS[model],
                color=COLORS[model],
                edgecolors="black",
                linewidths=0.4,
                s=28,
                zorder=3,
                label=model if ctx == "WITHIN_EUROC" else None,
            )
            rec = [p for p in paired_summaries if p["context"] == ctx and p["model"] == model][0]
            mean = float(rec["mean_paired_delta"])
            lo = float(rec["ci_low"])
            hi = float(rec["ci_high"])
            ax.errorbar(
                mean_x + (i - 1) * 0.18,
                mean,
                yerr=np.array([[mean - lo], [hi - mean]]),
                fmt=MARKERS[model],
                color=COLORS[model],
                ecolor="black",
                elinewidth=0.9,
                capsize=3.5,
                markersize=8,
                zorder=4,
            )
        ax.axhline(0.0, color="black", lw=0.7)
        ax.axvline(len(seqs) - 0.35, color="#bbbbbb", lw=0.6, linestyle="--")
        xticks = list(x) + [mean_x]
        labels = [s.replace("_snapdragon", "") for s in seqs] + ["mean"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(CTX_TITLE[ctx], fontsize=9)
        ax.set_ylabel(r"$\Delta$ RMSE (m/s$^2$); negative = lower than B3 Ridge")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)
