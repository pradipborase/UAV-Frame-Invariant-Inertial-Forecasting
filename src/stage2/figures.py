"""Simple publication figures. Y-axes start at zero. Source CSVs stored beside PDFs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["B0_PERSISTENCE", "B1_LINEAR", "B2_RIDGE_QF", "B3_RIDGE_QF_QW"]
MODEL_LABEL = {
    "B0_PERSISTENCE": "Persistence",
    "B1_LINEAR": "Linear",
    "B2_RIDGE_QF": r"Ridge $q_f$",
    "B3_RIDGE_QF_QW": r"Ridge $q_f{+}q_w$",
}
COLORS = {
    "B0_PERSISTENCE": "#4d4d4d",
    "B1_LINEAR": "#1f77b4",
    "B2_RIDGE_QF": "#d62728",
    "B3_RIDGE_QF_QW": "#2ca02c",
}


def _save(fig: plt.Figure, pdf: Path, png: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def grouped_rmse_by_sequence(rows: list[dict], *, title: str, pdf: Path, png: Path, csv_path: Path) -> None:
    seqs = []
    for r in rows:
        if r["sequence_id"] not in seqs:
            seqs.append(r["sequence_id"])
    x = np.arange(len(seqs), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    csv_rows = ["sequence_id,model,rmse_all_horizons"]
    for i, model in enumerate(MODEL_ORDER):
        vals = []
        for s in seqs:
            hit = [r for r in rows if r["sequence_id"] == s and r["model"] == model]
            vals.append(float(hit[0]["rmse_all_horizons"]) if hit else np.nan)
            if hit:
                csv_rows.append(f"{s},{model},{hit[0]['rmse_all_horizons']}")
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=MODEL_LABEL[model], color=COLORS[model], edgecolor="none")
    ax.set_xticks(x)
    labels = [s.replace("_snapdragon", "") for s in seqs]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"Sequence RMSE (m/s$^2$)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def horizon_curves(horizon_rows: list[dict], *, pdf: Path, png: Path, csv_path: Path) -> None:
    contexts = ["WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"]
    titles = {
        "WITHIN_EUROC": "Within EuRoC",
        "WITHIN_UZH": "Within UZH-FPV",
        "EUROC_TO_UZH": r"EuRoC $\rightarrow$ UZH",
        "UZH_TO_EUROC": r"UZH $\rightarrow$ EuRoC",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    csv_lines = ["evaluation_context,model,horizon_index,horizon_ms,equal_sequence_mean_rmse"]
    for ax, ctx in zip(axes.ravel(), contexts, strict=True):
        for model in MODEL_ORDER:
            pts = [r for r in horizon_rows if r["evaluation_context"] == ctx and r["model"] == model]
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
        ax.set_title(titles[ctx])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Horizon (ms)")
        ax.set_ylabel(r"Equal-sequence RMSE (m/s$^2$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)


def skill_by_sequence(rows: list[dict], *, pdf: Path, png: Path, csv_path: Path) -> None:
    contexts = ["WITHIN_EUROC", "WITHIN_UZH", "EUROC_TO_UZH", "UZH_TO_EUROC"]
    titles = {
        "WITHIN_EUROC": "Within EuRoC",
        "WITHIN_UZH": "Within UZH-FPV",
        "EUROC_TO_UZH": r"EuRoC $\rightarrow$ UZH",
        "UZH_TO_EUROC": r"UZH $\rightarrow$ EuRoC",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    csv_lines = ["evaluation_context,sequence_id,model,skill"]
    learned = ["B1_LINEAR", "B2_RIDGE_QF", "B3_RIDGE_QF_QW"]
    for ax, ctx in zip(axes.ravel(), contexts, strict=True):
        sub = [r for r in rows if r["evaluation_context"] == ctx and r["model"] in learned]
        seqs = []
        for r in sub:
            if r["sequence_id"] not in seqs:
                seqs.append(r["sequence_id"])
        x = np.arange(len(seqs), dtype=np.float64)
        width = 0.24
        for i, model in enumerate(learned):
            vals = []
            for s in seqs:
                hit = [r for r in sub if r["sequence_id"] == s and r["model"] == model]
                vals.append(float(hit[0]["skill"]) if hit and hit[0]["skill"] != "" else np.nan)
                if hit:
                    csv_lines.append(f"{ctx},{s},{model},{hit[0]['skill']}")
            ax.bar(x + (i - 1) * width, vals, width=width, label=MODEL_LABEL[model], color=COLORS[model], edgecolor="none")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_snapdragon", "") for s in seqs], rotation=40, ha="right", fontsize=7)
        ax.set_title(titles[ctx])
        ax.set_ylabel("Skill vs persistence")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    _save(fig, pdf, png)
