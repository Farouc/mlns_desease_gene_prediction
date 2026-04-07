"""Evaluation orchestration and figure generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import precision_recall_curve, roc_curve
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for plotting ROC/PR curves") from exc

from src.evaluation.metrics import compute_auc_pr, compute_auc_roc
from src.evaluation.ranking import compute_ranking_metrics, rank_predictions_per_disease
from src.utils.io import ensure_dir, save_json


# ── Shared style ──────────────────────────────────────────────────────────────

COLORS = {
    "primary":   "#2563EB",   # blue
    "secondary": "#DC2626",   # red
    "tertiary":  "#16A34A",   # green
    "accent":    "#9333EA",   # purple
    "baseline":  "#6B7280",   # gray
}

MODEL_COLORS = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["tertiary"],
    COLORS["accent"],
    COLORS["baseline"],
]


def _apply_elegant_style(ax) -> None:
    """Apply a clean, publication-ready style to an axes object."""
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#CCCCCC", linewidth=0.6, linestyle="-")
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.3, linestyle="-")
    ax.minorticks_on()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("#888888")
    ax.tick_params(axis="both", labelsize=11, length=3, width=0.7)
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Container for computed metrics and ranked predictions."""

    metrics: dict[str, float]
    ranked_predictions: pd.DataFrame


def evaluate_predictions(
    predictions: pd.DataFrame,
    score_col: str,
    label_col: str,
    disease_col: str,
    gene_col: str,
    top_k: int,
) -> EvaluationResult:
    """Evaluate binary and ranking metrics from prediction dataframe."""
    y_true = predictions[label_col].astype(int).to_numpy()
    y_score = predictions[score_col].astype(float).to_numpy()

    auc_roc = compute_auc_roc(y_true=y_true, y_score=y_score)
    auc_pr = compute_auc_pr(y_true=y_true, y_score=y_score)

    ranked = rank_predictions_per_disease(
        predictions=predictions,
        disease_col=disease_col,
        gene_col=gene_col,
        score_col=score_col,
        label_col=label_col,
    )
    ranking_metrics = compute_ranking_metrics(
        ranked_predictions=ranked,
        disease_col=disease_col,
        label_col=label_col,
        k=top_k,
    )

    metrics = {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        **ranking_metrics,
    }
    return EvaluationResult(metrics=metrics, ranked_predictions=ranked)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(
    predictions: pd.DataFrame,
    score_col: str,
    label_col: str,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Plot and save ROC and PR curves."""
    out_dir = ensure_dir(output_dir)
    y_true = predictions[label_col].astype(int).to_numpy()
    y_score = predictions[score_col].astype(float).to_numpy()

    roc_fpr, roc_tpr, _ = roc_curve(y_true, y_score)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)

    auc_roc = compute_auc_roc(y_true, y_score)
    auc_pr  = compute_auc_pr(y_true, y_score)

    # ── ROC curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        roc_fpr, roc_tpr,
        color=COLORS["primary"], linewidth=2.0,
        label=f"AUC-ROC = {auc_roc:.3f}",
    )
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", linewidth=1.2, color="#AAAAAA", label="Random",
    )
    ax.fill_between(roc_fpr, roc_tpr, alpha=0.07, color=COLORS["primary"])
    ax.set_xlabel("False Positive Rate", fontsize=13, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=13, labelpad=8)
    ax.set_title(f"ROC Curve — {prefix}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    _apply_elegant_style(ax)
    fig.tight_layout()
    roc_path = out_dir / f"{prefix}_roc_curve.png"
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── PR curve ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(
        pr_recall, pr_precision,
        color=COLORS["secondary"], linewidth=2.0,
        label=f"AUC-PR = {auc_pr:.3f}",
    )
    baseline = y_true.mean()
    ax.axhline(
        baseline, linestyle="--", linewidth=1.2,
        color="#AAAAAA", label=f"Random ({baseline:.2f})",
    )
    ax.fill_between(pr_recall, pr_precision, alpha=0.07, color=COLORS["secondary"])
    ax.set_xlabel("Recall", fontsize=13, labelpad=8)
    ax.set_ylabel("Precision", fontsize=13, labelpad=8)
    ax.set_title(f"Precision-Recall Curve — {prefix}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    _apply_elegant_style(ax)
    fig.tight_layout()
    pr_path = out_dir / f"{prefix}_pr_curve.png"
    fig.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"roc_curve": roc_path, "pr_curve": pr_path}


def plot_alpha_performance(
    alpha_values: list[float],
    metric_values: list[float],
    metric_name: str,
    output_path: str | Path,
) -> None:
    """Plot alpha ablation curve for hybrid fusion."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_idx = int(np.argmax(metric_values))
    best_alpha = alpha_values[best_idx]
    best_metric = metric_values[best_idx]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        alpha_values, metric_values,
        color=COLORS["primary"], linewidth=2.0,
        marker="o", markersize=5, markerfacecolor="white",
        markeredgewidth=1.5, markeredgecolor=COLORS["primary"],
        zorder=3,
    )
    ax.scatter(
        [best_alpha], [best_metric],
        color=COLORS["secondary"], s=80, zorder=4,
        label=f"Best α = {best_alpha:.2f} ({metric_name} = {best_metric:.3f})",
    )
    ax.axvline(
        best_alpha, linestyle="--", linewidth=1.0,
        color=COLORS["secondary"], alpha=0.5,
    )
    ax.set_xlabel("α  (weight on GNN score)", fontsize=13, labelpad=8)
    ax.set_ylabel(metric_name, fontsize=13, labelpad=8)
    ax.set_title(f"Hybrid Fusion: α vs {metric_name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    _apply_elegant_style(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(
    metric_by_model: dict[str, float],
    metric_name: str,
    output_path: str | Path,
) -> None:
    """Plot bar chart for model ablation comparison."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = list(metric_by_model.keys())
    values = [metric_by_model[label] for label in labels]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(5.5, len(labels) * 1.4), 4.5))

    bars = ax.bar(
        labels, values,
        color=colors, width=0.5,
        edgecolor="white", linewidth=0.8,
        zorder=3,
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333",
        )

    ax.set_ylabel(metric_name, fontsize=13, labelpad=8)
    ax.set_title(f"Model Comparison — {metric_name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, min(1.0, max(values) * 1.15))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, rotation=15, ha="right")
    _apply_elegant_style(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    """Save metrics dictionary to JSON."""
    save_json(metrics, path)