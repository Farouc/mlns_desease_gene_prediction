"""Evaluation orchestration and figure generation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from sklearn.metrics import precision_recall_curve, roc_curve
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for plotting ROC/PR curves") from exc

from src.evaluation.metrics import compute_auc_pr, compute_auc_roc
from src.evaluation.ranking import compute_ranking_metrics, rank_predictions_per_disease
from src.utils.io import ensure_dir, save_json


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

    roc_path = out_dir / f"{prefix}_roc_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(roc_fpr, roc_tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    pr_path = out_dir / f"{prefix}_pr_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(pr_recall, pr_precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

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

    plt.figure(figsize=(6, 4))
    plt.plot(alpha_values, metric_values, marker="o")
    plt.xlabel("alpha")
    plt.ylabel(metric_name)
    plt.title(f"alpha vs {metric_name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


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

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel(metric_name)
    plt.title(f"Ablation ({metric_name})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    """Save metrics dictionary to JSON."""
    save_json(metrics, path)
