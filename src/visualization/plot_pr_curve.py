"""Precision-Recall comparison plot across all available models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for PR curve plotting.") from exc

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    MODEL_SCORE_COLUMNS,
    apply_publication_style,
    available_models,
    load_model_predictions,
    save_figure,
)


def generate_pr_curve_comparison(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Generate one PR curve figure comparing all model predictions."""
    apply_publication_style()

    models = [m for m in MODEL_ORDER if m in available_models(result_dir)]
    if not models:
        raise RuntimeError(f"No prediction files found in {result_dir}")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    baseline: float | None = None

    for model in models:
        pred_df = load_model_predictions(result_dir, model=model)
        score_col = MODEL_SCORE_COLUMNS.get(model)
        if score_col is None or score_col not in pred_df.columns:
            continue

        y_true = pred_df["label"].astype(int).to_numpy()
        y_score = pred_df[score_col].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_pr = float(average_precision_score(y_true, y_score))
        ax.plot(
            recall,
            precision,
            color=MODEL_COLORS.get(model),
            label=f"{MODEL_DISPLAY_NAMES.get(model, model)} (AUC-PR={auc_pr:.3f})",
        )

        if baseline is None:
            baseline = float(np.mean(y_true))

    if baseline is not None:
        ax.hlines(
            baseline,
            0.0,
            1.0,
            colors="gray",
            linestyles="--",
            linewidth=1.5,
            label=f"Random baseline ({baseline:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="pr_curve_comparison.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved
