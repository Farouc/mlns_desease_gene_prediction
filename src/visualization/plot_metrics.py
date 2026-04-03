"""Model-wise metric comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    METRIC_DISPLAY_NAMES,
    METRIC_ORDER,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    apply_publication_style,
    load_model_metrics,
    save_figure,
)


def generate_model_comparison_bar(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Generate grouped bar chart comparing all models across key metrics."""
    apply_publication_style()
    metrics_by_model = load_model_metrics(result_dir)

    present_models = [m for m in MODEL_ORDER if m in metrics_by_model]
    if not present_models:
        raise RuntimeError(f"No model metric files found in {result_dir}")

    x = np.arange(len(METRIC_ORDER), dtype=float)
    width = 0.18 if len(present_models) >= 4 else 0.24

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for idx, model in enumerate(present_models):
        values = [float(metrics_by_model[model].get(metric, 0.0)) for metric in METRIC_ORDER]
        offset = (idx - (len(present_models) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_DISPLAY_NAMES.get(model, model),
            color=MODEL_COLORS.get(model, None),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY_NAMES[m] for m in METRIC_ORDER])
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model Comparison on Final CUDA Run")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="model_comparison_bar.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved
