"""Hits@k curve visualization for model ranking quality."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    apply_publication_style,
    compute_hits_curve_from_ranked,
    load_model_ranked_predictions,
    save_figure,
)


def _infer_disease_column(df_columns: list[str]) -> str:
    """Infer disease identifier column from ranked prediction columns."""
    if "disease_local_id" in df_columns:
        return "disease_local_id"
    if "disease_global_id" in df_columns:
        return "disease_global_id"
    raise RuntimeError(f"Unable to infer disease column from: {df_columns}")


def generate_hits_at_k_plot(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
    k_max: int = 50,
) -> bool:
    """Generate Hits@k curves for k in [1, k_max] across all models."""
    apply_publication_style()

    x = np.arange(1, k_max + 1)
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    plotted_any = False

    for model in MODEL_ORDER:
        ranked_path = Path(result_dir) / f"{model}_ranked_predictions.csv"
        if not ranked_path.exists():
            continue

        ranked_df = load_model_ranked_predictions(result_dir, model=model)
        if ranked_df.empty or "label" not in ranked_df.columns or "rank" not in ranked_df.columns:
            continue

        disease_col = _infer_disease_column(ranked_df.columns.tolist())
        curve = compute_hits_curve_from_ranked(
            ranked=ranked_df,
            disease_col=disease_col,
            label_col="label",
            k_max=k_max,
        )
        ax.plot(
            x,
            curve,
            color=MODEL_COLORS.get(model),
            label=MODEL_DISPLAY_NAMES.get(model, model),
        )
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f"No ranked prediction files available in {result_dir}")

    ax.set_xlim(1, k_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("k")
    ax.set_ylabel("Hits@k")
    ax.set_title("Hits@k Curves")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="hits_at_k.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved
