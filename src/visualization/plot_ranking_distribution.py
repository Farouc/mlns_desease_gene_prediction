"""Distribution plots for rank positions of true disease-gene links."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    apply_publication_style,
    load_model_ranked_predictions,
    save_figure,
)


def _positive_rank_positions(result_dir: Path, model: str) -> np.ndarray:
    """Return rank positions for positive edges from ranked predictions."""
    ranked_df = load_model_ranked_predictions(result_dir, model=model)
    if ranked_df.empty or "label" not in ranked_df.columns or "rank" not in ranked_df.columns:
        return np.array([], dtype=int)

    positives = ranked_df.loc[ranked_df["label"].astype(int) == 1, "rank"]
    if positives.empty:
        return np.array([], dtype=int)
    return positives.astype(int).to_numpy()


def generate_ranking_distribution_plot(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Compare positive-edge rank distributions for Node2Vec and Hybrid."""
    apply_publication_style()
    result_path = Path(result_dir)

    node2vec_path = result_path / "node2vec_ranked_predictions.csv"
    hybrid_path = result_path / "hybrid_ranked_predictions.csv"
    if not node2vec_path.exists() or not hybrid_path.exists():
        raise RuntimeError("Both node2vec_ranked_predictions.csv and hybrid_ranked_predictions.csv are required.")

    ranks_by_model = {
        "node2vec": _positive_rank_positions(result_path, "node2vec"),
        "hybrid": _positive_rank_positions(result_path, "hybrid"),
    }

    if not any(arr.size > 0 for arr in ranks_by_model.values()):
        raise RuntimeError("No positive-edge rank positions found for Node2Vec/Hybrid.")

    max_rank = max(int(arr.max()) for arr in ranks_by_model.values() if arr.size > 0)
    if max_rank <= 1:
        bins = np.array([1, 2])
    else:
        bins = np.unique(np.geomspace(1, max_rank + 1, num=45).astype(int))
        if len(bins) < 5:
            bins = np.linspace(1, max_rank + 1, num=10, dtype=int)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for model, ranks in ranks_by_model.items():
        if ranks.size == 0:
            continue
        ax.hist(
            ranks,
            bins=bins,
            alpha=0.25,
            linewidth=1.8,
            histtype="stepfilled",
            color=MODEL_COLORS.get(model),
            label=f"{MODEL_DISPLAY_NAMES.get(model, model)} (n={ranks.size})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Rank position of true edge (log scale)")
    ax.set_ylabel("Count")
    ax.set_title("True-Edge Rank Distribution: Node2Vec vs Hybrid")
    ax.legend(loc="upper right", frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="ranking_distribution.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved
