"""Training dynamics plots from saved per-epoch history artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    MODEL_COLORS,
    apply_publication_style,
    save_figure,
)


def _load_history(result_dir: Path, model: str) -> pd.DataFrame | None:
    """Load per-epoch training history for a model if present."""
    path = result_dir / f"{model}_training_history.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def _has_metric(df: pd.DataFrame, col: str) -> bool:
    """Check whether metric column exists and has at least one non-null value."""
    return col in df.columns and df[col].notna().any()


def generate_training_dynamics_plot(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Generate a compact training dynamics figure for HAN and Node2Vec."""
    apply_publication_style()

    result_path = Path(result_dir)
    histories: list[tuple[str, pd.DataFrame]] = []
    for model in ("node2vec", "han"):
        df = _load_history(result_path, model)
        if df is not None and not df.empty:
            histories.append((model, df))

    if not histories:
        return False

    ncols = len(histories)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6.8 * ncols, 5.2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (model, hist_df) in zip(axes_flat, histories, strict=False):
        color = MODEL_COLORS.get(model, "#4E79A7")
        ax.plot(
            hist_df["epoch"],
            hist_df["train_loss"],
            color=color,
            label="Train loss",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax.set_title(f"{model.upper()} Training Dynamics")

        has_val_auc_pr = _has_metric(hist_df, "val_auc_pr")
        has_val_auc_roc = _has_metric(hist_df, "val_auc_roc")

        if has_val_auc_pr or has_val_auc_roc:
            ax2 = ax.twinx()
            if has_val_auc_pr:
                ax2.plot(
                    hist_df["epoch"],
                    hist_df["val_auc_pr"],
                    color="#E15759",
                    linestyle="--",
                    marker="o",
                    markersize=3,
                    label="Val AUC-PR",
                )
            if has_val_auc_roc:
                ax2.plot(
                    hist_df["epoch"],
                    hist_df["val_auc_roc"],
                    color="#59A14F",
                    linestyle="-.",
                    marker="s",
                    markersize=3,
                    label="Val AUC-ROC",
                )
            ax2.set_ylabel("Validation metric")
            ax2.set_ylim(0.0, 1.0)

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
        else:
            ax.legend(loc="best", frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="training_dynamics.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved

