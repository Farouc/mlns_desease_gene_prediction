"""Alpha fusion trade-off plots for hybrid score analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for alpha trade-off plotting.") from exc

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    apply_publication_style,
    compute_ranking_metrics_from_scores,
    load_model_predictions,
    save_figure,
)

_ALPHA_SWEEP_CANDIDATES = [
    "alpha_tradeoff_metrics.csv",
    "alpha_sweep.csv",
    "alpha_tradeoff.csv",
    "alpha_vs_performance.csv",
    "alpha_metrics.csv",
]
_REQUIRED_SWEEP_COLUMNS = {"alpha", "auc_pr", "hits@10", "mrr"}


def _load_existing_alpha_sweep(result_dir: Path) -> pd.DataFrame | None:
    """Load precomputed alpha sweep metrics when present."""
    for filename in _ALPHA_SWEEP_CANDIDATES:
        path = result_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if _REQUIRED_SWEEP_COLUMNS.issubset(df.columns):
            return df.sort_values("alpha").reset_index(drop=True)
    return None


def _compute_alpha_sweep(result_dir: Path, num_points: int) -> pd.DataFrame:
    """Approximate alpha sweep from saved HAN/path scores without retraining."""
    hybrid_df = load_model_predictions(result_dir, model="hybrid")
    gnn_col: str | None = None
    for candidate in ("score_gnn_for_hybrid", "score_han", "score_node2vec"):
        if candidate in hybrid_df.columns:
            gnn_col = candidate
            break

    required_cols = {"disease_local_id", "label", "score_path"}
    missing_cols = required_cols.difference(hybrid_df.columns)
    if missing_cols:
        raise RuntimeError(f"Hybrid predictions missing required columns: {sorted(missing_cols)}")
    if gnn_col is None:
        raise RuntimeError(
            "Hybrid predictions missing a GNN score column. "
            "Expected one of: score_gnn_for_hybrid, score_han, score_node2vec."
        )

    labels = hybrid_df["label"].astype(int).to_numpy()
    han_scores = hybrid_df[gnn_col].astype(float).to_numpy()
    path_scores = hybrid_df["score_path"].astype(float).to_numpy()
    alpha_values = np.linspace(0.0, 1.0, num_points)

    rows: list[dict[str, float]] = []
    for alpha in alpha_values:
        fused = float(alpha) * han_scores + (1.0 - float(alpha)) * path_scores
        auc_pr = float(average_precision_score(labels, fused)) if len(np.unique(labels)) >= 2 else float("nan")

        eval_df = hybrid_df[["disease_local_id", "label"]].copy()
        eval_df["score_alpha"] = fused
        hits10, mrr = compute_ranking_metrics_from_scores(
            eval_df,
            disease_col="disease_local_id",
            score_col="score_alpha",
            label_col="label",
            top_k=10,
        )

        rows.append(
            {
                "alpha": float(alpha),
                "auc_pr": auc_pr,
                "hits@10": float(hits10),
                "mrr": float(mrr),
            }
        )

    return pd.DataFrame(rows)


def generate_alpha_tradeoff_plot(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
    num_points: int = 21,
) -> bool:
    """Generate alpha-vs-performance curves (AUC-PR, Hits@10, MRR)."""
    apply_publication_style()

    result_path = Path(result_dir)
    sweep_df = _load_existing_alpha_sweep(result_path)
    if sweep_df is None:
        sweep_df = _compute_alpha_sweep(result_path, num_points=num_points)
        sweep_path = result_path / "alpha_tradeoff_metrics.csv"
        if not sweep_path.exists():
            sweep_df.to_csv(sweep_path, index=False)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.plot(
        sweep_df["alpha"],
        sweep_df["auc_pr"],
        marker="o",
        markersize=4,
        color="#4E79A7",
        label="AUC-PR",
    )
    ax.plot(
        sweep_df["alpha"],
        sweep_df["mrr"],
        marker="s",
        markersize=4,
        color="#59A14F",
        label="MRR",
    )
    ax.plot(
        sweep_df["alpha"],
        sweep_df["hits@10"],
        marker="^",
        markersize=4,
        color="#E15759",
        label="Hits@10",
    )

    best_idx = int(sweep_df["auc_pr"].idxmax())
    best_alpha = float(sweep_df.loc[best_idx, "alpha"])
    ax.axvline(best_alpha, color="black", linestyle=":", linewidth=1.5, label=f"Best AUC-PR alpha={best_alpha:.2f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Alpha (weight on GNN score)")
    ax.set_ylabel("Metric value")
    ax.set_title("Hybrid Alpha Trade-off")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="alpha_tradeoff.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved
