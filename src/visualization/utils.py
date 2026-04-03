"""Shared utilities for publication-ready visualization generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json

import numpy as np
import pandas as pd

DEFAULT_RUN_DIR = Path("experiments/results/final_full_cuda_hybrid_e15_fast")
DEFAULT_FIGURE_DIR = Path("experiments/figures/final_full_cuda_hybrid_e15_fast")

MODEL_ORDER = ["heuristics", "node2vec", "han", "hybrid"]
MODEL_DISPLAY_NAMES = {
    "heuristics": "Heuristics",
    "node2vec": "Node2Vec",
    "han": "HAN",
    "hybrid": "Hybrid",
}
MODEL_COLORS = {
    "heuristics": "#4E79A7",
    "node2vec": "#F28E2B",
    "han": "#59A14F",
    "hybrid": "#E15759",
}

METRIC_ORDER = ["auc_roc", "auc_pr", "hits@10", "mrr"]
METRIC_DISPLAY_NAMES = {
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "hits@10": "Hits@10",
    "mrr": "MRR",
}

MODEL_SCORE_COLUMNS = {
    "heuristics": "score_heuristic_avg",
    "node2vec": "score_node2vec",
    "han": "score_han",
    "hybrid": "score_hybrid",
}
MODEL_DISEASE_COLUMNS = {
    "heuristics": "disease_global_id",
    "node2vec": "disease_global_id",
    "han": "disease_local_id",
    "hybrid": "disease_local_id",
}


def apply_publication_style() -> None:
    """Apply consistent matplotlib style tuned for paper figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "figure.figsize": (9, 5.5),
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: str | Path) -> dict:
    """Load JSON as dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def available_models(result_dir: str | Path) -> list[str]:
    """Return models with prediction files available."""
    root = Path(result_dir)
    models: list[str] = []
    for model in MODEL_ORDER:
        if (root / f"{model}_predictions.csv").exists():
            models.append(model)
    return models


def load_model_metrics(result_dir: str | Path) -> dict[str, dict[str, float]]:
    """Load per-model metric JSON files."""
    root = Path(result_dir)
    metrics: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        path = root / f"{model}_metrics.json"
        if path.exists():
            metrics[model] = load_json(path)
    return metrics


def load_model_predictions(result_dir: str | Path, model: str) -> pd.DataFrame:
    """Load `<model>_predictions.csv`."""
    return pd.read_csv(Path(result_dir) / f"{model}_predictions.csv")


def load_model_ranked_predictions(result_dir: str | Path, model: str) -> pd.DataFrame:
    """Load `<model>_ranked_predictions.csv`."""
    return pd.read_csv(Path(result_dir) / f"{model}_ranked_predictions.csv")


def should_generate_figure(figure_path: str | Path, overwrite: bool) -> bool:
    """Return whether we should generate (or regenerate) a figure."""
    path = Path(figure_path)
    if path.exists() and not overwrite:
        return False
    return True


def save_figure(
    fig,
    figure_dir: str | Path,
    filename: str,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Save figure as PNG and optional PDF.

    Returns:
        True if saved, False if skipped due to existing file and overwrite=False.
    """
    out_dir = ensure_dir(figure_dir)
    png_path = out_dir / filename
    if not should_generate_figure(png_path, overwrite=overwrite):
        return False

    fig.savefig(png_path)
    if save_pdf:
        fig.savefig(out_dir / f"{Path(filename).stem}.pdf")
    return True


def compute_ranked_table(
    df: pd.DataFrame,
    disease_col: str,
    score_col: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """Rank predictions by disease and descending score."""
    ranked_parts: list[pd.DataFrame] = []
    for disease_id, group in df.groupby(disease_col):
        ordered = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        ordered[disease_col] = disease_id
        ranked_parts.append(ordered[[disease_col, score_col, label_col, "rank"]])
    if not ranked_parts:
        return pd.DataFrame(columns=[disease_col, score_col, label_col, "rank"])
    return pd.concat(ranked_parts, ignore_index=True)


def compute_hits_at_k(labels: Iterable[int], k: int) -> float:
    """Hits@k for a single ranked list."""
    labels_list = list(labels)
    if not labels_list:
        return 0.0
    return float(any(int(x) == 1 for x in labels_list[:k]))


def compute_reciprocal_rank(labels: Iterable[int]) -> float:
    """Reciprocal rank for a single ranked list."""
    for i, x in enumerate(labels, start=1):
        if int(x) == 1:
            return 1.0 / float(i)
    return 0.0


def compute_hits_curve_from_scores(
    df: pd.DataFrame,
    disease_col: str,
    score_col: str,
    label_col: str = "label",
    k_max: int = 50,
) -> np.ndarray:
    """Compute Hits@k for k=1..k_max from raw scores."""
    ranked = compute_ranked_table(df, disease_col=disease_col, score_col=score_col, label_col=label_col)
    return compute_hits_curve_from_ranked(
        ranked,
        disease_col=disease_col,
        label_col=label_col,
        k_max=k_max,
    )


def compute_hits_curve_from_ranked(
    ranked: pd.DataFrame,
    disease_col: str,
    label_col: str = "label",
    k_max: int = 50,
) -> np.ndarray:
    """Compute Hits@k for k=1..k_max from pre-ranked data."""
    values = np.zeros(k_max, dtype=float)
    if ranked.empty:
        return values

    grouped = [
        g.sort_values("rank")[label_col].astype(int).to_numpy()
        for _, g in ranked.groupby(disease_col)
    ]
    for k in range(1, k_max + 1):
        hits = [compute_hits_at_k(labels, k) for labels in grouped]
        values[k - 1] = float(np.mean(hits))
    return values


def compute_ranking_metrics_from_scores(
    df: pd.DataFrame,
    disease_col: str,
    score_col: str,
    label_col: str = "label",
    top_k: int = 10,
) -> tuple[float, float]:
    """Compute (Hits@top_k, MRR) from raw score table."""
    ranked = compute_ranked_table(df, disease_col=disease_col, score_col=score_col, label_col=label_col)
    if ranked.empty:
        return 0.0, 0.0

    hits_values: list[float] = []
    rr_values: list[float] = []
    for _, group in ranked.groupby(disease_col):
        labels = group.sort_values("rank")[label_col].astype(int).to_list()
        hits_values.append(compute_hits_at_k(labels, top_k))
        rr_values.append(compute_reciprocal_rank(labels))

    return float(np.mean(hits_values)), float(np.mean(rr_values))
