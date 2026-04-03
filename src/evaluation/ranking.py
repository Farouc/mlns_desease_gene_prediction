"""Ranking utilities for per-disease link prediction analysis."""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import hits_at_k, reciprocal_rank


def rank_predictions_per_disease(
    predictions: pd.DataFrame,
    disease_col: str,
    gene_col: str,
    score_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Create per-disease ranked predictions with rank indices."""
    ranked_frames: list[pd.DataFrame] = []

    for disease_id, group in predictions.groupby(disease_col):
        sorted_group = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        sorted_group["rank"] = range(1, len(sorted_group) + 1)
        sorted_group[disease_col] = disease_id
        ranked_frames.append(sorted_group[[disease_col, gene_col, score_col, label_col, "rank"]])

    if not ranked_frames:
        return pd.DataFrame(columns=[disease_col, gene_col, score_col, label_col, "rank"])

    return pd.concat(ranked_frames, ignore_index=True)


def compute_ranking_metrics(
    ranked_predictions: pd.DataFrame,
    disease_col: str,
    label_col: str,
    k: int,
) -> dict[str, float]:
    """Compute aggregate Hits@k and MRR over disease-specific rankings."""
    hits_values: list[float] = []
    rr_values: list[float] = []

    for _, group in ranked_predictions.groupby(disease_col):
        labels = group.sort_values("rank")[label_col].astype(int).tolist()
        hits_values.append(hits_at_k(labels, k=k))
        rr_values.append(reciprocal_rank(labels))

    if not hits_values:
        return {f"hits@{k}": 0.0, "mrr": 0.0}

    return {
        f"hits@{k}": float(sum(hits_values) / len(hits_values)),
        "mrr": float(sum(rr_values) / len(rr_values)),
    }
