"""Generate interpretability outputs for top disease-gene predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import save_json


def _counts_by_pair(
    count_df: pd.DataFrame,
    disease_col: str,
    gene_col: str,
) -> dict[tuple[int, int], dict[str, int]]:
    grouped: dict[tuple[int, int], dict[str, int]] = {}
    for row in count_df.itertuples(index=False):
        key = (int(getattr(row, disease_col)), int(getattr(row, gene_col)))
        grouped.setdefault(key, {})[str(row.metapath)] = int(row.count)
    return grouped


def build_explanations(
    predictions: pd.DataFrame,
    metapath_count_df: pd.DataFrame,
    score_col: str,
    metapath_weights: dict[str, float] | None,
    top_n: int,
    disease_col: str = "disease_local_id",
    gene_col: str = "gene_local_id",
) -> list[dict[str, Any]]:
    """Build explainability records for top scoring predictions."""
    top_predictions = predictions.sort_values(score_col, ascending=False).head(top_n)
    counts_lookup = _counts_by_pair(metapath_count_df, disease_col=disease_col, gene_col=gene_col)

    explanations: list[dict[str, Any]] = []
    for row in top_predictions.itertuples(index=False):
        disease_id = int(getattr(row, disease_col))
        gene_id = int(getattr(row, gene_col))
        score = float(getattr(row, score_col))
        counts = counts_lookup.get((disease_id, gene_id), {})

        if not counts:
            top_metapath = "none"
        else:
            if metapath_weights:
                weighted = {
                    name: counts.get(name, 0) * float(metapath_weights.get(name, 0.0))
                    for name in counts.keys()
                }
                top_metapath = max(weighted, key=weighted.get)
            else:
                top_metapath = max(counts, key=counts.get)

        explanations.append(
            {
                "disease": disease_id,
                "gene": gene_id,
                "score": score,
                "top_metapath": top_metapath,
                "path_counts": counts,
            }
        )

    return explanations


def save_explanations(explanations: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save explanations to JSON."""
    save_json(explanations, output_path)
