"""Path extraction utilities for metapath-level interpretability."""

from __future__ import annotations

import pandas as pd

from src.graph.metapaths import MetapathCounter


def extract_paths_for_pair(
    adjacency: dict[str, dict[str, dict[int, list[int]]]],
    start_id: int,
    end_id: int,
    metapath_types: list[str],
    max_paths: int = 20,
    avoid_cycles: bool = True,
) -> list[list[dict[str, int | str]]]:
    """Extract concrete typed paths for one pair under one metapath."""
    if len(metapath_types) < 2:
        return []

    extracted: list[list[dict[str, int | str]]] = []

    def neighbors(src_type: str, dst_type: str, src_id: int) -> list[int]:
        return adjacency.get(src_type, {}).get(dst_type, {}).get(src_id, [])

    def dfs(
        step: int,
        current_id: int,
        current_path: list[tuple[str, int]],
        visited: set[tuple[str, int]],
    ) -> None:
        if len(extracted) >= max_paths:
            return

        current_type = metapath_types[step]
        if step == len(metapath_types) - 1:
            if current_id == end_id:
                extracted.append(
                    [{"type": node_type, "id": node_id} for node_type, node_id in current_path]
                )
            return

        next_type = metapath_types[step + 1]
        for next_id in neighbors(current_type, next_type, current_id):
            typed = (next_type, int(next_id))
            if avoid_cycles and typed in visited:
                continue
            dfs(
                step + 1,
                int(next_id),
                current_path + [typed],
                visited if not avoid_cycles else visited | {typed},
            )

    start_typed = (metapath_types[0], int(start_id))
    dfs(0, int(start_id), [start_typed], {start_typed})
    return extracted


def compute_metapath_counts_for_predictions(
    predictions: pd.DataFrame,
    counter: MetapathCounter,
    metapaths: dict[str, list[str]],
    disease_col: str = "disease_local_id",
    gene_col: str = "gene_local_id",
) -> pd.DataFrame:
    """Compute metapath counts for a set of disease-gene predictions."""
    return counter.count_for_pairs(
        pairs=predictions,
        metapaths=metapaths,
        disease_col=disease_col,
        gene_col=gene_col,
    )
