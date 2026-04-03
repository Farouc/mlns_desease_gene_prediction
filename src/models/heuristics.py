"""Graph heuristic baselines for disease-gene link prediction."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class HeuristicScores:
    """Container for heuristic scores."""

    common_neighbors: float
    adamic_adar: float


def common_neighbors_score(graph: nx.Graph, node_u: int, node_v: int) -> float:
    """Compute Common Neighbors score for a node pair."""
    if graph.is_directed():
        undirected = graph.to_undirected()
    else:
        undirected = graph
    return float(len(list(nx.common_neighbors(undirected, node_u, node_v))))


def adamic_adar_score(graph: nx.Graph, node_u: int, node_v: int) -> float:
    """Compute Adamic-Adar score for a node pair."""
    if graph.is_directed():
        undirected = graph.to_undirected()
    else:
        undirected = graph

    score = 0.0
    for _, _, aa in nx.adamic_adar_index(undirected, [(node_u, node_v)]):
        score = float(aa)
    return score


def score_pairs_with_heuristics(
    graph: nx.Graph,
    pairs_df: pd.DataFrame,
    disease_col: str = "disease_global_id",
    gene_col: str = "gene_global_id",
) -> pd.DataFrame:
    """Score disease-gene pairs with Common Neighbors and Adamic-Adar."""
    rows: list[dict[str, float | int]] = []

    for row in pairs_df.itertuples(index=False):
        disease_id = int(getattr(row, disease_col))
        gene_id = int(getattr(row, gene_col))

        cn = common_neighbors_score(graph, disease_id, gene_id)
        aa = adamic_adar_score(graph, disease_id, gene_id)

        rows.append(
            {
                "disease_global_id": disease_id,
                "gene_global_id": gene_id,
                "score_common_neighbors": cn,
                "score_adamic_adar": aa,
                "score_heuristic_avg": (cn + aa) / 2.0,
            }
        )

    return pd.DataFrame(rows)
