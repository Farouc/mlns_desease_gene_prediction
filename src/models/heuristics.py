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


@dataclass(frozen=True)
class HeuristicNormalizationStats:
    """Train-derived normalization statistics for heuristic scores."""

    mean_common_neighbors: float
    std_common_neighbors: float
    mean_adamic_adar: float
    std_adamic_adar: float


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


def compute_heuristic_normalization_stats(
    scores_df: pd.DataFrame,
) -> HeuristicNormalizationStats:
    """Compute normalization statistics from raw heuristic scores."""
    mean_cn = float(scores_df["score_common_neighbors"].mean())
    std_cn = float(scores_df["score_common_neighbors"].std(ddof=0))
    mean_aa = float(scores_df["score_adamic_adar"].mean())
    std_aa = float(scores_df["score_adamic_adar"].std(ddof=0))

    return HeuristicNormalizationStats(
        mean_common_neighbors=mean_cn,
        std_common_neighbors=std_cn if std_cn > 0.0 else 1.0,
        mean_adamic_adar=mean_aa,
        std_adamic_adar=std_aa if std_aa > 0.0 else 1.0,
    )


def _normalize_score(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def score_pairs_with_heuristics(
    graph: nx.Graph,
    pairs_df: pd.DataFrame,
    disease_col: str = "disease_global_id",
    gene_col: str = "gene_global_id",
    normalization_stats: HeuristicNormalizationStats | None = None,
) -> pd.DataFrame:
    """Score disease-gene pairs with Common Neighbors and Adamic-Adar."""
    rows: list[dict[str, float | int]] = []

    for row in pairs_df.itertuples(index=False):
        disease_id = int(getattr(row, disease_col))
        gene_id = int(getattr(row, gene_col))

        cn = common_neighbors_score(graph, disease_id, gene_id)
        aa = adamic_adar_score(graph, disease_id, gene_id)
        avg_raw = (cn + aa) / 2.0

        if normalization_stats is None:
            rows.append(
                {
                    "disease_global_id": disease_id,
                    "gene_global_id": gene_id,
                    "score_common_neighbors": cn,
                    "score_adamic_adar": aa,
                    "score_common_neighbors_norm": float("nan"),
                    "score_adamic_adar_norm": float("nan"),
                    "score_heuristic_avg_raw": avg_raw,
                    "score_heuristic_avg": avg_raw,
                }
            )
        else:
            cn_norm = _normalize_score(
                cn,
                normalization_stats.mean_common_neighbors,
                normalization_stats.std_common_neighbors,
            )
            aa_norm = _normalize_score(
                aa,
                normalization_stats.mean_adamic_adar,
                normalization_stats.std_adamic_adar,
            )
            avg_norm = (cn_norm + aa_norm) / 2.0

            rows.append(
                {
                    "disease_global_id": disease_id,
                    "gene_global_id": gene_id,
                    "score_common_neighbors": cn,
                    "score_adamic_adar": aa,
                    "score_common_neighbors_norm": cn_norm,
                    "score_adamic_adar_norm": aa_norm,
                    "score_heuristic_avg_raw": avg_raw,
                    "score_heuristic_avg": avg_norm,
                }
            )

    return pd.DataFrame(rows)
