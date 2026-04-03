"""Training pipeline for Node2Vec link prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.models.node2vec_model import Node2VecConfig, Node2VecLinkPredictor
from src.training.trainer_utils import load_split_dataframe
from src.utils.io import ensure_dir, save_dataframe


def _attach_labels(
    score_df: pd.DataFrame,
    split_df: pd.DataFrame,
    score_col: str,
    disease_col: str,
    gene_col: str,
) -> pd.DataFrame:
    merged = split_df.merge(
        score_df,
        left_on=[disease_col, gene_col],
        right_on=[disease_col, gene_col],
        how="left",
    )
    if merged[score_col].isnull().any():
        raise RuntimeError("Some Node2Vec scores are missing after merge.")
    return merged


def run_node2vec_training(
    graph_path: str | Path,
    node_mapping_path: str | Path,
    train_split_path: str | Path,
    val_split_path: str | Path,
    test_split_path: str | Path,
    model_config: dict[str, Any],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Train Node2Vec and export validation/test predictions."""
    graph = nx.read_gpickle(graph_path)
    nodes_df = pd.read_csv(node_mapping_path)
    num_nodes = int(nodes_df["global_id"].max()) + 1

    cfg = Node2VecConfig(
        embedding_dim=int(model_config["embedding_dim"]),
        walk_length=int(model_config["walk_length"]),
        context_size=int(model_config["context_size"]),
        walks_per_node=int(model_config["walks_per_node"]),
        num_negative_samples=int(model_config["num_negative_samples"]),
        p=float(model_config["p"]),
        q=float(model_config["q"]),
        sparse=bool(model_config["sparse"]),
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        lr=float(model_config["lr"]),
    )

    predictor = Node2VecLinkPredictor(cfg, device=device)
    predictor.fit(graph=graph, num_nodes=num_nodes)

    _ = load_split_dataframe(train_split_path)
    val_df = load_split_dataframe(val_split_path)
    test_df = load_split_dataframe(test_split_path)

    val_scores = predictor.score_pairs(val_df)
    test_scores = predictor.score_pairs(test_df)

    val_pred = _attach_labels(
        score_df=val_scores,
        split_df=val_df,
        score_col="score_node2vec",
        disease_col="disease_global_id",
        gene_col="gene_global_id",
    )
    test_pred = _attach_labels(
        score_df=test_scores,
        split_df=test_df,
        score_col="score_node2vec",
        disease_col="disease_global_id",
        gene_col="gene_global_id",
    )

    out_dir = ensure_dir(output_dir)
    weights_path = out_dir / "node2vec_weights.pt"
    predictor.save_weights(weights_path)

    val_path = out_dir / "node2vec_val_predictions.csv"
    test_path = out_dir / "node2vec_test_predictions.csv"
    save_dataframe(val_pred, val_path)
    save_dataframe(test_pred, test_path)

    return {
        "weights_path": weights_path,
        "val_predictions": val_pred,
        "test_predictions": test_pred,
        "val_predictions_path": val_path,
        "test_predictions_path": test_path,
    }
