"""Training pipeline for Node2Vec link prediction."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import networkx as nx
import pandas as pd

from src.evaluation.metrics import compute_auc_pr, compute_auc_roc
from src.models.node2vec_model import Node2VecConfig, Node2VecLinkPredictor
from src.training.trainer_utils import load_split_dataframe
from src.utils.io import ensure_dir, save_dataframe, save_json


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
    with Path(graph_path).open("rb") as handle:
        graph: nx.Graph = pickle.load(handle)
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

    train_df = load_split_dataframe(train_split_path)
    val_df = load_split_dataframe(val_split_path)
    test_df = load_split_dataframe(test_split_path)

    predictor = Node2VecLinkPredictor(cfg, device=device)

    eval_every = int(model_config.get("eval_every", 5))

    def _val_metrics(epoch: int, model: Node2VecLinkPredictor) -> dict[str, float]:
        val_scores_df = model.score_pairs(val_df)
        y_true = val_df["label"].astype(int).to_numpy()
        y_score = val_scores_df["score_node2vec"].to_numpy()
        return {
            "val_auc_roc": float(compute_auc_roc(y_true, y_score)),
            "val_auc_pr": float(compute_auc_pr(y_true, y_score)),
        }

    history_records = predictor.fit(
        graph=graph,
        num_nodes=num_nodes,
        eval_every=eval_every,
        eval_callback=_val_metrics,
    )

    train_scores = predictor.score_pairs(train_df)
    val_scores = predictor.score_pairs(val_df)
    test_scores = predictor.score_pairs(test_df)

    train_pred = _attach_labels(
        score_df=train_scores,
        split_df=train_df,
        score_col="score_node2vec",
        disease_col="disease_global_id",
        gene_col="gene_global_id",
    )
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

    train_path = out_dir / "node2vec_train_predictions.csv"
    val_path = out_dir / "node2vec_val_predictions.csv"
    test_path = out_dir / "node2vec_test_predictions.csv"
    history_path = out_dir / "node2vec_training_history.csv"
    history_summary_path = out_dir / "node2vec_training_summary.json"
    save_dataframe(train_pred, train_path)
    save_dataframe(val_pred, val_path)
    save_dataframe(test_pred, test_path)
    history_df = pd.DataFrame(history_records)
    if not history_df.empty and "epoch" in history_df.columns:
        history_df["epoch"] = history_df["epoch"].astype(int)
    save_dataframe(history_df, history_path)

    best_auc_pr = (
        float(history_df["val_auc_pr"].max())
        if "val_auc_pr" in history_df.columns and history_df["val_auc_pr"].notna().any()
        else None
    )
    best_epoch_auc_pr = (
        int(history_df.loc[history_df["val_auc_pr"].idxmax(), "epoch"])
        if "val_auc_pr" in history_df.columns and history_df["val_auc_pr"].notna().any()
        else None
    )
    summary = {
        "epochs": int(cfg.epochs),
        "eval_every": int(eval_every),
        "best_val_auc_pr": best_auc_pr,
        "best_val_auc_pr_epoch": best_epoch_auc_pr,
    }
    save_json(summary, history_summary_path)

    return {
        "weights_path": weights_path,
        "train_predictions": train_pred,
        "val_predictions": val_pred,
        "test_predictions": test_pred,
        "training_history": history_df,
        "train_predictions_path": train_path,
        "val_predictions_path": val_path,
        "test_predictions_path": test_path,
        "training_history_path": history_path,
        "training_summary_path": history_summary_path,
    }
