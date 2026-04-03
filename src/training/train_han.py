"""Training pipeline for HAN-based disease-gene link prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for HAN training") from exc

from src.evaluation.metrics import compute_auc_roc
from src.models.han_model import HANLinkPredictor
from src.training.trainer_utils import (
    build_pair_tensors,
    load_split_dataframe,
)
from src.utils.io import ensure_dir, save_dataframe


def _merge_scores(split_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    merged = split_df.merge(
        score_df,
        on=["disease_local_id", "gene_local_id"],
        how="left",
    )
    if merged["score_han"].isnull().any():
        raise RuntimeError("Some HAN scores are missing after merge.")
    return merged


def run_han_training(
    hetero_graph_path: str | Path,
    train_split_path: str | Path,
    val_split_path: str | Path,
    test_split_path: str | Path,
    model_config: dict[str, Any],
    output_dir: str | Path,
    disease_type: str,
    gene_type: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Train HAN on disease-gene links and export predictions."""
    data = torch.load(hetero_graph_path, map_location=device)
    data = data.to(device)
    if not hasattr(data, "metadata"):
        raise RuntimeError("Loaded hetero graph does not expose metadata().")

    train_df = load_split_dataframe(train_split_path)
    val_df = load_split_dataframe(val_split_path)
    test_df = load_split_dataframe(test_split_path)

    metadata = data.metadata()
    num_nodes_dict = {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}

    model = HANLinkPredictor(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        output_dim=int(model_config["output_dim"]),
        heads=int(model_config["heads"]),
        dropout=float(model_config["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config["lr"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
    )

    epochs = int(model_config["epochs"])
    _ = int(model_config["batch_size"])

    for epoch in tqdm(range(1, epochs + 1), desc="HAN Training", leave=False):
        model.train()
        disease_idx, gene_idx, labels = build_pair_tensors(
            train_df,
            disease_col="disease_local_id",
            gene_col="gene_local_id",
            label_col="label",
            device=device,
        )
        optimizer.zero_grad()
        loss = model.loss(
            data=data,
            disease_indices=disease_idx,
            gene_indices=gene_idx,
            labels=labels,
            disease_type=disease_type,
            gene_type=gene_type,
        )
        loss.backward()
        optimizer.step()

        if int(model_config.get("eval_every", 5)) > 0 and epoch % int(
            model_config.get("eval_every", 5)
        ) == 0:
            val_scores = model.score_pairs(
                data=data,
                pairs_df=val_df,
                disease_type=disease_type,
                gene_type=gene_type,
                disease_col="disease_local_id",
                gene_col="gene_local_id",
                device=device,
            )
            val_auc = compute_auc_roc(
                y_true=val_df["label"].astype(int).to_numpy(),
                y_score=val_scores["score_han"].to_numpy(),
            )
            tqdm.write(
                f"Epoch {epoch:03d} | loss={float(loss.item()):.4f} | val_auc={val_auc:.4f}"
            )

    val_scores = model.score_pairs(
        data=data,
        pairs_df=val_df,
        disease_type=disease_type,
        gene_type=gene_type,
        disease_col="disease_local_id",
        gene_col="gene_local_id",
        device=device,
    )
    test_scores = model.score_pairs(
        data=data,
        pairs_df=test_df,
        disease_type=disease_type,
        gene_type=gene_type,
        disease_col="disease_local_id",
        gene_col="gene_local_id",
        device=device,
    )

    val_pred = _merge_scores(val_df, val_scores)
    test_pred = _merge_scores(test_df, test_scores)

    out_dir = ensure_dir(output_dir)
    weights_path = out_dir / "han_weights.pt"
    model.save_weights(weights_path)

    val_path = out_dir / "han_val_predictions.csv"
    test_path = out_dir / "han_test_predictions.csv"
    save_dataframe(val_pred, val_path)
    save_dataframe(test_pred, test_path)

    return {
        "weights_path": weights_path,
        "val_predictions": val_pred,
        "test_predictions": test_pred,
        "val_predictions_path": val_path,
        "test_predictions_path": test_path,
    }
