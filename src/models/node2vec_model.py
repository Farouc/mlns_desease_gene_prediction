"""Node2Vec model wrapper for homogeneous graph embedding link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import networkx as nx
import pandas as pd

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for Node2Vec") from exc

try:
    from torch_geometric.nn import Node2Vec
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch-geometric is required for Node2Vec") from exc


@dataclass(frozen=True)
class Node2VecConfig:
    """Configuration for Node2Vec training."""

    embedding_dim: int
    walk_length: int
    context_size: int
    walks_per_node: int
    num_negative_samples: int
    p: float
    q: float
    sparse: bool
    epochs: int
    batch_size: int
    lr: float


class Node2VecLinkPredictor:
    """Train Node2Vec embeddings and score links by dot product."""

    def __init__(self, config: Node2VecConfig, device: str) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model: Node2Vec | None = None

    @staticmethod
    def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
        edges = list(graph.edges())
        if not edges:
            raise RuntimeError("Graph has no edges for Node2Vec training.")

        src = [int(u) for u, _ in edges]
        dst = [int(v) for _, v in edges]

        # Node2Vec expects a directed edge index; add both directions.
        src_all = src + dst
        dst_all = dst + src
        return torch.tensor([src_all, dst_all], dtype=torch.long)

    def fit(
        self,
        graph: nx.Graph,
        num_nodes: int,
        eval_every: int = 0,
        eval_callback: Callable[[int, "Node2VecLinkPredictor"], dict[str, float] | None] | None = None,
    ) -> list[dict[str, float]]:
        """Train Node2Vec embeddings and return per-epoch training history.

        Args:
            graph: Input homogeneous graph.
            num_nodes: Number of nodes in the graph.
            eval_every: If >0 and eval_callback is provided, evaluate every N epochs.
            eval_callback: Optional callback returning validation metrics for an epoch.

        Returns:
            List of dictionaries (one per epoch) with loss/time and optional metrics.
        """
        edge_index = self._edge_index_from_graph(graph).to(self.device)

        self.model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=self.config.embedding_dim,
            walk_length=self.config.walk_length,
            context_size=self.config.context_size,
            walks_per_node=self.config.walks_per_node,
            num_negative_samples=self.config.num_negative_samples,
            p=self.config.p,
            q=self.config.q,
            sparse=self.config.sparse,
            num_nodes=num_nodes,
        ).to(self.device)

        loader = self.model.loader(
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = (
            torch.optim.SparseAdam(list(self.model.parameters()), lr=self.config.lr)
            if self.config.sparse
            else torch.optim.Adam(list(self.model.parameters()), lr=self.config.lr)
        )

        history: list[dict[str, float]] = []
        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = perf_counter()
            batch_losses: list[float] = []
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_seconds = float(perf_counter() - epoch_start)
            avg_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
            record: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": avg_loss,
                "epoch_seconds": epoch_seconds,
            }

            if eval_callback is not None and eval_every > 0 and epoch % eval_every == 0:
                metrics = eval_callback(epoch, self)
                if metrics:
                    for key, value in metrics.items():
                        record[str(key)] = float(value)

            history.append(record)

        return history

    def embeddings(self) -> torch.Tensor:
        """Return all node embeddings."""
        if self.model is None:
            raise RuntimeError("Node2Vec model is not trained yet.")
        self.model.eval()
        with torch.no_grad():
            emb = self.model.embedding.weight.detach().cpu()
        return emb

    def score_pairs(
        self,
        pairs_df: pd.DataFrame,
        disease_col: str = "disease_global_id",
        gene_col: str = "gene_global_id",
    ) -> pd.DataFrame:
        """Score links by embedding dot product."""
        embeddings = self.embeddings()
        scores: list[dict[str, float | int]] = []

        for row in pairs_df.itertuples(index=False):
            d = int(getattr(row, disease_col))
            g = int(getattr(row, gene_col))
            score = float(torch.dot(embeddings[d], embeddings[g]).item())
            scores.append(
                {
                    "disease_global_id": d,
                    "gene_global_id": g,
                    "score_node2vec": score,
                }
            )

        return pd.DataFrame(scores)

    def save_weights(self, path: str | Path) -> None:
        """Save trained model parameters."""
        if self.model is None:
            raise RuntimeError("Node2Vec model is not trained yet.")
        torch.save(self.model.state_dict(), path)
