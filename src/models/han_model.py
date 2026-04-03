"""Heterogeneous Attention Network for Disease-Gene link prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for HAN") from exc

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HANConv
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch-geometric is required for HAN") from exc


class HANLinkPredictor(nn.Module):
    """HAN encoder with typed node embeddings and dot-product decoder."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        num_nodes_dict: dict[str, int],
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_embeddings = nn.ModuleDict(
            {
                ntype: nn.Embedding(num_embeddings=count, embedding_dim=input_dim)
                for ntype, count in num_nodes_dict.items()
            }
        )

        self.han_conv = HANConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )

        self.projections = nn.ModuleDict(
            {ntype: nn.Linear(hidden_dim, output_dim) for ntype in num_nodes_dict.keys()}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode all node types into dense embeddings."""
        x_dict = {ntype: emb.weight for ntype, emb in self.node_embeddings.items()}
        z_dict = self.han_conv(x_dict, data.edge_index_dict)

        projected = {
            ntype: self.projections[ntype](self.dropout(F.elu(z)))
            for ntype, z in z_dict.items()
        }
        return projected

    @staticmethod
    def pair_scores(
        z_dict: dict[str, torch.Tensor],
        disease_indices: torch.Tensor,
        gene_indices: torch.Tensor,
        disease_type: str,
        gene_type: str,
    ) -> torch.Tensor:
        """Compute dot-product link scores for disease-gene pairs."""
        z_d = z_dict[disease_type][disease_indices]
        z_g = z_dict[gene_type][gene_indices]
        return torch.sum(z_d * z_g, dim=-1)

    def loss(
        self,
        data: HeteroData,
        disease_indices: torch.Tensor,
        gene_indices: torch.Tensor,
        labels: torch.Tensor,
        disease_type: str,
        gene_type: str,
    ) -> torch.Tensor:
        """Binary cross-entropy loss for link prediction."""
        z_dict = self.forward(data)
        logits = self.pair_scores(z_dict, disease_indices, gene_indices, disease_type, gene_type)
        return F.binary_cross_entropy_with_logits(logits, labels)

    @torch.no_grad()
    def score_pairs(
        self,
        data: HeteroData,
        pairs_df: pd.DataFrame,
        disease_type: str,
        gene_type: str,
        disease_col: str = "disease_local_id",
        gene_col: str = "gene_local_id",
        device: str = "cpu",
    ) -> pd.DataFrame:
        """Score a dataframe of disease-gene pairs."""
        self.eval()
        z_dict = self.forward(data)

        disease_idx = torch.tensor(
            pairs_df[disease_col].astype(int).tolist(), dtype=torch.long, device=device
        )
        gene_idx = torch.tensor(
            pairs_df[gene_col].astype(int).tolist(), dtype=torch.long, device=device
        )

        logits = self.pair_scores(z_dict, disease_idx, gene_idx, disease_type, gene_type)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        result = pairs_df[[disease_col, gene_col]].copy()
        result["score_han"] = probs
        return result

    def save_weights(self, path: str | Path) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
