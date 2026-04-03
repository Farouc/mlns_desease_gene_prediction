"""Shared utilities for training and batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def infer_device(requested: str) -> str:
    """Resolve requested device string into an available runtime device."""
    if requested != "auto":
        return requested
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_split_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a split CSV with required columns."""
    df = pd.read_csv(path)
    required = {
        "disease_local_id",
        "gene_local_id",
        "disease_global_id",
        "gene_global_id",
        "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Split file {path} is missing columns: {sorted(missing)}")
    return df


def iterate_minibatches(
    df: pd.DataFrame,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
):
    """Yield dataframe batches."""
    indices = np.arange(len(df))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield df.iloc[batch_idx].reset_index(drop=True)


def build_pair_tensors(
    batch_df: pd.DataFrame,
    disease_col: str,
    gene_col: str,
    label_col: str = "label",
    device: str = "cpu",
):
    """Convert batch dataframe columns into PyTorch tensors."""
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor conversion.")

    disease_idx = torch.tensor(
        batch_df[disease_col].astype(int).tolist(), dtype=torch.long, device=device
    )
    gene_idx = torch.tensor(
        batch_df[gene_col].astype(int).tolist(), dtype=torch.long, device=device
    )
    labels = torch.tensor(
        batch_df[label_col].astype(float).tolist(), dtype=torch.float32, device=device
    )
    return disease_idx, gene_idx, labels
