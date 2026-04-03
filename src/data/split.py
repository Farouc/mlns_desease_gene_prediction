"""Deterministic train/val/test splitting for disease-gene link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_dataframe


@dataclass(frozen=True)
class SplitArtifacts:
    """Paths for generated split files."""

    train_path: Path
    val_path: Path
    test_path: Path


def extract_positive_disease_gene_pairs(
    encoded_edges: pd.DataFrame,
    disease_type: str,
    gene_type: str,
) -> pd.DataFrame:
    """Extract unique positive Disease-Gene pairs from encoded edges."""
    direct = encoded_edges[
        (encoded_edges["src_type"] == disease_type)
        & (encoded_edges["dst_type"] == gene_type)
    ][["src_local_id", "dst_local_id", "src_global_id", "dst_global_id"]].copy()
    direct.columns = [
        "disease_local_id",
        "gene_local_id",
        "disease_global_id",
        "gene_global_id",
    ]

    reverse = encoded_edges[
        (encoded_edges["src_type"] == gene_type)
        & (encoded_edges["dst_type"] == disease_type)
    ][["dst_local_id", "src_local_id", "dst_global_id", "src_global_id"]].copy()
    reverse.columns = [
        "disease_local_id",
        "gene_local_id",
        "disease_global_id",
        "gene_global_id",
    ]

    positives = pd.concat([direct, reverse], ignore_index=True)
    positives.drop_duplicates(inplace=True)
    positives.reset_index(drop=True, inplace=True)
    positives["label"] = 1
    return positives


def _split_dataframe(
    positives: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split positives into train/val/test with deterministic shuffling."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(positives))
    rng.shuffle(indices)

    n_total = len(indices)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        positives.iloc[train_idx].reset_index(drop=True),
        positives.iloc[val_idx].reset_index(drop=True),
        positives.iloc[test_idx].reset_index(drop=True),
    )


def _sample_unique_negatives(
    positives_set: set[tuple[int, int]],
    disease_nodes: np.ndarray,
    gene_nodes: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
    reserved: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Sample unique negative Disease-Gene pairs."""
    negatives: set[tuple[int, int]] = set()
    reserved = reserved or set()

    max_attempts = max(10 * n_samples, 10_000)
    attempts = 0
    while len(negatives) < n_samples and attempts < max_attempts:
        d = int(rng.choice(disease_nodes))
        g = int(rng.choice(gene_nodes))
        pair = (d, g)
        if pair in positives_set or pair in reserved or pair in negatives:
            attempts += 1
            continue
        negatives.add(pair)
        attempts += 1

    if len(negatives) < n_samples:
        raise RuntimeError(
            f"Could only sample {len(negatives)} negatives out of requested {n_samples}."
        )

    neg_df = pd.DataFrame(list(negatives), columns=["disease_local_id", "gene_local_id"])
    neg_df["label"] = 0
    return neg_df


def _attach_global_ids(
    pairs: pd.DataFrame,
    node_mapping: pd.DataFrame,
    disease_type: str,
    gene_type: str,
) -> pd.DataFrame:
    """Add global IDs to local Disease-Gene pairs."""
    disease_map = node_mapping[node_mapping["node_type"] == disease_type][
        ["local_id", "global_id"]
    ].rename(columns={"local_id": "disease_local_id", "global_id": "disease_global_id"})

    gene_map = node_mapping[node_mapping["node_type"] == gene_type][["local_id", "global_id"]].rename(
        columns={"local_id": "gene_local_id", "global_id": "gene_global_id"}
    )

    merged = pairs.merge(disease_map, on="disease_local_id", how="left")
    merged = merged.merge(gene_map, on="gene_local_id", how="left")
    if merged[["disease_global_id", "gene_global_id"]].isnull().any().any():
        raise RuntimeError("Failed to map some local IDs to global IDs.")
    return merged


def create_splits(
    encoded_edges_path: str | Path,
    node_mapping_path: str | Path,
    output_dir: str | Path,
    disease_type: str,
    gene_type: str,
    val_ratio: float,
    test_ratio: float,
    negative_ratio: int,
    seed: int,
) -> SplitArtifacts:
    """Generate train/val/test splits with positives and sampled negatives."""
    edges = pd.read_csv(encoded_edges_path)
    nodes = pd.read_csv(node_mapping_path)

    positives = extract_positive_disease_gene_pairs(edges, disease_type, gene_type)
    train_pos, val_pos, test_pos = _split_dataframe(positives, val_ratio, test_ratio, seed)

    disease_nodes = (
        nodes[nodes["node_type"] == disease_type]["local_id"].astype(int).to_numpy()
    )
    gene_nodes = nodes[nodes["node_type"] == gene_type]["local_id"].astype(int).to_numpy()

    positives_set = set(
        zip(
            positives["disease_local_id"].astype(int).tolist(),
            positives["gene_local_id"].astype(int).tolist(),
            strict=False,
        )
    )

    rng = np.random.RandomState(seed)
    used_negatives: set[tuple[int, int]] = set()

    def build_split(split_pos: pd.DataFrame) -> pd.DataFrame:
        n_neg = len(split_pos) * negative_ratio
        neg_df = _sample_unique_negatives(
            positives_set=positives_set,
            disease_nodes=disease_nodes,
            gene_nodes=gene_nodes,
            n_samples=n_neg,
            rng=rng,
            reserved=used_negatives,
        )
        for pair in zip(
            neg_df["disease_local_id"].astype(int).tolist(),
            neg_df["gene_local_id"].astype(int).tolist(),
            strict=False,
        ):
            used_negatives.add(pair)
        merged = pd.concat([split_pos, neg_df], ignore_index=True)
        merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        merged = _attach_global_ids(merged, nodes, disease_type, gene_type)
        return merged[
            [
                "disease_local_id",
                "gene_local_id",
                "disease_global_id",
                "gene_global_id",
                "label",
            ]
        ]

    train_df = build_split(train_pos)
    val_df = build_split(val_pos)
    test_df = build_split(test_pos)

    splits_dir = ensure_dir(output_dir)
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    save_dataframe(train_df, train_path)
    save_dataframe(val_df, val_path)
    save_dataframe(test_df, test_path)

    return SplitArtifacts(train_path=train_path, val_path=val_path, test_path=test_path)
