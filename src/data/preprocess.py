"""Preprocessing utilities for typed node encoding and adjacency creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, save_dataframe, save_json


@dataclass(frozen=True)
class ProcessedArtifacts:
    """Paths for processed graph artifacts."""

    encoded_edges_path: Path
    node_mapping_path: Path
    metadata_path: Path


def encode_nodes_and_edges(
    edges: pd.DataFrame,
    node_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encode raw node IDs into local and global integer IDs.

    Args:
        edges: Standardized edge dataframe.
        node_types: Ordered node types for deterministic ID creation.

    Returns:
        Tuple of encoded edges, node mapping dataframe, metadata dictionary.
    """
    node_records: list[dict[str, Any]] = []
    key_to_ids: dict[tuple[str, str], tuple[int, int]] = {}

    global_counter = 0
    for node_type in node_types:
        src_nodes = edges.loc[edges["src_type"] == node_type, "src_raw_id"]
        dst_nodes = edges.loc[edges["dst_type"] == node_type, "dst_raw_id"]
        unique_nodes = sorted(set(src_nodes.astype(str)).union(set(dst_nodes.astype(str))))

        for local_id, raw_id in enumerate(unique_nodes):
            key_to_ids[(node_type, raw_id)] = (local_id, global_counter)
            node_records.append(
                {
                    "node_type": node_type,
                    "raw_id": raw_id,
                    "local_id": local_id,
                    "global_id": global_counter,
                }
            )
            global_counter += 1

    nodes_df = pd.DataFrame(node_records)

    encoded_rows: list[dict[str, Any]] = []
    for row in edges.itertuples(index=False):
        src_local, src_global = key_to_ids[(row.src_type, row.src_raw_id)]
        dst_local, dst_global = key_to_ids[(row.dst_type, row.dst_raw_id)]

        encoded_rows.append(
            {
                "src_type": row.src_type,
                "dst_type": row.dst_type,
                "src_raw_id": row.src_raw_id,
                "dst_raw_id": row.dst_raw_id,
                "src_local_id": src_local,
                "dst_local_id": dst_local,
                "src_global_id": src_global,
                "dst_global_id": dst_global,
                "edge_type": row.edge_type,
            }
        )

    encoded_edges = pd.DataFrame(encoded_rows)
    encoded_edges.drop_duplicates(inplace=True)
    encoded_edges.reset_index(drop=True, inplace=True)

    node_counts = (
        nodes_df.groupby("node_type")["local_id"].max().add(1).astype(int).to_dict()
        if not nodes_df.empty
        else {}
    )

    metadata: dict[str, Any] = {
        "num_nodes_total": int(len(nodes_df)),
        "num_edges": int(len(encoded_edges)),
        "node_types": node_types,
        "node_counts": node_counts,
    }
    return encoded_edges, nodes_df, metadata


def build_typed_adjacency(
    encoded_edges: pd.DataFrame,
    make_undirected: bool,
) -> dict[str, dict[str, dict[int, list[int]]]]:
    """Build adjacency grouped by source and destination node types.

    Args:
        encoded_edges: Encoded edges dataframe.
        make_undirected: If True, add reverse-direction entries.

    Returns:
        Nested dictionary: src_type -> dst_type -> src_local_id -> [dst_local_ids].
    """
    adjacency: dict[str, dict[str, dict[int, set[int]]]] = {}

    for row in encoded_edges.itertuples(index=False):
        adjacency.setdefault(row.src_type, {}).setdefault(row.dst_type, {}).setdefault(
            int(row.src_local_id), set()
        ).add(int(row.dst_local_id))

        if make_undirected:
            adjacency.setdefault(row.dst_type, {}).setdefault(row.src_type, {}).setdefault(
                int(row.dst_local_id), set()
            ).add(int(row.src_local_id))

    serialized: dict[str, dict[str, dict[int, list[int]]]] = {}
    for src_type, dst_map in adjacency.items():
        serialized[src_type] = {}
        for dst_type, src_map in dst_map.items():
            serialized[src_type][dst_type] = {
                int(src): sorted(list(dst_set)) for src, dst_set in src_map.items()
            }
    return serialized


def run_preprocessing(
    edges: pd.DataFrame,
    node_types: list[str],
    processed_dir: str | Path,
    make_undirected: bool = True,
) -> ProcessedArtifacts:
    """Encode data and save processed artifacts to disk."""
    output_dir = ensure_dir(processed_dir)

    encoded_edges, nodes_df, metadata = encode_nodes_and_edges(edges, node_types)
    adjacency = build_typed_adjacency(encoded_edges, make_undirected)
    metadata["typed_adjacency"] = adjacency

    encoded_edges_path = output_dir / "edges_encoded.csv"
    node_mapping_path = output_dir / "node_mapping.csv"
    metadata_path = output_dir / "graph_metadata.json"

    save_dataframe(encoded_edges, encoded_edges_path)
    save_dataframe(nodes_df, node_mapping_path)
    save_json(metadata, metadata_path)

    return ProcessedArtifacts(
        encoded_edges_path=encoded_edges_path,
        node_mapping_path=node_mapping_path,
        metadata_path=metadata_path,
    )
