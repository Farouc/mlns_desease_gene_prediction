"""Graph construction utilities for heterogeneous and homogeneous representations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from torch_geometric.data import HeteroData
except ImportError:  # pragma: no cover
    HeteroData = None

from src.utils.io import ensure_dir


@dataclass(frozen=True)
class GraphBuildResult:
    """Graph objects created from processed files."""

    nx_graph: nx.Graph
    hetero_data: Any | None


def sanitize_relation_name(relation: str) -> str:
    """Convert relation names into PyG-safe identifiers."""
    relation = relation.strip().lower()
    relation = re.sub(r"[^a-z0-9]+", "_", relation)
    return relation.strip("_") or "rel"


def build_networkx_graph(
    encoded_edges: pd.DataFrame,
    node_mapping: pd.DataFrame,
    undirected: bool,
) -> nx.Graph:
    """Build a homogeneous NetworkX graph with typed node attributes."""
    graph: nx.Graph | nx.DiGraph
    graph = nx.Graph() if undirected else nx.DiGraph()

    for node in node_mapping.itertuples(index=False):
        graph.add_node(
            int(node.global_id),
            node_type=str(node.node_type),
            local_id=int(node.local_id),
            raw_id=str(node.raw_id),
        )

    for edge in encoded_edges.itertuples(index=False):
        graph.add_edge(
            int(edge.src_global_id),
            int(edge.dst_global_id),
            edge_type=str(edge.edge_type),
            src_type=str(edge.src_type),
            dst_type=str(edge.dst_type),
        )

    return graph


def build_heterodata(
    encoded_edges: pd.DataFrame,
    node_mapping: pd.DataFrame,
    undirected: bool,
) -> Any:
    """Build PyG HeteroData graph when PyG is installed."""
    if HeteroData is None or torch is None:
        return None

    data = HeteroData()

    node_counts = (
        node_mapping.groupby("node_type")["local_id"].max().add(1).astype(int).to_dict()
        if not node_mapping.empty
        else {}
    )

    for node_type, count in node_counts.items():
        data[node_type].num_nodes = int(count)

    relation_to_edges: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for row in encoded_edges.itertuples(index=False):
        relation = sanitize_relation_name(str(row.edge_type))
        key = (str(row.src_type), relation, str(row.dst_type))
        relation_to_edges.setdefault(key, []).append(
            (int(row.src_local_id), int(row.dst_local_id))
        )

        if undirected:
            reverse_key = (str(row.dst_type), f"rev_{relation}", str(row.src_type))
            relation_to_edges.setdefault(reverse_key, []).append(
                (int(row.dst_local_id), int(row.src_local_id))
            )

    for relation, edge_pairs in relation_to_edges.items():
        src_nodes = [src for src, _ in edge_pairs]
        dst_nodes = [dst for _, dst in edge_pairs]
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        data[relation].edge_index = edge_index

    return data


def save_graph_artifacts(
    graph_result: GraphBuildResult,
    processed_dir: str | Path,
) -> dict[str, Path]:
    """Persist graph objects to disk."""
    out_dir = ensure_dir(processed_dir)
    nx_path = out_dir / "graph_nx.gpickle"
    nx.write_gpickle(graph_result.nx_graph, nx_path)

    artifacts: dict[str, Path] = {"networkx": nx_path}
    if graph_result.hetero_data is not None and torch is not None:
        pyg_path = out_dir / "graph_hetero.pt"
        torch.save(graph_result.hetero_data, pyg_path)
        artifacts["hetero"] = pyg_path
    return artifacts


def build_graphs_from_processed(
    encoded_edges_path: str | Path,
    node_mapping_path: str | Path,
    processed_dir: str | Path,
    undirected: bool,
) -> dict[str, Path]:
    """Load processed CSVs, build graph objects, and save serialized artifacts."""
    edges = pd.read_csv(encoded_edges_path)
    node_mapping = pd.read_csv(node_mapping_path)

    nx_graph = build_networkx_graph(edges, node_mapping, undirected)
    hetero_data = build_heterodata(edges, node_mapping, undirected)

    return save_graph_artifacts(
        GraphBuildResult(nx_graph=nx_graph, hetero_data=hetero_data),
        processed_dir=processed_dir,
    )
