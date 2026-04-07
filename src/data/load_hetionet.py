"""Load Hetionet edges from CSV files into a standardized format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EdgeColumns:
    """Column names identified in the raw edge file."""

    src_id: str
    dst_id: str
    src_type: str
    dst_type: str
    edge_type: str


SOURCE_ID_CANDIDATES = ["source", "source_id", "src", "x_id", "start_id"]
TARGET_ID_CANDIDATES = ["target", "target_id", "dst", "y_id", "end_id"]
SOURCE_TYPE_CANDIDATES = ["source_type", "src_type", "x_type", "start_type"]
TARGET_TYPE_CANDIDATES = ["target_type", "dst_type", "y_type", "end_type"]
EDGE_TYPE_CANDIDATES = ["edge_type", "relation", "metaedge", "type"]


class HetionetLoaderError(RuntimeError):
    """Raised when loading Hetionet data fails."""


def _pick_column(columns: list[str], candidates: list[str], role: str) -> str:
    """Pick the first matching column from candidates."""
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise HetionetLoaderError(
        f"Could not infer column for {role}. Available columns: {columns}"
    )


def infer_edge_columns(columns: list[str]) -> EdgeColumns:
    """Infer edge schema from raw CSV columns.

    Args:
        columns: Column names in the CSV file.

    Returns:
        Normalized edge column mapping.
    """
    return EdgeColumns(
        src_id=_pick_column(columns, SOURCE_ID_CANDIDATES, "source node id"),
        dst_id=_pick_column(columns, TARGET_ID_CANDIDATES, "target node id"),
        src_type=_pick_column(columns, SOURCE_TYPE_CANDIDATES, "source node type"),
        dst_type=_pick_column(columns, TARGET_TYPE_CANDIDATES, "target node type"),
        edge_type=_pick_column(columns, EDGE_TYPE_CANDIDATES, "edge type"),
    )


def load_hetionet_edges(
    edge_csv_path: str | Path,
    valid_node_types: list[str],
) -> pd.DataFrame:
    """Load and standardize Hetionet edges.

    Args:
        edge_csv_path: Path to raw edge CSV.
        valid_node_types: Node types to keep.

    Returns:
        DataFrame with standardized columns:
        src_raw_id, dst_raw_id, src_type, dst_type, edge_type.
    """
    path = Path(edge_csv_path)
    if not path.exists():
        raise HetionetLoaderError(f"Raw edge file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise HetionetLoaderError(f"Raw edge file is empty: {path}")

    inferred = infer_edge_columns(list(df.columns))
    standardized = pd.DataFrame(
        {
            "src_raw_id": df[inferred.src_id].astype(str),
            "dst_raw_id": df[inferred.dst_id].astype(str),
            "src_type": df[inferred.src_type].astype(str),
            "dst_type": df[inferred.dst_type].astype(str),
            "edge_type": df[inferred.edge_type].astype(str),
        }
    )

    valid_set = set(valid_node_types)
    filtered = standardized[
        standardized["src_type"].isin(valid_set) & standardized["dst_type"].isin(valid_set)
    ].copy()

    filtered.drop_duplicates(inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered
