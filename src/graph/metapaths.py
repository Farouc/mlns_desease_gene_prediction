"""Metapath counting with cycle-safe traversal and caching."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


DEFAULT_METAPATHS: dict[str, list[str]] = {
    "DpSpDaG": ["Disease", "Symptom", "Disease", "Gene"],
    "DaGpPWpG": ["Disease", "Gene", "Pathway", "Gene"],
    "GpPWpGaD": ["Gene", "Pathway", "Gene", "Disease"],
}


@dataclass(frozen=True)
class CountResult:
    """Metapath count for one disease-gene pair."""

    disease_local_id: int
    gene_local_id: int
    metapath: str
    count: int


class LRUCache:
    """Lightweight LRU cache for pair/metapath counting."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[tuple[Any, ...], int] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> int | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: tuple[Any, ...], value: int) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


class MetapathCounter:
    """Count metapath instances between typed node pairs."""

    def __init__(
        self,
        adjacency: dict[str, dict[str, dict[int, list[int]]]],
        node_counts: dict[str, int] | None = None,
        cache_size: int = 100_000,
    ) -> None:
        self.adjacency = adjacency
        self.node_counts = node_counts or {}
        self.count_cache = LRUCache(max_size=cache_size)
        self.neighbor_cache: dict[tuple[str, str, int], tuple[int, ...]] = {}
        self.adj_matrix_cache: dict[tuple[str, str], Any] = {}
        self.metapath_matrix_cache: dict[tuple[str, ...], Any] = {}

    def _type_size(self, node_type: str) -> int:
        """Infer number of nodes for a node type."""
        if node_type in self.node_counts:
            return int(self.node_counts[node_type])

        max_id = -1
        # IDs can appear as keys (source side) and values (target side).
        for dst_map in self.adjacency.get(node_type, {}).values():
            if dst_map:
                max_id = max(max_id, max(int(k) for k in dst_map.keys()))
            for dst_ids in dst_map.values():
                if dst_ids:
                    max_id = max(max_id, max(int(v) for v in dst_ids))
        return max_id + 1 if max_id >= 0 else 0

    def _adjacency_matrix(self, src_type: str, dst_type: str):
        """Build sparse adjacency matrix for one typed relation."""
        key = (src_type, dst_type)
        if key in self.adj_matrix_cache:
            return self.adj_matrix_cache[key]

        if sp is None:
            return None

        src_map = self.adjacency.get(src_type, {}).get(dst_type, {})
        n_src = self._type_size(src_type)
        n_dst = self._type_size(dst_type)
        if n_src <= 0 or n_dst <= 0:
            mat = sp.csr_matrix((0, 0), dtype="float32")
            self.adj_matrix_cache[key] = mat
            return mat

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for src_id, dst_ids in src_map.items():
            s = int(src_id)
            for d in dst_ids:
                rows.append(s)
                cols.append(int(d))
                data.append(1.0)

        mat = sp.csr_matrix((data, (rows, cols)), shape=(n_src, n_dst), dtype="float32")
        self.adj_matrix_cache[key] = mat
        return mat

    def _metapath_count_matrix(self, metapath_types: list[str]):
        """Compute sparse count matrix for a full metapath."""
        if sp is None or len(metapath_types) < 2:
            return None

        key = tuple(metapath_types)
        if key in self.metapath_matrix_cache:
            return self.metapath_matrix_cache[key]

        matrices = [
            self._adjacency_matrix(metapath_types[i], metapath_types[i + 1])
            for i in range(len(metapath_types) - 1)
        ]
        if any(mat is None for mat in matrices):
            return None

        # Right-associative multiplication tends to avoid massive intermediates
        # for metapaths like G->P->G->D.
        result = matrices[-1]
        for mat in reversed(matrices[:-1]):
            result = mat @ result
        result = result.tocsr()

        self.metapath_matrix_cache[key] = result
        return result

    def _neighbors(self, src_type: str, dst_type: str, src_id: int) -> tuple[int, ...]:
        key = (src_type, dst_type, src_id)
        if key in self.neighbor_cache:
            return self.neighbor_cache[key]

        neighbors = tuple(
            self.adjacency.get(src_type, {}).get(dst_type, {}).get(src_id, [])
        )
        self.neighbor_cache[key] = neighbors
        return neighbors

    def count_paths(
        self,
        start_id: int,
        end_id: int,
        metapath_types: list[str],
        avoid_cycles: bool = True,
    ) -> int:
        """Count path instances matching a typed metapath.

        Args:
            start_id: Local node id for the first node type.
            end_id: Local node id for the final node type.
            metapath_types: Ordered sequence of node types.
            avoid_cycles: If True, disallow revisiting typed nodes.

        Returns:
            Number of valid typed paths.
        """
        cache_key = (start_id, end_id, tuple(metapath_types), avoid_cycles)
        cached = self.count_cache.get(cache_key)
        if cached is not None:
            return cached

        if len(metapath_types) < 2:
            return 0

        def dfs(
            step: int,
            current_id: int,
            visited: set[tuple[str, int]],
        ) -> int:
            current_type = metapath_types[step]
            if step == len(metapath_types) - 1:
                return int(current_id == end_id)

            next_type = metapath_types[step + 1]
            total = 0
            for next_id in self._neighbors(current_type, next_type, current_id):
                typed_next = (next_type, int(next_id))
                if avoid_cycles and typed_next in visited:
                    continue
                next_visited = visited if not avoid_cycles else visited | {typed_next}
                total += dfs(step + 1, int(next_id), next_visited)
            return total

        initial_visited = (
                {(metapath_types[0], start_id), (metapath_types[-1], end_id)}
                if avoid_cycles else set()
            )
        count = dfs(step=0, current_id=start_id, visited=initial_visited)
        self.count_cache.set(cache_key, count)
        return count

    def count_for_pairs(
        self,
        pairs: pd.DataFrame,
        metapaths: dict[str, list[str]],
        disease_col: str = "disease_local_id",
        gene_col: str = "gene_local_id",
    ) -> pd.DataFrame:
        """Compute all metapath counts for each pair in a dataframe."""
        pair_records = [
            (
                int(getattr(row, disease_col)),
                int(getattr(row, gene_col)),
            )
            for row in pairs.itertuples(index=False)
        ]

        records: list[dict[str, Any]] = []
        for name, types in metapaths.items():
            count_matrix = self._metapath_count_matrix(types)
            use_matrix = count_matrix is not None and sp is not None

            for disease_id, gene_id in pair_records:
                if types[0] != "Disease" or types[-1] != "Gene":
                    if types[0] == "Gene" and types[-1] == "Disease":
                        if use_matrix:
                            count = int(count_matrix[gene_id, disease_id])
                        else:
                            count = self.count_paths(gene_id, disease_id, types)
                    else:
                        count = 0
                else:
                    if use_matrix:
                        count = int(count_matrix[disease_id, gene_id])
                    else:
                        count = self.count_paths(disease_id, gene_id, types)

                records.append(
                    {
                        "disease_local_id": disease_id,
                        "gene_local_id": gene_id,
                        "metapath": name,
                        "count": int(count),
                    }
                )
        return pd.DataFrame(records)


def pivot_metapath_counts(count_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format metapath counts into feature columns."""
    if count_df.empty:
        return pd.DataFrame(
            columns=["disease_local_id", "gene_local_id", *list(DEFAULT_METAPATHS.keys())]
        )

    pivoted = (
        count_df.pivot_table(
            index=["disease_local_id", "gene_local_id"],
            columns="metapath",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivoted
