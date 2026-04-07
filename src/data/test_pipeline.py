"""
Exploratory script to understand the pipeline step by step.
Run from the project root with: python explore.py
"""

import tempfile
import json
from pathlib import Path
import pandas as pd

from src.data.load_hetionet import load_hetionet_edges, infer_edge_columns
from src.data.preprocess import encode_nodes_and_edges, build_typed_adjacency, run_preprocessing
from src.data.split import extract_positive_disease_gene_pairs, create_splits


# ── 0. Create a fake raw CSV (mimics a real Hetionet edge file) ───────────────

tmp_dir = Path(tempfile.mkdtemp())
raw_csv = tmp_dir / "raw_edges.csv"

pd.DataFrame({
    "source":      ["D0", "D0", "D1", "D1", "D2", "D3", "D4", "D5", "G0", "G1"],
    "target":      ["G0", "G1", "G2", "G3", "G4", "G0", "G2", "G4", "G1", "G2"],
    "source_type": ["Disease"] * 8 + ["Gene"] * 2,
    "target_type": ["Gene"]    * 8 + ["Gene"] * 2,
    "metaedge":    ["DaG"]     * 8 + ["GiG"] * 2,
}).to_csv(raw_csv, index=False)

print("=== Raw CSV written to", raw_csv, "===")


# ── 1. Loader ─────────────────────────────────────────────────────────────────

print("\n=== infer_edge_columns: detects column names automatically ===")
sample_cols = ["source", "target", "source_type", "target_type", "metaedge"]
ec = infer_edge_columns(sample_cols)
print(f"  src_id={ec.src_id!r}, dst_id={ec.dst_id!r}, edge_type={ec.edge_type!r}")

print("\n=== load_hetionet_edges: standardize + filter to valid node types ===")
edges = load_hetionet_edges(raw_csv, valid_node_types=["Disease", "Gene"])
print(edges.to_string(index=False))
print(f"\nTotal edges loaded: {len(edges)}")
print("Note: all rows kept because both Disease and Gene are valid types.")


# ── 2. Preprocessing: encode nodes and edges ──────────────────────────────────

print("\n=== encode_nodes_and_edges: assign local + global integer IDs ===")
node_types = ["Disease", "Gene"]
encoded_edges, nodes_df, metadata = encode_nodes_and_edges(edges, node_types)

print("\n-- Node mapping (raw string ID -> local ID per type, global ID across all) --")
print(nodes_df.to_string(index=False))
print("\nNote: local_id resets to 0 for each node type.")
print("      global_id is unique across Disease and Gene combined.")

print("\n-- Encoded edges --")
print(encoded_edges.to_string(index=False))

print("\n-- Metadata --")
print(f"  Total nodes : {metadata['num_nodes_total']}")
print(f"  Total edges : {metadata['num_edges']}")
print(f"  Node counts : {metadata['node_counts']}")


# ── 3. Adjacency list ─────────────────────────────────────────────────────────

print("\n=== build_typed_adjacency: who is connected to whom ===")
adj = build_typed_adjacency(encoded_edges, make_undirected=True)

for src_type, dst_map in adj.items():
    for dst_type, src_map in dst_map.items():
        print(f"\n  {src_type} -> {dst_type}:")
        for local_id, neighbors in src_map.items():
            raw = nodes_df[
                (nodes_df["node_type"] == src_type) &
                (nodes_df["local_id"] == local_id)
            ]["raw_id"].values[0]
            neighbor_raws = nodes_df[
                (nodes_df["node_type"] == dst_type) &
                (nodes_df["local_id"].isin(neighbors))
            ]["raw_id"].tolist()
            print(f"    local_id={local_id} ({raw}) -> {neighbors} {neighbor_raws}")


# ── 4. run_preprocessing: save everything to disk ────────────────────────────

print("\n=== run_preprocessing: encodes + saves artifacts to disk ===")
artifacts = run_preprocessing(edges, node_types, processed_dir=tmp_dir / "processed")
print(f"  Encoded edges : {artifacts.encoded_edges_path}")
print(f"  Node mapping  : {artifacts.node_mapping_path}")
print(f"  Metadata      : {artifacts.metadata_path}")

with open(artifacts.metadata_path) as f:
    saved_meta = json.load(f)
print(f"\n  Metadata keys saved: {list(saved_meta.keys())}")


# ── 5. Splitting ──────────────────────────────────────────────────────────────

print("\n=== extract_positive_disease_gene_pairs: known Disease-Gene links ===")
positives = extract_positive_disease_gene_pairs(encoded_edges, "Disease", "Gene")
print(positives.to_string(index=False))
print(f"\nThese are the edges the model will learn to predict (label=1).")

print("\n=== create_splits: train/val/test with sampled negatives ===")
split_artifacts = create_splits(
    encoded_edges_path=artifacts.encoded_edges_path,
    node_mapping_path=artifacts.node_mapping_path,
    output_dir=tmp_dir / "splits",
    disease_type="Disease",
    gene_type="Gene",
    val_ratio=0.2,
    test_ratio=0.2,
    negative_ratio=1,
    seed=42,
)

for name, path in [("train", split_artifacts.train_path),
                   ("val",   split_artifacts.val_path),
                   ("test",  split_artifacts.test_path)]:
    df = pd.read_csv(path)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    print(f"\n  {name}.csv ({len(df)} rows, {n_pos} positive, {n_neg} negative):")
    print(df.to_string(index=False))

print("\nNegatives are randomly sampled Disease-Gene pairs that have no known association.")
print("The model's task: given a (disease, gene) pair, predict label 1 or 0.")