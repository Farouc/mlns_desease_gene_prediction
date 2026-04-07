"""Generate interpretability outputs for top disease-gene predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_json


# ── Style ─────────────────────────────────────────────────────────────────────

METAPATH_COLORS = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#16A34A",  # green
    "#9333EA",  # purple
    "#EA580C",  # orange
]


def _apply_elegant_style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#CCCCCC", linewidth=0.6, linestyle="-")
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.3, linestyle="-")
    ax.minorticks_on()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("#888888")
    ax.tick_params(axis="both", labelsize=11, length=3, width=0.7)
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")


# ── Name mapping ──────────────────────────────────────────────────────────────

def build_name_lookup(
    node_mapping: pd.DataFrame,
    hetionet_nodes_path: str | Path | None,
) -> dict[tuple[str, int], str]:
    """Build a lookup from (node_type, local_id) -> human-readable name.

    Falls back to raw_id if hetionet nodes file is not available.

    Args:
        node_mapping: Processed node mapping dataframe.
        hetionet_nodes_path: Path to hetionet-v1.0-nodes.tsv.

    Returns:
        Dictionary mapping (node_type, local_id) to name string.
    """
    lookup: dict[tuple[str, int], str] = {}
    for row in node_mapping.itertuples(index=False):
        lookup[(str(row.node_type), int(row.local_id))] = str(row.raw_id)

    if hetionet_nodes_path is None:
        return lookup

    path = Path(hetionet_nodes_path)
    if not path.exists():
        return lookup

    nodes_df = pd.read_csv(path, sep="\t")
    raw_id_to_name: dict[str, str] = {}
    for row in nodes_df.itertuples(index=False):
        raw_id = str(row.id).split("::", 1)[-1]
        raw_id_to_name[raw_id] = str(row.name)

    for row in node_mapping.itertuples(index=False):
        name = raw_id_to_name.get(str(row.raw_id), str(row.raw_id))
        lookup[(str(row.node_type), int(row.local_id))] = name

    return lookup


# ── Core helpers ──────────────────────────────────────────────────────────────

def _counts_by_pair(
    count_df: pd.DataFrame,
    disease_col: str,
    gene_col: str,
) -> dict[tuple[int, int], dict[str, int]]:
    grouped: dict[tuple[int, int], dict[str, int]] = {}
    for row in count_df.itertuples(index=False):
        key = (int(getattr(row, disease_col)), int(getattr(row, gene_col)))
        grouped.setdefault(key, {})[str(row.metapath)] = int(row.count)
    return grouped


def _weighted_contributions(
    counts: dict[str, int],
    metapath_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Compute weighted contribution of each metapath for a pair.
    Fallback used when standardized contrib_ columns are not available.
    """
    if not metapath_weights:
        return {name: float(count) for name, count in counts.items()}
    return {
        name: count * float(metapath_weights.get(name, 0.0))
        for name, count in counts.items()
    }


# ── Main explanation builder ──────────────────────────────────────────────────

def build_explanations(
    predictions: pd.DataFrame,
    metapath_count_df: pd.DataFrame,
    score_col: str,
    metapath_weights: dict[str, float] | None,
    top_n: int,
    disease_col: str = "disease_local_id",
    gene_col: str = "gene_local_id",
    name_lookup: dict[tuple[str, int], str] | None = None,
    disease_type: str = "Disease",
    gene_type: str = "Gene",
) -> list[dict[str, Any]]:
    """Build rich explainability records for top scoring predictions.

    Uses standardized contrib_ columns (scaled count × logistic coefficient)
    if present in predictions, otherwise falls back to raw count × weight.

    Each record includes:
    - Human-readable disease and gene names
    - Hybrid score, GNN score, path score (if available)
    - Raw metapath counts
    - Standardized weighted contributions (what logistic regression actually used)
    - Top contributing metapath
    - Rank among all predictions
    - Ground truth label
    """
    top_predictions = predictions.sort_values(score_col, ascending=False).head(top_n).copy()
    top_predictions["rank"] = range(1, len(top_predictions) + 1)
    counts_lookup = _counts_by_pair(metapath_count_df, disease_col=disease_col, gene_col=gene_col)

    # Detect standardized contribution columns added by main2.py
    # These are named contrib_<metapath_name> and reflect scaled count × coefficient
    contrib_cols = [
        col.replace("contrib_", "")
        for col in predictions.columns
        if col.startswith("contrib_")
    ]

    explanations: list[dict[str, Any]] = []
    for row in top_predictions.itertuples(index=False):
        disease_id = int(getattr(row, disease_col))
        gene_id = int(getattr(row, gene_col))
        score = float(getattr(row, score_col))
        label = int(getattr(row, "label")) if hasattr(row, "label") else None
        rank = int(getattr(row, "rank"))

        # Human-readable names
        if name_lookup:
            disease_name = name_lookup.get((disease_type, disease_id), str(disease_id))
            gene_name = name_lookup.get((gene_type, gene_id), str(gene_id))
        else:
            disease_name = str(disease_id)
            gene_name = str(gene_id)

        counts = counts_lookup.get((disease_id, gene_id), {})

        # Use standardized contributions if available, else fall back
        if contrib_cols:
            contributions = {
                name: float(getattr(row, f"contrib_{name}", 0.0))
                for name in contrib_cols
            }
        else:
            contributions = _weighted_contributions(counts, metapath_weights)

        top_metapath = max(contributions, key=contributions.get) if contributions else "none"
        total_path_evidence = sum(counts.values())

        gnn_score = (
            float(getattr(row, "score_han")) if hasattr(row, "score_han") else
            float(getattr(row, "score_node2vec")) if hasattr(row, "score_node2vec") else
            None
        )
        path_score = float(getattr(row, "score_path")) if hasattr(row, "score_path") else None
        empirical_confidence = (
            float(getattr(row, "empirical_confidence"))
            if hasattr(row, "empirical_confidence")
            else None
        )
        score_quantile_train = (
            float(getattr(row, "score_quantile_train"))
            if hasattr(row, "score_quantile_train")
            else None
        )

        record: dict[str, Any] = {
            "rank": rank,
            "disease_id": disease_id,
            "disease_name": disease_name,
            "gene_id": gene_id,
            "gene_name": gene_name,
            "hybrid_score": round(score, 4),
            "label": label,
            "is_known_link": bool(label == 1) if label is not None else None,
            "top_metapath": top_metapath,
            "total_path_evidence": total_path_evidence,
            "path_counts": counts,
            "weighted_contributions": {k: round(v, 4) for k, v in contributions.items()},
        }
        if gnn_score is not None:
            record["gnn_score"] = round(gnn_score, 4)
        if path_score is not None:
            record["path_score"] = round(path_score, 4)
        if empirical_confidence is not None:
            record["empirical_confidence"] = round(empirical_confidence, 4)
        if score_quantile_train is not None:
            record["score_quantile_train"] = round(score_quantile_train, 4)
        if metapath_weights:
            record["metapath_weights_used"] = {k: round(v, 4) for k, v in metapath_weights.items()}

        explanations.append(record)

    return explanations


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(
    explanations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert explanations list into a clean flat summary dataframe."""
    rows = []
    for exp in explanations:
        row: dict[str, Any] = {
            "rank":                exp["rank"],
            "disease_name":        exp["disease_name"],
            "gene_name":           exp["gene_name"],
            "hybrid_score":        exp["hybrid_score"],
            "is_known_link":       exp["is_known_link"],
            "top_metapath":        exp["top_metapath"],
            "total_path_evidence": exp["total_path_evidence"],
        }
        if "gnn_score" in exp:
            row["gnn_score"] = exp["gnn_score"]
        if "path_score" in exp:
            row["path_score"] = exp["path_score"]
        if "empirical_confidence" in exp:
            row["empirical_confidence"] = exp["empirical_confidence"]
        if "score_quantile_train" in exp:
            row["score_quantile_train"] = exp["score_quantile_train"]
        for name, count in exp["path_counts"].items():
            row[f"count_{name}"] = count
        for name, contrib in exp.get("weighted_contributions", {}).items():
            row[f"contrib_{name}"] = contrib
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_top_predictions_metapaths(
    explanations: list[dict[str, Any]],
    output_dir: str | Path,
    top_n_plot: int = 20,
) -> Path:
    """Bar chart showing standardized metapath contributions for top N predictions."""
    out_dir = ensure_dir(output_dir)
    subset = explanations[:top_n_plot]

    all_metapaths = []
    for exp in subset:
        for name in exp["weighted_contributions"]:
            if name not in all_metapaths:
                all_metapaths.append(name)

    labels = [
        f"#{exp['rank']}\n{exp['disease_name'][:18]}\n→ {exp['gene_name'][:12]}"
        for exp in subset
    ]

    x = np.arange(len(subset))
    n_metapaths = len(all_metapaths)
    width = 0.8 / max(n_metapaths, 1)

    fig, ax = plt.subplots(figsize=(max(12, len(subset) * 0.9), 5.5))

    for i, metapath in enumerate(all_metapaths):
        values = [exp["weighted_contributions"].get(metapath, 0.0) for exp in subset]
        offset = (i - n_metapaths / 2 + 0.5) * width
        ax.bar(
            x + offset, values,
            width=width * 0.9,
            color=METAPATH_COLORS[i % len(METAPATH_COLORS)],
            label=metapath,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center", rotation=30)
    ax.set_ylabel("Standardized Metapath Contribution\n(scaled count × logistic coeff)", fontsize=11, labelpad=8)
    ax.set_title("Metapath Evidence for Top Predictions", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#CCCCCC", loc="upper right")
    _apply_elegant_style(ax)
    fig.tight_layout()

    path = out_dir / "top_predictions_metapath_contributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_breakdown(
    explanations: list[dict[str, Any]],
    output_dir: str | Path,
    top_n_plot: int = 20,
) -> Path | None:
    """Stacked bar showing GNN vs path score for top predictions."""
    out_dir = ensure_dir(output_dir)
    subset = [e for e in explanations[:top_n_plot] if "gnn_score" in e and "path_score" in e]
    if not subset:
        return None

    labels = [
        f"#{e['rank']} {e['disease_name'][:15]}→{e['gene_name'][:10]}"
        for e in subset
    ]
    gnn_scores  = [e["gnn_score"] for e in subset]
    path_scores = [e["path_score"] for e in subset]
    x = np.arange(len(subset))

    fig, ax = plt.subplots(figsize=(max(10, len(subset) * 0.8), 5))
    ax.bar(x, gnn_scores,  label="GNN score (HAN)",  color="#2563EB", edgecolor="white", zorder=3)
    ax.bar(x, path_scores, label="Path score (logistic)", color="#16A34A", edgecolor="white",
           bottom=gnn_scores, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Score (probability)", fontsize=12, labelpad=8)
    ax.set_title("GNN vs Metapath Score Breakdown — Top Predictions",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    _apply_elegant_style(ax)
    fig.tight_layout()

    path = out_dir / "top_predictions_score_breakdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metapath_weight_importance(
    metapath_weights: dict[str, float],
    output_dir: str | Path,
) -> Path:
    """Horizontal bar chart showing learned logistic regression coefficients."""
    out_dir = ensure_dir(output_dir)

    names  = list(metapath_weights.keys())
    values = [metapath_weights[n] for n in names]
    colors = [METAPATH_COLORS[i % len(METAPATH_COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.8)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            fontsize=11, color="#333333",
        )

    ax.set_xlabel("Logistic Regression Coefficient (standardized)", fontsize=12, labelpad=8)
    ax.set_title("Metapath Importance", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
    _apply_elegant_style(ax)
    fig.tight_layout()

    path = out_dir / "metapath_weight_importance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Save ──────────────────────────────────────────────────────────────────────

def save_explanations(explanations: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save explanations to JSON."""
    save_json(explanations, output_path)


def save_all_interpretability_outputs(
    explanations: list[dict[str, Any]],
    metapath_weights: dict[str, float] | None,
    result_dir: str | Path,
    figure_dir: str | Path,
    top_n_plot: int = 20,
) -> None:
    """Save all interpretability outputs: JSON, CSV, and plots."""
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)

    save_explanations(explanations, result_dir / "interpretability_top_predictions.json")

    summary_df = build_summary_table(explanations)
    summary_df.to_csv(result_dir / "interpretability_summary.csv", index=False)

    plot_top_predictions_metapaths(explanations, figure_dir, top_n_plot=top_n_plot)
    plot_score_breakdown(explanations, figure_dir, top_n_plot=top_n_plot)

    if metapath_weights:
        plot_metapath_weight_importance(metapath_weights, figure_dir)
