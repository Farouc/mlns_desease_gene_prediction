"""Interpretability-focused visualizations for hybrid predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.utils import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_RUN_DIR,
    apply_publication_style,
    load_json,
    load_model_predictions,
    load_model_ranked_predictions,
    save_figure,
)

METAPATH_ORDER = ["DaGpPpG", "DpPhG", "GcGaD"]


def _load_metapath_weights(result_dir: Path) -> dict[str, float]:
    """Load metapath weights from saved hybrid artifacts."""
    weight_path = result_dir / "metapath_weights.json"
    if not weight_path.exists():
        return {name: 1.0 for name in METAPATH_ORDER}

    raw = load_json(weight_path)
    return {name: float(raw.get(name, 1.0)) for name in METAPATH_ORDER}


def _load_top_prediction_pairs(result_dir: Path) -> set[tuple[int, int]]:
    """Load (disease_local_id, gene_local_id) pairs from top prediction explanations."""
    explain_path = result_dir / "interpretability_top_predictions.json"
    if not explain_path.exists():
        return set()

    items = load_json(explain_path)
    pairs: set[tuple[int, int]] = set()
    if not isinstance(items, list):
        return pairs

    for row in items:
        try:
            disease = int(row["disease"])
            gene = int(row["gene"])
        except (KeyError, TypeError, ValueError):
            continue
        pairs.add((disease, gene))
    return pairs


def generate_metapath_contributions_plot(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Plot average weighted metapath contribution on top explained predictions."""
    apply_publication_style()
    result_path = Path(result_dir)

    counts_path = result_path / "metapath_counts_test_long.csv"
    if not counts_path.exists():
        raise RuntimeError(f"Missing required file: {counts_path}")

    counts_df = pd.read_csv(counts_path)
    expected_cols = {"disease_local_id", "gene_local_id", "metapath", "count"}
    missing = expected_cols.difference(counts_df.columns)
    if missing:
        raise RuntimeError(f"metapath_counts_test_long.csv missing columns: {sorted(missing)}")

    top_pairs = _load_top_prediction_pairs(result_path)
    if top_pairs:
        pair_index = pd.MultiIndex.from_frame(counts_df[["disease_local_id", "gene_local_id"]])
        top_index = pd.MultiIndex.from_tuples(
            sorted(top_pairs),
            names=["disease_local_id", "gene_local_id"],
        )
        counts_df = counts_df.loc[pair_index.isin(top_index)].copy()

    if counts_df.empty:
        raise RuntimeError("No rows available to compute metapath contributions.")

    weights = _load_metapath_weights(result_path)
    counts_df["weight"] = counts_df["metapath"].map(weights).fillna(1.0).astype(float)
    counts_df["contribution"] = counts_df["count"].astype(float) * counts_df["weight"]

    contribution = (
        counts_df.groupby("metapath", as_index=True)["contribution"]
        .mean()
        .reindex(METAPATH_ORDER, fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    bars = ax.bar(
        METAPATH_ORDER,
        contribution.values.astype(float),
        color=["#4E79A7", "#F28E2B", "#59A14F"],
        edgecolor="black",
        linewidth=0.6,
    )

    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Metapath")
    ax.set_ylabel("Average weighted contribution")
    ax.set_title("Average Metapath Contribution (Top Predictions)")

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="metapath_contributions.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved


def generate_performance_vs_interpretability_plot(
    result_dir: str | Path = DEFAULT_RUN_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
    overwrite: bool = False,
    save_pdf: bool = True,
) -> bool:
    """Scatter plot: interpretability ratio vs rank quality on hybrid predictions."""
    apply_publication_style()
    result_path = Path(result_dir)

    hybrid_df = load_model_predictions(result_path, model="hybrid")
    required_cols = {"disease_local_id", "gene_local_id", "label", "score_path", "score_hybrid"}
    missing = required_cols.difference(hybrid_df.columns)
    if missing:
        raise RuntimeError(f"hybrid_predictions.csv missing columns: {sorted(missing)}")

    ranked_path = result_path / "hybrid_ranked_predictions.csv"
    if ranked_path.exists():
        ranked_df = load_model_ranked_predictions(result_path, model="hybrid")
        rank_cols = ["disease_local_id", "gene_local_id", "rank"]
        if set(rank_cols).issubset(ranked_df.columns):
            hybrid_df = hybrid_df.merge(ranked_df[rank_cols], on=["disease_local_id", "gene_local_id"], how="left")
        else:
            hybrid_df["rank"] = np.nan
    else:
        hybrid_df["rank"] = np.nan

    s_path = hybrid_df["score_path"].astype(float).to_numpy()
    s_final = hybrid_df["score_hybrid"].astype(float).to_numpy()

    interpretability_score = np.divide(
        s_path,
        s_final,
        out=np.full_like(s_path, np.nan, dtype=float),
        where=np.abs(s_final) > 1e-12,
    )
    hybrid_df["interpretability_score"] = interpretability_score

    plot_df = hybrid_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["interpretability_score", "rank", "label"]
    )
    if plot_df.empty:
        raise RuntimeError("No valid rows found for performance vs interpretability scatter plot.")

    low_q, high_q = np.nanquantile(plot_df["interpretability_score"], [0.01, 0.99])
    plot_df["interpretability_score"] = plot_df["interpretability_score"].clip(lower=low_q, upper=high_q)

    negatives = plot_df.loc[plot_df["label"].astype(int) == 0]
    positives = plot_df.loc[plot_df["label"].astype(int) == 1]

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    if not negatives.empty:
        ax.scatter(
            negatives["interpretability_score"],
            negatives["rank"],
            s=12,
            alpha=0.25,
            color="#4E79A7",
            label="Negative edge",
            rasterized=True,
        )
    if not positives.empty:
        ax.scatter(
            positives["interpretability_score"],
            positives["rank"],
            s=16,
            alpha=0.55,
            color="#E15759",
            label="Positive edge",
            rasterized=True,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Interpretability score (s_path / s_final)")
    ax.set_ylabel("Per-disease rank (log scale, lower is better)")
    ax.set_title("Performance vs Interpretability (Hybrid)")
    ax.legend(loc="upper right", frameon=False)

    saved = save_figure(
        fig,
        figure_dir=figure_dir,
        filename="performance_vs_interpretability.png",
        overwrite=overwrite,
        save_pdf=save_pdf,
    )
    plt.close(fig)
    return saved
