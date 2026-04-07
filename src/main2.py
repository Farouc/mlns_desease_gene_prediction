"""Main entrypoint for interpretable disease-gene prediction experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Any
import sys

import networkx as nx
import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.load_hetionet import load_hetionet_edges
from src.data.preprocess import run_preprocessing
from src.data.split import create_splits
from src.evaluation.evaluator import (
    evaluate_predictions,
    plot_ablation_bars,
    plot_alpha_performance,
    plot_roc_pr_curves,
    save_metrics,
)
from src.evaluation.metrics import compute_auc_pr
from src.graph.build_graph import build_graphs_from_processed
from src.graph.metapaths import DEFAULT_METAPATHS, MetapathCounter, pivot_metapath_counts
from src.interpretability.explain import (
    build_name_lookup,
    build_explanations,
    save_all_interpretability_outputs,
)
from src.interpretability.path_extraction import compute_metapath_counts_for_predictions
from src.models.heuristics import score_pairs_with_heuristics
from src.models.hybrid_model import HybridModel, grid_search_alpha
from src.training.train_han import run_han_training
from src.training.train_node2vec import run_node2vec_training
from src.training.trainer_utils import infer_device, load_split_dataframe
from src.utils.config import deep_update, load_yaml_config, parse_overrides, resolve_path
from src.utils.io import copy_file, create_run_artifact_dirs, ensure_dir, load_json, save_dataframe, save_json
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interpretable Disease-Gene Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Optional key=value overrides (dot notation)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run folder name")
    return parser.parse_args()


def deserialize_typed_adjacency(
    raw: dict[str, dict[str, dict[str, list[int]]]],
) -> dict[str, dict[str, dict[int, list[int]]]]:
    """Convert JSON-loaded adjacency keys back to integer node IDs."""
    restored: dict[str, dict[str, dict[int, list[int]]]] = {}
    for src_type, dst_map in raw.items():
        restored[src_type] = {}
        for dst_type, src_nodes in dst_map.items():
            restored[src_type][dst_type] = {
                int(src_id): [int(dst_id) for dst_id in dst_ids]
                for src_id, dst_ids in src_nodes.items()
            }
    return restored


def save_effective_config(config: dict[str, Any], path: str | Path) -> None:
    """Save final resolved config as YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def evaluate_and_persist(
    predictions: pd.DataFrame,
    score_col: str,
    model_name: str,
    result_dir: Path,
    figure_dir: Path,
    top_k: int,
    disease_col: str,
    gene_col: str,
) -> dict[str, float]:
    """Evaluate one model, save predictions/metrics/plots, and return metrics."""
    eval_result = evaluate_predictions(
        predictions=predictions,
        score_col=score_col,
        label_col="label",
        disease_col=disease_col,
        gene_col=gene_col,
        top_k=top_k,
    )

    pred_path = result_dir / f"{model_name}_predictions.csv"
    metrics_path = result_dir / f"{model_name}_metrics.json"
    ranked_path = result_dir / f"{model_name}_ranked_predictions.csv"

    save_dataframe(predictions, pred_path)
    save_dataframe(eval_result.ranked_predictions, ranked_path)
    save_metrics(eval_result.metrics, metrics_path)

    plot_roc_pr_curves(
        predictions=predictions,
        score_col=score_col,
        label_col="label",
        output_dir=figure_dir,
        prefix=model_name,
    )
    return eval_result.metrics


def filter_encoded_edges_to_train(
    encoded_edges: pd.DataFrame,
    train_df: pd.DataFrame,
    disease_type: str,
    gene_type: str,
) -> pd.DataFrame:
    """Remove val/test positive Disease-Gene edges from the encoded edge set.

    All other edge types are kept as-is since they carry no leakage.
    Only Disease-Gene edges not in the train split are removed.
    """
    train_positives = train_df[train_df["label"] == 1][
        ["disease_local_id", "gene_local_id"]
    ].astype(int).drop_duplicates()

    is_dg_forward = (
        (encoded_edges["src_type"] == disease_type) &
        (encoded_edges["dst_type"] == gene_type)
    )
    is_dg_reverse = (
        (encoded_edges["src_type"] == gene_type) &
        (encoded_edges["dst_type"] == disease_type)
    )
    is_dg_edge = is_dg_forward | is_dg_reverse

    forward_edges = encoded_edges[is_dg_forward].copy()
    forward_edges["src_local_id"] = forward_edges["src_local_id"].astype(int)
    forward_edges["dst_local_id"] = forward_edges["dst_local_id"].astype(int)
    forward_kept = forward_edges.merge(
        train_positives,
        left_on=["src_local_id", "dst_local_id"],
        right_on=["disease_local_id", "gene_local_id"],
        how="inner",
    ).drop(columns=["disease_local_id", "gene_local_id"])

    reverse_edges = encoded_edges[is_dg_reverse].copy()
    reverse_edges["src_local_id"] = reverse_edges["src_local_id"].astype(int)
    reverse_edges["dst_local_id"] = reverse_edges["dst_local_id"].astype(int)
    reverse_kept = reverse_edges.merge(
        train_positives,
        left_on=["dst_local_id", "src_local_id"],
        right_on=["disease_local_id", "gene_local_id"],
        how="inner",
    ).drop(columns=["disease_local_id", "gene_local_id"])

    non_dg = encoded_edges[~is_dg_edge]

    result = pd.concat([non_dg, forward_kept, reverse_kept], ignore_index=True)
    result.reset_index(drop=True, inplace=True)
    return result


def run_pipeline(config: dict[str, Any], config_path: str | Path, run_name: str | None) -> None:
    """Run the complete experiment pipeline."""
    project_root = Path(config.get("project_root", ".")).resolve()

    seed = int(config["seed"])
    seed_everything(seed)

    experiment_cfg = config["experiment"]
    run_dirs = create_run_artifact_dirs(
        results_root=resolve_path(project_root, experiment_cfg["results_root"]),
        logs_root=resolve_path(project_root, experiment_cfg["logs_root"]),
        figures_root=resolve_path(project_root, experiment_cfg["figures_root"]),
        run_name=run_name,
    )

    result_dir = Path(run_dirs["result_dir"]).resolve()
    log_dir = Path(run_dirs["log_dir"]).resolve()
    figure_dir = Path(run_dirs["figure_dir"]).resolve()
    weights_dir = ensure_dir(result_dir / "weights")

    logger = setup_logger(name="hetionet_pipeline", log_file=log_dir / "run.log")
    save_effective_config(config, result_dir / "config_resolved.yaml")
    copy_file(config_path, result_dir / "config_input.yaml")

    logger.info("Run directory: %s", result_dir)

    data_cfg = config["data"]
    node_types = data_cfg["node_types"]
    disease_type = data_cfg["disease_type"]
    gene_type = data_cfg["gene_type"]

    # ── STEP 1: Load and preprocess raw edges ────────────────────────────────

    raw_edge_path = resolve_path(
        project_root,
        Path(data_cfg["raw_dir"]) / data_cfg["edges_file"],
    )

    logger.info("Loading raw edges from %s", raw_edge_path)
    edges_df = load_hetionet_edges(raw_edge_path, valid_node_types=node_types)

    processed_artifacts = run_preprocessing(
        edges=edges_df,
        node_types=node_types,
        processed_dir=resolve_path(project_root, data_cfg["processed_dir"]),
        make_undirected=bool(config["graph"]["undirected"]),
    )

    # Build name lookup from hetionet nodes file for interpretability
    node_mapping_df = pd.read_csv(processed_artifacts.node_mapping_path)
    name_lookup = build_name_lookup(
        node_mapping=node_mapping_df,
        hetionet_nodes_path=resolve_path(project_root, "data/raw/hetionet_nodes.tsv"),
    )

    # ── STEP 2: Split BEFORE building the graph ───────────────────────────────

    logger.info("Creating train/val/test splits")
    split_cfg = data_cfg["split"]
    split_artifacts = create_splits(
        encoded_edges_path=processed_artifacts.encoded_edges_path,
        node_mapping_path=processed_artifacts.node_mapping_path,
        output_dir=resolve_path(project_root, data_cfg["splits_dir"]),
        disease_type=disease_type,
        gene_type=gene_type,
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        negative_ratio=int(split_cfg["negative_ratio"]),
        seed=seed,
    )

    # ── STEP 3: Build graph from TRAIN edges only ─────────────────────────────

    logger.info("Building train-only graph (removing val/test positive edges)")
    train_df = load_split_dataframe(split_artifacts.train_path)
    full_encoded_edges = pd.read_csv(processed_artifacts.encoded_edges_path, low_memory=False)

    train_only_edges = filter_encoded_edges_to_train(
        encoded_edges=full_encoded_edges,
        train_df=train_df,
        disease_type=disease_type,
        gene_type=gene_type,
    )

    processed_dir = resolve_path(project_root, data_cfg["processed_dir"])
    train_edges_path = Path(processed_dir) / "edges_encoded_train_only.csv"
    save_dataframe(train_only_edges, train_edges_path)

    logger.info(
        "Full encoded edges: %d | Train-only edges: %d | Removed: %d",
        len(full_encoded_edges),
        len(train_only_edges),
        len(full_encoded_edges) - len(train_only_edges),
    )

    graph_artifacts = build_graphs_from_processed(
        encoded_edges_path=train_edges_path,
        node_mapping_path=processed_artifacts.node_mapping_path,
        processed_dir=processed_dir,
        undirected=bool(config["graph"]["undirected"]),
    )

    # ── STEP 4: Load metadata and rebuild adjacency from train-only edges ─────

    test_df = load_split_dataframe(split_artifacts.test_path)
    val_df = load_split_dataframe(split_artifacts.val_path)
    metadata = load_json(processed_artifacts.metadata_path)

    from src.data.preprocess import build_typed_adjacency
    train_adjacency = build_typed_adjacency(
        train_only_edges,
        make_undirected=bool(config["graph"]["undirected"]),
    )
    logger.info("Adjacency rebuilt from train-only edges for metapath counting")

    model_outputs: dict[str, dict[str, Any]] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}

    run_heuristics = bool(config["models"]["run"].get("heuristics", True))
    run_node2vec   = bool(config["models"]["run"].get("node2vec", True))
    run_han        = bool(config["models"]["run"].get("han", True))

    # ── STEP 5: Heuristics ────────────────────────────────────────────────────

    if run_heuristics:
        logger.info("Running heuristic baselines")
        with Path(graph_artifacts["networkx"]).open("rb") as handle:
            nx_graph = pickle.load(handle)
        heur_scores = score_pairs_with_heuristics(nx_graph, test_df)
        heur_pred = test_df.merge(
            heur_scores,
            on=["disease_global_id", "gene_global_id"],
            how="left",
        )

        heur_metrics = evaluate_and_persist(
            predictions=heur_pred,
            score_col="score_heuristic_avg",
            model_name="heuristics",
            result_dir=result_dir,
            figure_dir=figure_dir,
            top_k=int(config["evaluation"]["top_k"]),
            disease_col="disease_global_id",
            gene_col="gene_global_id",
        )
        metrics_by_model["heuristics"] = heur_metrics
        model_outputs["heuristics"] = {
            "test": heur_pred,
            "score_col": "score_heuristic_avg",
            "disease_col": "disease_global_id",
            "gene_col": "gene_global_id",
        }

    device = infer_device(config["runtime"]["device"])

    # ── STEP 6: Node2Vec ──────────────────────────────────────────────────────

    if run_node2vec:
        logger.info("Training Node2Vec on device=%s", device)
        n2v_result = run_node2vec_training(
            graph_path=graph_artifacts["networkx"],
            node_mapping_path=processed_artifacts.node_mapping_path,
            train_split_path=split_artifacts.train_path,
            val_split_path=split_artifacts.val_path,
            test_split_path=split_artifacts.test_path,
            model_config=config["models"]["node2vec"],
            output_dir=weights_dir,
            device=device,
        )

        node2vec_metrics = evaluate_and_persist(
            predictions=n2v_result["test_predictions"],
            score_col="score_node2vec",
            model_name="node2vec",
            result_dir=result_dir,
            figure_dir=figure_dir,
            top_k=int(config["evaluation"]["top_k"]),
            disease_col="disease_global_id",
            gene_col="gene_global_id",
        )
        metrics_by_model["node2vec"] = node2vec_metrics
        model_outputs["node2vec"] = {
            "val": n2v_result["val_predictions"],
            "test": n2v_result["test_predictions"],
            "score_col": "score_node2vec",
            "disease_col": "disease_global_id",
            "gene_col": "gene_global_id",
        }

    # ── STEP 7: HAN ───────────────────────────────────────────────────────────

    if run_han:
        if "hetero" not in graph_artifacts:
            logger.warning("Skipping HAN because torch-geometric hetero graph is unavailable")
        else:
            logger.info("Training HAN on device=%s", device)
            han_result = run_han_training(
                hetero_graph_path=graph_artifacts["hetero"],
                train_split_path=split_artifacts.train_path,
                val_split_path=split_artifacts.val_path,
                test_split_path=split_artifacts.test_path,
                model_config=config["models"]["han"],
                output_dir=weights_dir,
                disease_type=disease_type,
                gene_type=gene_type,
                device=device,
                seed=seed,
            )

            han_metrics = evaluate_and_persist(
                predictions=han_result["test_predictions"],
                score_col="score_han",
                model_name="han",
                result_dir=result_dir,
                figure_dir=figure_dir,
                top_k=int(config["evaluation"]["top_k"]),
                disease_col="disease_local_id",
                gene_col="gene_local_id",
            )
            metrics_by_model["han"] = han_metrics
            model_outputs["han"] = {
                "val": han_result["val_predictions"],
                "test": han_result["test_predictions"],
                "score_col": "score_han",
                "disease_col": "disease_local_id",
                "gene_col": "gene_local_id",
            }

    # ── STEP 8: Hybrid fusion ─────────────────────────────────────────────────

    hybrid_cfg = config["models"]["hybrid"]
    gnn_source = hybrid_cfg["gnn_source"]

    if gnn_source in model_outputs and "val" in model_outputs[gnn_source]:
        logger.info("Running hybrid fusion using %s scores", gnn_source)
        metapaths = config.get("metapaths", {}).get("definitions", DEFAULT_METAPATHS)

        counter = MetapathCounter(
            adjacency=train_adjacency,
            node_counts=metadata.get("node_counts", {}),
            cache_size=int(config.get("metapaths", {}).get("cache_size", 100_000)),
        )

        source_val  = model_outputs[gnn_source]["val"].copy()
        source_test = model_outputs[gnn_source]["test"].copy()
        source_score_col = model_outputs[gnn_source]["score_col"]

        val_counts_long  = compute_metapath_counts_for_predictions(source_val,  counter, metapaths)
        test_counts_long = compute_metapath_counts_for_predictions(source_test, counter, metapaths)

        val_counts  = pivot_metapath_counts(val_counts_long)
        test_counts = pivot_metapath_counts(test_counts_long)

        join_keys = ["disease_local_id", "gene_local_id"]
        val_merged  = source_val.merge(val_counts,   on=join_keys, how="left").fillna(0)
        test_merged = source_test.merge(test_counts, on=join_keys, how="left").fillna(0)

        metapath_names = list(metapaths.keys())
        for col in metapath_names:
            if col not in val_merged.columns:
                val_merged[col] = 0
            if col not in test_merged.columns:
                test_merged[col] = 0

        x_val  = val_merged[metapath_names].to_numpy(dtype=float)
        y_val  = val_merged["label"].to_numpy(dtype=int)
        gnn_val = val_merged[source_score_col].to_numpy(dtype=float)

        hybrid_model = HybridModel(
            alpha=float(hybrid_cfg["alpha"]),
            metapath_names=metapath_names,
        )

        if bool(hybrid_cfg.get("learn_metapath_weights", True)):
            learned_weights = hybrid_model.fit_path_weights(x_val, y_val)
        else:
            configured_weights = hybrid_cfg.get("metapath_weights", {})
            learned_weights = np.array(
                [float(configured_weights.get(name, 1.0)) for name in metapath_names],
                dtype=float,
            )
            hybrid_model.metapath_weights = learned_weights

        # path_score now returns proper probabilities from logistic regression
        val_path_scores = hybrid_model.path_score(x_val)

        alpha_values = np.linspace(0.0, 1.0, 21)
        alpha_metrics = [
            compute_auc_pr(y_val, alpha * gnn_val + (1.0 - alpha) * val_path_scores)
            for alpha in alpha_values
        ]

        if bool(hybrid_cfg.get("search_alpha", True)):
            best_alpha, _ = grid_search_alpha(
                gnn_scores=gnn_val,
                path_scores=val_path_scores,
                labels=y_val,
                metric_fn=compute_auc_pr,
                alpha_grid=alpha_values,
            )
            hybrid_model.alpha = best_alpha

        x_test   = test_merged[metapath_names].to_numpy(dtype=float)
        
        gnn_test = test_merged[source_score_col].to_numpy(dtype=float)

        test_merged["score_path"]   = hybrid_model.path_score(x_test)
        test_merged["score_hybrid"] = hybrid_model.final_score(gnn_test, x_test)

        # ── Standardized contributions for interpretability ───────────────────
        # Compute scaled_count × logistic_coeff for each metapath per test pair.
        # This reflects what the logistic regression actually used, on the same
        # scale, so the interpretability plots are honest.
        x_test_scaled = hybrid_model._scaler.transform(x_test)
        
        standardized_contributions = pd.DataFrame(
            x_test_scaled * hybrid_model.metapath_weights,
            columns=metapath_names,
            index=test_merged.index,
        )
        test_merged_with_contribs = test_merged.copy()
        for name in metapath_names:
            test_merged_with_contribs[f"contrib_{name}"] = standardized_contributions[name].values

        hybrid_metrics = evaluate_and_persist(
            predictions=test_merged,
            score_col="score_hybrid",
            model_name="hybrid",
            result_dir=result_dir,
            figure_dir=figure_dir,
            top_k=int(config["evaluation"]["top_k"]),
            disease_col="disease_local_id",
            gene_col="gene_local_id",
        )
        metrics_by_model["hybrid"] = hybrid_metrics

        plot_alpha_performance(
            alpha_values=alpha_values.tolist(),
            metric_values=[float(v) for v in alpha_metrics],
            metric_name="AUC-PR",
            output_path=figure_dir / "alpha_vs_performance.png",
        )

        metapath_weight_dict = {
            name: float(weight)
            for name, weight in zip(metapath_names, learned_weights, strict=False)
        }
        save_json(metapath_weight_dict, result_dir / "metapath_weights.json")
        save_dataframe(test_counts_long, result_dir / "metapath_counts_test_long.csv")

        # Pass test_merged_with_contribs so explain.py uses standardized contributions
        explanations = build_explanations(
            predictions=test_merged_with_contribs,   # ← has contrib_ columns
            metapath_count_df=test_counts_long,
            score_col="score_hybrid",
            metapath_weights=metapath_weight_dict,
            top_n=int(config["interpretability"]["top_n"]),
            disease_col="disease_local_id",
            gene_col="gene_local_id",
            name_lookup=name_lookup,
            disease_type=data_cfg["disease_type"],
            gene_type=data_cfg["gene_type"],
        )

        save_all_interpretability_outputs(
            explanations=explanations,
            metapath_weights=metapath_weight_dict,
            result_dir=result_dir,
            figure_dir=figure_dir,
            top_n_plot=20,
        )

    # ── STEP 9: Ablation plots and summary ────────────────────────────────────

    if metrics_by_model:
        aucroc_ablation = {
            m: float(v.get("auc_roc", 0.0)) for m, v in metrics_by_model.items()
        }
        aucpr_ablation = {
            m: float(v.get("auc_pr", 0.0)) for m, v in metrics_by_model.items()
        }

        plot_ablation_bars(
            metric_by_model=aucroc_ablation,
            metric_name="AUC-ROC",
            output_path=figure_dir / "ablation_aucroc.png",
        )
        plot_ablation_bars(
            metric_by_model=aucpr_ablation,
            metric_name="AUC-PR",
            output_path=figure_dir / "ablation_aucpr.png",
        )

    summary = {
        "run_dir": str(result_dir),
        "model_metrics": metrics_by_model,
    }
    save_json(summary, result_dir / "metrics.json")
    logger.info("Pipeline completed. Summary saved to %s", result_dir / "metrics.json")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_yaml_config(args.config)
    overrides = parse_overrides(args.override)
    config = deep_update(config, overrides)
    run_pipeline(config=config, config_path=args.config, run_name=args.run_name)


if __name__ == "__main__":
    main()