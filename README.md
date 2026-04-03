# Interpretable Disease-Gene Prediction via Hybrid Graph Learning and Metapath Reasoning

This repository provides a modular research pipeline for disease-gene link prediction on a Hetionet v1.0 subset (Disease, Gene, Pathway, Phenotype), with explicit interpretability via metapath counting and a publication-ready visualization layer.

## Features

- Heterogeneous graph construction from Hetionet edge CSV.
- Link prediction baselines and learned models:
  - Common Neighbors
  - Adamic-Adar
  - Node2Vec (dot-product scoring)
  - HAN (Heterogeneous Attention Network)
- Metapath reasoning with cycle-aware traversal and caching for:
  - `DaGpPpG`
  - `DpPhG`
  - `GcGaD`
- Hybrid fusion:
  - `s_final = alpha * s_GNN + (1 - alpha) * s_path`
- Evaluation metrics:
  - AUC-ROC
  - AUC-PR
  - Hits@10
  - MRR
- Interpretability artifacts:
  - top metapath per prediction
  - metapath path counts
- Visualization pipeline from saved artifacts only (no retraining), including:
  - model comparison bar plot
  - PR curve comparison
  - alpha trade-off curve
  - Hits@k curve
  - ranking distribution
  - metapath contribution plot
  - performance vs interpretability scatter

## Repository Structure

```text
project_root/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   ├── graph/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── interpretability/
│   ├── visualization/
│   ├── utils/
│   └── main.py
├── configs/
├── experiments/
│   ├── results/
│   ├── logs/
│   └── figures/
├── reports/
├── notebooks/
├── scripts/
├── requirements.txt
├── setup.py
└── main.py
```

## Installation

```bash
base
pip install -r requirements.txt
pip install -e .
```

Note: `torch-geometric` must match your local PyTorch/CUDA setup.

## Dataset Instructions

1. Place the project input file at:

```text
data/raw/hetionet_subset_edges.csv
```

2. The loader infers common column aliases. Required semantics:

- source node ID (`source`, `source_id`, `src`, ...)
- target node ID (`target`, `target_id`, `dst`, ...)
- source node type (`source_type`, `src_type`, ...)
- target node type (`target_type`, `dst_type`, ...)
- edge type (`edge_type`, `relation`, `metaedge`, ...)

3. Node types are filtered to:

- Disease
- Gene
- Pathway
- Phenotype

## Run Training / Evaluation

Main entrypoint:

```bash
python main.py --config configs/default.yaml
```

Alternative:

```bash
python src/main.py --config configs/han.yaml
```

Override example:

```bash
python main.py \
  --config configs/han.yaml \
  --override models.hybrid.alpha=0.5 models.hybrid.search_alpha=false
```

Helper scripts:

```bash
bash scripts/run_all.sh
bash scripts/ablation_alpha.sh configs/han.yaml
```

## Generate Publication Figures (No Retraining)

This uses only saved artifacts from the final run:

```bash
python scripts/generate_plots.py \
  --result-dir experiments/results/final_full_cuda_hybrid_e15_fast \
  --figure-dir experiments/figures/final_full_cuda_hybrid_e15_fast
```

Optional:

```bash
python scripts/generate_plots.py --overwrite
```

## Retained Runs in This Repository

- `experiments/results/n2v_full_cuda`
- `experiments/results/han_full_cuda`
- `experiments/results/final_full_cuda_hybrid_e15_fast`
- matching figure folders under `experiments/figures/`

## Final Full CUDA Hybrid Results

Run folder: `experiments/results/final_full_cuda_hybrid_e15_fast`

| Model      | AUC-ROC | AUC-PR | Hits@10 | MRR |
|------------|--------:|-------:|--------:|----:|
| Heuristics | 0.7525  | 0.4251 | 0.8309  | 0.5857 |
| Node2Vec   | 0.8498  | 0.4938 | 0.8603  | 0.5866 |
| HAN        | 0.8248  | 0.4573 | 0.6397  | 0.3550 |
| Hybrid     | 0.8495  | 0.5192 | 0.8088  | 0.6287 |

## Expected Artifacts

Per run in `experiments/results/<run_id>/`:

- `config_input.yaml`
- `config_resolved.yaml`
- `metrics.json`
- `<model>_metrics.json`
- `<model>_predictions.csv`
- `<model>_ranked_predictions.csv`
- `metapath_weights.json` (hybrid runs)
- `metapath_counts_test_long.csv` (hybrid runs)
- `interpretability_top_predictions.json` (hybrid runs)
- `weights/` with saved model outputs and checkpoints

Figures in `experiments/figures/final_full_cuda_hybrid_e15_fast/` include:

- existing training-time plots (ROC/PR, ablations, alpha-vs-performance)
- publication plots:
  - `model_comparison_bar.png`
  - `pr_curve_comparison.png`
  - `alpha_tradeoff.png`
  - `hits_at_k.png`
  - `ranking_distribution.png`
  - `metapath_contributions.png`
  - `performance_vs_interpretability.png`

Both `.png` and `.pdf` are exported for publication plots.

## Reproducibility

- Global seeding in `src/utils/seed.py`
- Deterministic split generation in `src/data/split.py`
- Config snapshots per run

## Notes

- HAN requires `torch-geometric`; if unavailable, HAN training is skipped.
- Hybrid fusion expects a GNN source with validation predictions (`han` or `node2vec`).
