# Interpretable Disease-Gene Prediction via Hybrid Graph Learning and Metapath Reasoning

This repository provides a complete, modular research pipeline for disease-gene link prediction on a Hetionet v1.0 subset (Disease, Gene, Pathway, Phenotype), with explicit interpretability via metapath counting.

## Features

- Heterogeneous graph construction from Hetionet edge CSV
- Link prediction baselines and learned models:
  - Common Neighbors
  - Adamic-Adar
  - Node2Vec (dot-product scoring)
  - HAN (Heterogeneous Attention Network)
- Metapath reasoning with cycle-aware traversal and caching for:
  - `DaGpPpG`
  - `DpPhG`
  - `GcGaD`
- Hybrid fusion model:
  - `s_final = alpha * s_GNN + (1 - alpha) * s_path`
- Evaluation metrics:
  - AUC-ROC
  - AUC-PR
  - Hits@10
  - MRR
- Reproducible experiments:
  - deterministic splits
  - random seeding
  - timestamped run directories
  - config snapshot per run
- Interpretability outputs:
  - top metapath per prediction
  - metapath path counts per pair

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
│   ├── utils/
│   └── main.py
├── configs/
├── experiments/
│   ├── results/
│   ├── logs/
│   └── figures/
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

Note: `torch-geometric` installation may require matching your local PyTorch/CUDA setup.

## Dataset Instructions

1. Place your Hetionet subset edge file at:

```text
data/raw/hetionet_subset_edges.csv
```

2. The loader infers columns automatically from common names. Your CSV should contain fields equivalent to:

- source node ID (`source`, `source_id`, `src`, ...)
- target node ID (`target`, `target_id`, `dst`, ...)
- source node type (`source_type`, `src_type`, ...)
- target node type (`target_type`, `dst_type`, ...)
- edge type (`edge_type`, `relation`, `metaedge`, ...)

3. Node types are filtered to the configured subset:

- Disease
- Gene
- Pathway
- Phenotype

## Run Experiments

### Main entrypoint

```bash
python main.py --config configs/default.yaml
```

or

```bash
python src/main.py --config configs/han.yaml
```

### Config overrides

```bash
python main.py \
  --config configs/han.yaml \
  --override models.hybrid.alpha=0.5 models.hybrid.search_alpha=false
```

### Helper scripts

Run all presets:

```bash
bash scripts/run_all.sh
```

Alpha ablation sweep:

```bash
bash scripts/ablation_alpha.sh configs/han.yaml
```

## Expected Outputs

Each run generates a timestamped folder under `experiments/results/<run_id>/` containing:

- `config_input.yaml`
- `config_resolved.yaml`
- `metrics.json`
- `<model>_metrics.json`
- `<model>_predictions.csv`
- `<model>_ranked_predictions.csv`
- `metapath_weights.json` (for hybrid)
- `metapath_counts_test_long.csv`
- `interpretability_top_predictions.json`
- `weights/` with trained model weights (`han_weights.pt`, `node2vec_weights.pt`)

Figures are saved under `experiments/figures/<run_id>/`:

- ROC curves
- PR curves
- alpha-vs-performance plot
- ablation bar plots

Logs are saved under `experiments/logs/<run_id>/run.log`.

## Reproducibility

- Global seeding in `src/utils/seed.py`
- Deterministic split generation in `src/data/split.py`
- Full config snapshot per run

## Notes

- HAN requires `torch-geometric`; if unavailable, HAN training is skipped.
- Hybrid fusion expects a GNN source with validation predictions (`han` or `node2vec`).
