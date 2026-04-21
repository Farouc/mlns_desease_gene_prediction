# Interpretable Disease-Gene Prediction via Hybrid Graph Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This repository implements a hybrid framework for disease-gene link prediction on heterogeneous biomedical knowledge graphs. We combine Graph Neural Network representations with interpretable metapath-based logistic regression, achieving state-of-the-art ranking performance while providing transparent, per-prediction explanations. The framework is evaluated on Hetionet v1.0 and demonstrates that phenotype-sharing metapaths are the dominant predictor of novel disease-gene associations.

## Motivation

Predicting associations between diseases and genes is fundamental to drug discovery, variant prioritisation, and personalised medicine. While Graph Neural Networks achieve strong predictive performance, they operate as black boxes—a critical limitation in biomedical settings where researchers must justify experimental follow-up decisions. Our hybrid approach bridges this gap by fusing learned GNN representations with explicit, interpretable metapath evidence.

## Methods

We implement and compare four approaches:

| Method | Description |
|--------|-------------|
| **Heuristics** | Common Neighbours and Adamic-Adar on projected bipartite graph |
| **Node2Vec** | Skip-gram embeddings from biased random walks on homogeneous projection |
| **HAN** | Heterogeneous Attention Network with node-level and semantic-level attention |
| **Hybrid** | Linear interpolation of GNN probabilities with metapath logistic regression |

### Hybrid Model

The hybrid model computes:

```
s_hybrid(d,g) = α · P_GNN(d,g) + (1-α) · P_path(d,g)
```

where `P_path` is obtained from logistic regression on standardised metapath counts:

```
P_path = σ(w₁·c̃₁ + w₂·c̃₂ + w₃·c̃₃ + b)
```

The fusion parameter `α` is optimised via grid search on validation AUC-PR.

### Metapaths

We define three biologically motivated metapaths:

| Metapath | Schema | Biological Rationale |
|----------|--------|---------------------|
| `DpSpDaG` | Disease→Symptom←Disease→Gene | Phenotype-driven gene discovery |
| `DaGpPWpG` | Disease→Gene→Pathway←Gene | Pathway co-participation |
| `DaGiG` | Disease→Gene↔Gene | Protein-protein interaction |

## Results

Best results from `experiments_hala/results/full_cuda_hybrid_han/`:

| Model | AUC-ROC | AUC-PR | Hits@10 | MRR |
|-------|--------:|-------:|--------:|----:|
| Heuristics | 0.752 | 0.425 | 0.831 | 0.586 |
| Node2Vec | 0.939 | 0.669 | 0.904 | 0.793 |
| HAN | 0.914 | 0.649 | 0.868 | 0.658 |
| **Hybrid** | **0.967** | **0.838** | **0.912** | **0.805** |

**Key findings:**
- Hybrid achieves +25% AUC-PR over HAN and +19% over Node2Vec
- Optimal fusion at α=0.7 (70% GNN, 30% metapath)
- `DpSpDaG` metapath receives highest coefficient (3.21), confirming phenotype sharing as primary predictor
- 86% of top-50 predictions are validated known associations

## Repository Structure

```
├── configs/                    # Experiment configurations
│   ├── default.yaml
│   ├── han.yaml               # HAN-specific config
│   ├── han_hala.yaml          # Extended metapath config
│   └── node2vec.yaml
├── data/
│   ├── raw/                   # Raw Hetionet edge CSV
│   ├── processed/             # Processed graph artifacts
│   └── splits/                # Train/val/test splits
├── src/
│   ├── data/                  # Data loading and splitting
│   ├── graph/                 # Graph construction (NetworkX, PyG)
│   ├── models/                # Model implementations
│   │   ├── heuristics.py      # CN, AA baselines
│   │   ├── node2vec_model.py  # Node2Vec wrapper
│   │   ├── han_model.py       # HAN implementation
│   │   └── hybrid_model.py    # Hybrid fusion with logistic regression
│   ├── training/              # Training loops
│   ├── evaluation/            # Metrics computation
│   ├── interpretability/      # Metapath explanation generation
│   ├── visualization/         # Publication figure generation
│   └── main.py                # Main experiment runner
├── experiments_hala/          # Experiment outputs
│   ├── results/               # Metrics, predictions, weights
│   └── figures/               # Generated plots (PNG/PDF)
├── reports/                   # LaTeX report and figures
├── scripts/                   # Helper scripts
│   ├── run_all.sh
│   ├── generate_plots.py
│   └── ablation_alpha.sh
└── requirements.txt
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd mlns_desease_gene_prediction

# Create environment
conda create -n disease-gene python=3.9
conda activate disease-gene

# Install dependencies
pip install -r requirements.txt
pip install -e .

# For GPU support (adjust CUDA version as needed)
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Dataset Preparation

1. Download Hetionet edges and place at `data/raw/hetionet_subset_edges.csv`

2. Required CSV columns (flexible naming):
   - Source node ID: `source`, `source_id`, `src`
   - Target node ID: `target`, `target_id`, `dst`
   - Source type: `source_type`, `src_type`
   - Target type: `target_type`, `dst_type`
   - Edge type: `edge_type`, `relation`, `metaedge`

3. Node types are filtered to: Disease, Gene, Pathway, Symptom/Phenotype

## Reproducing Experiments

### Quick Start

```bash
# Run full pipeline with HAN + Hybrid
python main.py --config configs/han_hala.yaml

# Run with GPU
python main.py --config configs/han_hala.yaml --override runtime.device=cuda
```

### Step-by-Step Reproduction

```bash
# 1. Train HAN baseline
python main.py --config configs/han.yaml \
  --override models.run.hybrid=false

# 2. Train Node2Vec baseline
python main.py --config configs/node2vec.yaml \
  --override models.run.hybrid=false

# 3. Train Hybrid model (HAN + metapath)
python main.py --config configs/han_hala.yaml \
  --override models.hybrid.gnn_source=han

# 4. Alpha sweep ablation
bash scripts/ablation_alpha.sh configs/han_hala.yaml
```

### Generate Figures

```bash
# From saved results (no retraining)
python scripts/generate_plots.py \
  --result-dir experiments_hala/results/full_cuda_hybrid_han \
  --figure-dir experiments_hala/figures/full_cuda_hybrid_han
```

### Run Interpretability Analysis

```bash
# Generate per-prediction explanations
python -c "
from src.interpretability.explain import save_all_interpretability_outputs
save_all_interpretability_outputs(
    'experiments_hala/results/full_cuda_hybrid_han',
    top_n=100
)
"
```

## Configuration Options

Key configuration parameters in YAML:

```yaml
models:
  hybrid:
    gnn_source: han          # Base GNN: 'han' or 'node2vec'
    alpha: 0.7               # Fusion weight (0=path only, 1=GNN only)
    search_alpha: true       # Grid search for optimal alpha
    learn_metapath_weights: true  # Fit logistic regression

metapaths:
  definitions:
    DpSpDaG: [Disease, Symptom, Disease, Gene]
    DaGpPWpG: [Disease, Gene, Pathway, Gene]
    DaGiG: [Disease, Gene, Gene]

evaluation:
  top_k: 10                  # Hits@k threshold

interpretability:
  top_n: 100                 # Number of predictions to explain
```

## Output Artifacts

Each experiment run produces:

| File | Description |
|------|-------------|
| `metrics.json` | All model metrics |
| `metapath_weights.json` | Learned logistic regression coefficients |
| `hybrid_ranked_predictions.csv` | Ranked predictions with scores |
| `interpretability_summary.csv` | Per-prediction explanations with contributions |
| `interpretability_top_predictions.json` | Detailed top-N explanations |

## Citation

If you use this code, please cite:

```bibtex
@article{yartaoui2025disease,
  title={Interpretable Disease-Gene Prediction via Hybrid Graph Learning and Metapath Reasoning},
  author={Yartaoui, Farouk and Chafik, Hala and Tbatou, Hamza and Maddah, Ilyas},
  journal={CentraleSup{\'e}lec MLNS Project Report},
  year={2025}
}
```

## Authors

- Farouk Yartaoui
- Hala Chafik
- Hamza Tbatou
- Ilyas Maddah

CentraleSupélec, 2025

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

- Hetionet dataset: [Himmelstein et al., eLife 2017](https://doi.org/10.7554/eLife.26726)
- HAN architecture: [Wang et al., WWW 2019](https://doi.org/10.1145/3308558.3313562)
- Node2Vec: [Grover & Leskovec, KDD 2016](https://doi.org/10.1145/2939672.2939754)
