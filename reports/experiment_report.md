# Experiment Report: Interpretable Disease-Gene Prediction

Date: 2026-04-03

## 1) Scope

This report summarizes the current repository state after:

- data acquisition and preprocessing integration,
- CUDA training runs (Node2Vec, HAN, Hybrid full run),
- evaluation and interpretability export,
- addition of a publication-ready visualization pipeline that reuses saved artifacts only.

## 2) What Was Done

### 2.1 Data acquisition and preparation

- Downloaded official Hetionet v1.0 from Zenodo (`10.5281/zenodo.268568`).
- Extracted raw files into `data/raw/`.
- Built `data/raw/hetionet_subset_edges.csv` filtered to:
  - `Disease`
  - `Gene`
  - `Pathway`
  - `Phenotype`
- Applied mapping: Hetionet `Symptom` -> project `Phenotype`.

### 2.2 Environment and dependencies

- Used `base` environment for all execution.
- Verified main dependencies: `torch`, `torch-geometric`, `networkx`, `pandas`, `scikit-learn`, `matplotlib`, `yaml`, `tqdm`.
- Installed missing PyG runtime components required by Node2Vec in this environment.

### 2.3 Pipeline/debug fixes

- Fixed split generation global-ID merge issue:
  - `src/data/split.py`
- Fixed NetworkX graph persistence compatibility:
  - `src/graph/build_graph.py`
  - `src/main.py`
  - `src/training/train_node2vec.py`
- Improved HAN training runtime efficiency:
  - `src/training/train_han.py`
- Accelerated metapath counting with sparse propagation + caching:
  - `src/graph/metapaths.py`
  - `src/main.py`

### 2.4 Executed CUDA runs retained in repo

- `experiments/results/n2v_full_cuda`
- `experiments/results/han_full_cuda`
- `experiments/results/final_full_cuda_hybrid_e15_fast`

Matching figure folders are kept under `experiments/figures/`.

### 2.5 Visualization extension (artifact-only)

Added `src/visualization/` modules and `scripts/generate_plots.py` to generate figures from:

- `experiments/results/final_full_cuda_hybrid_e15_fast`

No retraining is performed by this visualization step.

## 3) Final Full CUDA Hybrid Results

Run: `experiments/results/final_full_cuda_hybrid_e15_fast`

| Model      | AUC-ROC | AUC-PR | Hits@10 | MRR |
|------------|--------:|-------:|--------:|----:|
| Heuristics | 0.7525  | 0.4251 | 0.8309  | 0.5857 |
| Node2Vec   | 0.8498  | 0.4938 | 0.8603  | 0.5866 |
| HAN        | 0.8248  | 0.4573 | 0.6397  | 0.3550 |
| Hybrid     | 0.8495  | 0.5192 | 0.8088  | 0.6287 |

Observations:

- Hybrid is best on AUC-PR and MRR.
- Node2Vec is best on AUC-ROC and Hits@10.
- HAN alone underperforms Node2Vec/Hybrid on ranking metrics in this setting.

## 4) Artifacts Produced

### 4.1 Results artifacts

For `final_full_cuda_hybrid_e15_fast`:

- `metrics.json` and per-model `*_metrics.json`
- `*_predictions.csv` and `*_ranked_predictions.csv`
- `metapath_counts_test_long.csv`
- `metapath_weights.json`
- `interpretability_top_predictions.json`
- `alpha_tradeoff_metrics.csv` (generated from saved hybrid scores for plot sweep)
- `weights/` checkpoints and saved train/val/test outputs

### 4.2 Figures generated

Directory:

- `experiments/figures/final_full_cuda_hybrid_e15_fast/`

Core publication figures:

- `model_comparison_bar.png` (+ `.pdf`)
- `pr_curve_comparison.png` (+ `.pdf`)
- `alpha_tradeoff.png` (+ `.pdf`)

Advanced figures:

- `hits_at_k.png` (+ `.pdf`)
- `ranking_distribution.png` (+ `.pdf`)
- `metapath_contributions.png` (+ `.pdf`)
- `performance_vs_interpretability.png` (+ `.pdf`)

Also preserved:

- per-model ROC/PR curves from training pipeline,
- ablation plots (`ablation_aucroc.png`, `ablation_aucpr.png`),
- original alpha plot (`alpha_vs_performance.png`).

## 5) Commands Used for Reproducible Figure Generation

```bash
python scripts/generate_plots.py \
  --result-dir experiments/results/final_full_cuda_hybrid_e15_fast \
  --figure-dir experiments/figures/final_full_cuda_hybrid_e15_fast
```

Optional overwrite:

```bash
python scripts/generate_plots.py --overwrite
```

## 6) Notes for Analysis

- The repository now includes both training/evaluation code and a clean post-hoc figure pipeline.
- The visualization script is idempotent: existing figures are scanned and skipped unless `--overwrite` is set.
- The retained CUDA runs are intended as the "honest" comparison baseline and are now set up to be versioned/pushed.
