"""Hybrid score fusion between GNN-based predictions and metapath reasoning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.linear_model import LinearRegression
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for hybrid weight fitting") from exc


@dataclass
class HybridModel:
    """Hybrid scorer: alpha * s_gnn + (1 - alpha) * s_path."""

    alpha: float
    metapath_names: list[str]
    metapath_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

    def fit_path_weights(self, path_features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit non-negative metapath weights using linear regression."""
        if path_features.ndim != 2:
            raise ValueError("path_features must be a 2D array.")

        model = LinearRegression(fit_intercept=False, positive=True)
        model.fit(path_features, labels)

        weights = model.coef_.astype(float)
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones(path_features.shape[1], dtype=float)
        self.metapath_weights = weights
        return weights

    def path_score(self, path_features: np.ndarray) -> np.ndarray:
        """Compute weighted metapath score."""
        if self.metapath_weights is None:
            raise RuntimeError("Metapath weights are not fit yet.")
        return np.dot(path_features, self.metapath_weights)

    def final_score(self, gnn_scores: np.ndarray, path_features: np.ndarray) -> np.ndarray:
        """Compute final hybrid score."""
        path_scores = self.path_score(path_features)
        return self.alpha * gnn_scores + (1.0 - self.alpha) * path_scores


def grid_search_alpha(
    gnn_scores: np.ndarray,
    path_scores: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    alpha_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Search alpha maximizing a validation metric.

    Args:
        gnn_scores: GNN model scores.
        path_scores: Path-based scores.
        labels: Binary labels.
        metric_fn: Callable(score, label) -> metric.
        alpha_grid: Candidate alphas.

    Returns:
        (best_alpha, best_metric)
    """
    candidates = alpha_grid if alpha_grid is not None else np.linspace(0.0, 1.0, 21)
    best_alpha = float(candidates[0])
    best_metric = float("-inf")

    for alpha in candidates:
        fused = float(alpha) * gnn_scores + (1.0 - float(alpha)) * path_scores
        metric = float(metric_fn(labels, fused))
        if metric > best_metric:
            best_metric = metric
            best_alpha = float(alpha)

    return best_alpha, best_metric
