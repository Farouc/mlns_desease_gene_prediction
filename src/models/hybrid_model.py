"""Hybrid score fusion between GNN-based predictions and metapath reasoning.

Both GNN and path scores are proper probabilities in [0, 1], so alpha
is a meaningful interpolation between the two models:

    hybrid_score = alpha * P_gnn(link) + (1 - alpha) * P_path(link)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for hybrid weight fitting") from exc


@dataclass
class HybridModel:
    """Hybrid scorer: alpha * s_gnn + (1 - alpha) * s_path.

    Both s_gnn and s_path are proper probabilities in [0, 1], so alpha
    is a meaningful interpolation between the two models.
    """

    alpha: float
    metapath_names: list[str]
    metapath_weights: np.ndarray | None = None
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _logistic_model: LogisticRegression | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

    def fit_path_weights(self, path_features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit logistic regression on metapath counts.

        Stores the full logistic regression model so that path_score()
        returns proper probabilities in [0, 1] rather than raw weighted
        sums. Features are standardized before fitting so coefficients
        are comparable across metapaths with different count scales.

        Args:
            path_features: Shape (n_pairs, n_metapaths), metapath counts.
            labels: Binary labels, shape (n_pairs,).

        Returns:
            Non-negative coefficient array of shape (n_metapaths,),
            exposed for interpretability only.
        """
        if path_features.ndim != 2:
            raise ValueError("path_features must be a 2D array.")

        # Standardize so high-count metapaths don't dominate by scale alone
        self._scaler = StandardScaler()
        path_features_scaled = self._scaler.fit_transform(path_features)

        self._logistic_model = LogisticRegression(
            fit_intercept=True,
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
        )
        self._logistic_model.fit(path_features_scaled, labels)

        # Expose coefficients clipped to non-negative for interpretability
        weights = np.clip(self._logistic_model.coef_[0].astype(float), 0.0, None)

        # Fall back to uniform weights if all are zero (degenerate case)
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones(path_features.shape[1], dtype=float)

        self.metapath_weights = weights
        return weights

    def path_score(self, path_features: np.ndarray) -> np.ndarray:
        """Compute path-based score as a proper probability in [0, 1].

        Uses the fitted logistic regression to predict P(label=1) given
        metapath count features. This ensures path_score and gnn_score
        are on the same scale before mixing with alpha.

        Args:
            path_features: Shape (n_pairs, n_metapaths), metapath counts.

        Returns:
            Probability array of shape (n_pairs,), values in [0, 1].
        """
        if self._logistic_model is None or self._scaler is None:
            raise RuntimeError("Model is not fitted yet. Call fit_path_weights first.")

        path_features_scaled = self._scaler.transform(path_features)
        # predict_proba returns (n_pairs, 2); column 1 is P(label=1)
        return self._logistic_model.predict_proba(path_features_scaled)[:, 1]

    def final_score(self, gnn_scores: np.ndarray, path_features: np.ndarray) -> np.ndarray:
        """Compute final hybrid score as a probability in [0, 1].

        Both gnn_scores and path_score() are probabilities, so alpha
        is a true interpolation:
            score = alpha * P_gnn(link) + (1 - alpha) * P_path(link)

        Args:
            gnn_scores: GNN probability scores, shape (n_pairs,), in [0,1].
            path_features: Metapath count matrix, shape (n_pairs, n_metapaths).

        Returns:
            Hybrid probability array of shape (n_pairs,), values in [0, 1].
        """
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
        gnn_scores: GNN probability scores, in [0, 1].
        path_scores: Path probability scores, in [0, 1].
        labels: Binary labels.
        metric_fn: Callable(labels, scores) -> metric.
        alpha_grid: Candidate alphas to try.

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