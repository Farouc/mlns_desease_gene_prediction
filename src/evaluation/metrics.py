"""Core classification and ranking metrics for link prediction."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for evaluation metrics") from exc


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC safely for binary labels."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under precision-recall curve."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def hits_at_k(ranked_labels: list[int], k: int) -> float:
    """Compute Hits@k for one ranked list (1 if any positive in top-k else 0)."""
    if not ranked_labels:
        return 0.0
    top_k = ranked_labels[:k]
    return float(any(label == 1 for label in top_k))


def reciprocal_rank(ranked_labels: list[int]) -> float:
    """Compute reciprocal rank for one ranked list."""
    for idx, label in enumerate(ranked_labels, start=1):
        if label == 1:
            return 1.0 / float(idx)
    return 0.0
