"""Reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during docs-only usage
    torch = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if available).

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
