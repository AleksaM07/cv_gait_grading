"""Loss helpers for optional deep-video work."""

from __future__ import annotations

import numpy as np


def smooth_l1(prediction: np.ndarray, target: np.ndarray, beta: float = 1.0) -> float:
    """Compute a NumPy Smooth L1 loss."""
    error = np.abs(prediction - target)
    loss = np.where(error < beta, 0.5 * error**2 / beta, error - 0.5 * beta)
    return float(loss.mean())
