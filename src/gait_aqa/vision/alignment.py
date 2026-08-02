"""Simple motion alignment helpers."""

from __future__ import annotations

import numpy as np


def estimate_global_translation(flow: np.ndarray) -> np.ndarray:
    """Estimate per-frame global translation as median dense flow."""
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("Flow must have shape T,H,W,2")
    return np.median(flow.reshape(flow.shape[0], -1, 2), axis=1)


def remove_global_translation(flow: np.ndarray) -> np.ndarray:
    """Return body-centered residual flow."""
    translation = estimate_global_translation(flow)
    return flow - translation[:, None, None, :]
