"""Approximate body-region summaries for fixed-camera clips."""

from __future__ import annotations

import numpy as np


REGIONS = {
    "torso": (0.20, 0.55, 0.25, 0.70),
    "left_leg": (0.50, 0.90, 0.15, 0.50),
    "right_leg": (0.50, 0.90, 0.50, 0.85),
    "left_foot": (0.75, 1.00, 0.05, 0.50),
    "right_foot": (0.75, 1.00, 0.50, 0.95),
}


def regional_flow_magnitudes(flow: np.ndarray) -> dict[str, np.ndarray]:
    """Return mean flow magnitude per approximate body region."""
    height, width = flow.shape[1:3]
    magnitude = np.linalg.norm(flow, axis=-1)
    summaries: dict[str, np.ndarray] = {}
    for name, (y0, y1, x0, x1) in REGIONS.items():
        ys = slice(int(y0 * height), int(y1 * height))
        xs = slice(int(x0 * width), int(x1 * width))
        summaries[name] = magnitude[:, ys, xs].mean(axis=(1, 2))
    return summaries
