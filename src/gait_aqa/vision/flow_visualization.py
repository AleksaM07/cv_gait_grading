"""Flow visualization utilities."""

from __future__ import annotations

import numpy as np


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert flow fields to a simple RGB magnitude/angle image."""
    if flow.ndim == 4:
        field = flow.mean(axis=0)
    else:
        field = flow
    magnitude = np.linalg.norm(field, axis=-1)
    angle = np.arctan2(field[..., 1], field[..., 0])
    hue = (angle + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)
    val = magnitude / max(float(np.percentile(magnitude, 95)), 1e-6)
    return _hsv_to_rgb(hue, sat, np.clip(val, 0.0, 1.0))


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    channels = np.zeros(h.shape + (3,), dtype=np.float32)
    choices = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    for case, vals in enumerate(choices):
        mask = i == case
        for channel, value in enumerate(vals):
            channels[..., channel][mask] = value[mask]
    return np.clip(channels * 255.0, 0, 255).astype(np.uint8)
