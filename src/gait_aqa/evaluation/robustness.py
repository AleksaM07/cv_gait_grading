"""Image degradation utilities for robustness experiments."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(frames: np.ndarray, sigma: float = 8.0, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise to RGB frames."""
    rng = np.random.default_rng(seed)
    noisy = frames.astype(float) + rng.normal(0.0, sigma, size=frames.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def drop_frames(frames: np.ndarray, every: int = 3) -> np.ndarray:
    """Drop every Nth frame while preserving at least two frames."""
    keep = np.asarray([index % every != 0 for index in range(len(frames))])
    degraded = frames[keep]
    return degraded if len(degraded) >= 2 else frames[:2]
