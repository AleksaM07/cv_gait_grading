"""Foreground extraction for fixed-camera videos."""

from __future__ import annotations

import numpy as np


def foreground_mask(gray_frames: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """Estimate a simple foreground mask by deviation from median background."""
    if gray_frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape T,H,W")
    background = np.median(gray_frames, axis=0)
    return np.abs(gray_frames - background) >= threshold


def crop_bounds_from_mask(mask: np.ndarray, padding: int = 4) -> tuple[int, int, int, int]:
    """Return `(y0, y1, x0, x1)` bounds covering foreground masks."""
    if mask.ndim != 3:
        raise ValueError("Expected mask with shape T,H,W")
    combined = mask.any(axis=0)
    if not combined.any():
        height, width = combined.shape
        return 0, height, 0, width
    y, x = np.nonzero(combined)
    height, width = combined.shape
    return (
        max(0, int(y.min()) - padding),
        min(height, int(y.max()) + padding + 1),
        max(0, int(x.min()) - padding),
        min(width, int(x.max()) + padding + 1),
    )
