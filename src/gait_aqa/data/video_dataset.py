"""Deterministic clip sampling utilities."""

from __future__ import annotations

import numpy as np


def deterministic_indices(frame_count: int, clip_length: int) -> np.ndarray:
    """Return deterministic evenly spaced frame indices."""
    if frame_count <= 0 or clip_length <= 0:
        raise ValueError("frame_count and clip_length must be positive")
    if frame_count >= clip_length:
        return np.linspace(0, frame_count - 1, clip_length).round().astype(int)
    tail = np.full(clip_length - frame_count, frame_count - 1, dtype=int)
    return np.concatenate([np.arange(frame_count), tail])
