"""Frame preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def resample_frames(
    frames: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample a clip to a fixed frame rate with deterministic timestamps.

    Optical flow measures displacement per frame. Standardizing the sampling
    rate therefore prevents otherwise identical motion from producing features
    on incompatible scales solely because the source FPS differs.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB frames with shape T,H,W,3")
    if frames.shape[0] < 2:
        raise ValueError("At least two frames are required")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be positive and finite")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be positive and finite")
    if np.isclose(source_fps, target_fps):
        return frames

    duration_seconds = (frames.shape[0] - 1) / source_fps
    output_count = max(2, round(duration_seconds * target_fps) + 1)
    timestamps = np.arange(output_count, dtype=float) / target_fps
    source_indices = np.rint(timestamps * source_fps).astype(int)
    source_indices = np.clip(source_indices, 0, frames.shape[0] - 1)
    return frames[source_indices]


def preprocess_frames(
    frames: np.ndarray,
    size: tuple[int, int] = (96, 96),
    grayscale: bool = True,
    denoise: bool = True,
) -> np.ndarray:
    """Resize frames and optionally convert to normalized grayscale."""
    processed: list[np.ndarray] = []
    for frame in frames:
        image = Image.fromarray(frame)
        image.thumbnail(size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", size, (0, 0, 0))
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        canvas.paste(image, offset)
        if denoise:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))
        if grayscale:
            gray = ImageOps.grayscale(canvas)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
        else:
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
        processed.append(arr)
    return np.asarray(processed, dtype=np.float32)
