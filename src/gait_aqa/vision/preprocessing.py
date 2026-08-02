"""Frame preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps


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
