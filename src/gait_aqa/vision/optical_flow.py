"""Dense optical-flow extraction."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np

from gait_aqa.vision.alignment import remove_global_translation


def compute_dense_flow(
    gray_frames: np.ndarray,
    mode: str = "body_centered_residual_flow",
) -> np.ndarray:
    """Compute dense flow with OpenCV Farneback or a deterministic fallback."""
    if gray_frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape T,H,W")
    if gray_frames.shape[0] < 2:
        raise ValueError("At least two frames are required for optical flow")
    flow = _opencv_farneback(gray_frames)
    if flow is None:
        flow = _fallback_dense_flow(gray_frames)
    if mode == "body_centered_residual_flow":
        flow = remove_global_translation(flow)
    elif mode != "absolute_flow":
        raise ValueError(f"Unknown flow mode: {mode}")
    return flow.astype(np.float32)


def cache_flow(path: str | Path, flow: np.ndarray) -> Path:
    """Save flow as compressed `.npz`."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, flow=flow)
    return output


def load_cached_flow(path: str | Path) -> np.ndarray:
    """Load cached flow."""
    with np.load(Path(path)) as data:
        return np.asarray(data["flow"], dtype=np.float32)


def _opencv_farneback(gray_frames: np.ndarray) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        return None
    frames_u8 = np.clip(gray_frames * 255.0, 0, 255).astype(np.uint8)
    flows: list[np.ndarray] = []
    for previous, current in pairwise(frames_u8):
        initial_flow = np.zeros((*previous.shape, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            initial_flow,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)
    return np.asarray(flows, dtype=np.float32)


def _fallback_dense_flow(gray_frames: np.ndarray) -> np.ndarray:
    """Approximate dense flow for tests when OpenCV is unavailable.

    It estimates global foreground centroid motion and distributes it over
    pixels that differ from the bright background. This is intentionally simple
    and deterministic; the documented baseline should use OpenCV Farneback.
    """
    flows: list[np.ndarray] = []
    centroids = [_foreground_centroid(frame) for frame in gray_frames]
    for index, (previous, current) in enumerate(pairwise(gray_frames)):
        dy, dx = np.gradient(current)
        diff = current - previous
        centroid_delta = centroids[index + 1] - centroids[index]
        mask = (current < 0.88).astype(np.float32)
        flow_x = mask * centroid_delta[0] + diff * dx * 4.0
        flow_y = mask * centroid_delta[1] + diff * dy * 4.0
        flows.append(np.stack([flow_x, flow_y], axis=-1))
    return np.asarray(flows, dtype=np.float32)


def _foreground_centroid(frame: np.ndarray) -> np.ndarray:
    mask = frame < 0.88
    if not mask.any():
        return np.asarray([0.0, 0.0], dtype=np.float32)
    y, x = np.nonzero(mask)
    return np.asarray([x.mean(), y.mean()], dtype=np.float32)
