"""Annotated video generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gait_aqa.data.video_io import draw_score_overlay, write_video


def annotate_video(
    frames: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    fps: float = 20.0,
) -> Path:
    """Write frames with score overlays."""
    annotated = draw_score_overlay(frames, scores)
    return write_video(output_path, annotated, fps)
