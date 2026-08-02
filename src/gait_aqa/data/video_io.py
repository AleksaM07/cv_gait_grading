"""Small video I/O layer with an `.npz` fallback for reproducible tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_video(path: str | Path) -> tuple[np.ndarray, float]:
    """Read a video-like file into RGB frames.

    `.npz` clips are first-class in the smoke workflow and store `frames` and
    `fps` arrays. MP4/AVI decoding is delegated to OpenCV when installed.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() == ".npz":
        with np.load(video_path) as data:
            frames = np.asarray(data["frames"], dtype=np.uint8)
            fps = float(np.asarray(data.get("fps", 20.0)).reshape(-1)[0])
        return _validate_frames(frames, video_path), fps

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"OpenCV is required to decode {video_path.suffix} files. "
            "Use `.npz` smoke clips or install opencv-python."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return _validate_frames(np.asarray(frames, dtype=np.uint8), video_path), fps


def write_video(path: str | Path, frames: np.ndarray, fps: float = 20.0) -> Path:
    """Write RGB frames.

    MP4 is used when OpenCV is installed. Without OpenCV, the function writes a
    `.npz` file at the requested stem and a GIF preview, then returns the `.npz`
    path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _validate_frames(np.asarray(frames, dtype=np.uint8), output_path)
    if output_path.suffix.lower() == ".npz":
        np.savez_compressed(output_path, frames=frames, fps=np.asarray([fps]))
        return output_path

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        fallback = output_path.with_suffix(".npz")
        np.savez_compressed(fallback, frames=frames, fps=np.asarray([fps]))
        write_gif(output_path.with_suffix(".gif"), frames, fps)
        return fallback

    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {output_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


def write_gif(path: str | Path, frames: np.ndarray, fps: float) -> Path:
    """Write a lightweight GIF preview."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(round(1000.0 / max(fps, 1.0)))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def draw_score_overlay(frames: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Draw simple score text and a red severity bar on frames."""
    annotated: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(image)
        score = float(scores[min(index, len(scores) - 1)])
        draw.rectangle((2, 2, 54, 16), fill=(0, 0, 0))
        draw.text((4, 4), f"{score:05.1f}", fill=(255, 255, 255))
        badness = int(round((100.0 - score) / 100.0 * (frame.shape[1] - 4)))
        draw.rectangle((2, frame.shape[0] - 6, 2 + badness, frame.shape[0] - 3), fill=(220, 20, 60))
        annotated.append(np.asarray(image, dtype=np.uint8))
    return np.asarray(annotated, dtype=np.uint8)


def _validate_frames(frames: np.ndarray, path: Path) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape T,H,W,3 in {path}")
    if frames.shape[0] < 2:
        raise ValueError(f"Video must contain at least two frames: {path}")
    return frames
