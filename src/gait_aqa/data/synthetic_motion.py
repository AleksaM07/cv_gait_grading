"""Synthetic gait-like videos for smoke tests and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gait_aqa.data.video_io import write_video
from gait_aqa.labels.score_components import score_from_severities


@dataclass(frozen=True)
class SyntheticSpec:
    """One synthetic clip specification."""

    clip_id: str
    policy_run: str
    checkpoint_step: int
    scenario: str
    seed: int
    camera: str
    sliding: float
    asymmetry: float
    jitter: float
    hopping: float
    micro_stepping: float
    toe_dragging: float
    fall: float
    command_ignoring: float


def generate_synthetic_dataset(
    output_dir: str | Path = "data/raw/synthetic",
    manifest_path: str | Path = "data/manifests/synthetic.csv",
    clip_count: int = 24,
    frame_count: int = 48,
    fps: float = 20.0,
    width: int = 96,
    height: int = 96,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset and return its manifest."""
    rng = np.random.default_rng(seed)
    output = Path(output_dir)
    telemetry_dir = output / "telemetry"
    output.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index in range(clip_count):
        severity = _sample_severity(index, clip_count, rng)
        spec = SyntheticSpec(
            clip_id=f"synthetic_{index:03d}",
            policy_run=f"synthetic_policy_{index // 4:02d}",
            checkpoint_step=1000 * (index // 3),
            scenario=("forward", "lateral", "turn", "diagonal")[index % 4],
            seed=100 + index,
            camera=("side", "front_oblique")[index % 2],
            sliding=severity[0],
            asymmetry=severity[1],
            jitter=severity[2],
            hopping=severity[3],
            micro_stepping=severity[4],
            toe_dragging=severity[5],
            fall=severity[6],
            command_ignoring=severity[7],
        )
        frames, telemetry = render_synthetic_clip(spec, frame_count, width, height)
        video_path = output / f"{spec.clip_id}.npz"
        telemetry_path = telemetry_dir / f"{spec.clip_id}.csv"
        write_video(video_path, frames, fps)
        telemetry.to_csv(telemetry_path, index=False)
        scores = score_from_severities(spec.__dict__)
        rows.append(
            {
                "clip_id": spec.clip_id,
                "video_path": str(video_path),
                "policy_run": spec.policy_run,
                "checkpoint_step": spec.checkpoint_step,
                "xml_model": "synthetic_rectangles",
                "scenario": spec.scenario,
                "seed": spec.seed,
                "camera": spec.camera,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": frame_count / fps,
                "split_group": f"{spec.policy_run}:{spec.checkpoint_step}",
                "overall_score": scores["overall_score"],
                "stability_score": scores["stability"],
                "contact_score": scores["foot_contact_quality"],
                "symmetry_score": scores["left_right_symmetry"],
                "periodicity_score": scores["periodicity"],
                "smoothness_score": scores["smoothness"],
                "tracking_score": scores["command_tracking"],
                "foot_sliding_label": int(spec.sliding >= 0.35),
                "hopping_label": int(spec.hopping >= 0.55),
                "micro_stepping_label": int(spec.micro_stepping >= 0.45),
                "asymmetry_label": int(spec.asymmetry >= 0.35),
                "torso_instability_label": int(spec.jitter >= 0.40),
                "toe_dragging_label": int(spec.toe_dragging >= 0.30),
                "fall_label": int(spec.fall >= 0.50),
                "command_ignoring_label": int(spec.command_ignoring >= 0.45),
                "telemetry_path": str(telemetry_path),
                "sliding_severity": spec.sliding,
                "asymmetry_severity": spec.asymmetry,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, index=False)
    return manifest


def render_synthetic_clip(
    spec: SyntheticSpec,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Render a simple walking-like rectangle figure."""
    rng = np.random.default_rng(spec.seed)
    frames = np.full((frame_count, height, width, 3), 235, dtype=np.uint8)
    telemetry_rows: list[dict[str, float]] = []
    base_x = width * 0.25
    stride = (0.50 - 0.35 * spec.micro_stepping) * width / frame_count
    torso_y = height * 0.42
    fall_drop = np.linspace(0.0, spec.fall * height * 0.35, frame_count)

    for t in range(frame_count):
        phase = 2.0 * np.pi * t / max(frame_count / 2.5, 1.0)
        x = base_x + stride * t
        y = torso_y + spec.hopping * 5.0 * np.sin(2.0 * phase) + fall_drop[t]
        x += rng.normal(0.0, spec.jitter * 1.8)
        y += rng.normal(0.0, spec.jitter * 1.8)

        left_amp = 13.0 * (1.0 + spec.asymmetry)
        right_amp = 13.0 * (1.0 - spec.asymmetry)
        left_leg = left_amp * np.sin(phase)
        right_leg = right_amp * np.sin(phase + np.pi)
        slide_offset = spec.sliding * 8.0 * t / frame_count
        toe_drop = spec.toe_dragging * 5.0

        frame = frames[t]
        _rect(frame, x - 7, y - 17, x + 7, y + 8, (55, 82, 120))
        _rect(frame, x - 4, y - 30, x + 4, y - 18, (80, 100, 135))
        _limb(frame, x - 4, y + 7, x - 10 + left_leg + slide_offset, y + 28 + toe_drop, (20, 120, 160))
        _limb(frame, x + 4, y + 7, x + 10 + right_leg - slide_offset, y + 28, (170, 70, 55))
        _rect(frame, 0, height - 10, width, height - 8, (90, 90, 90))
        telemetry_rows.append(
            {
                "frame": float(t),
                "time_seconds": float(t / 20.0),
                "foot_sliding": spec.sliding,
                "left_right_asymmetry": spec.asymmetry,
                "torso_instability": spec.jitter,
                "hopping": spec.hopping,
                "micro_stepping": spec.micro_stepping,
                "toe_dragging": spec.toe_dragging,
                "fall_or_near_fall": spec.fall,
                "command_ignoring": spec.command_ignoring,
            }
        )
    return frames, pd.DataFrame(telemetry_rows)


def _sample_severity(index: int, clip_count: int, rng: np.random.Generator) -> np.ndarray:
    ramp = index / max(clip_count - 1, 1)
    values = rng.beta(1.2 + 2.0 * ramp, 3.0, size=8)
    values[index % 8] = ramp
    return np.clip(values, 0.0, 1.0)


def _rect(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    xa, xb = sorted((int(round(x0)), int(round(x1))))
    ya, yb = sorted((int(round(y0)), int(round(y1))))
    frame[max(0, ya):min(h, yb), max(0, xa):min(w, xb)] = color


def _limb(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for i in range(steps + 1):
        alpha = i / steps
        x = (1 - alpha) * x0 + alpha * x1
        y = (1 - alpha) * y0 + alpha * y1
        _rect(frame, x - 2, y - 2, x + 2, y + 2, color)
