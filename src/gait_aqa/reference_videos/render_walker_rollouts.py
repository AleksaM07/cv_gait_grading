"""Render a leakage-safe video dataset from successful walker policies.

The external walker checkout remains the runtime owner of environments and
checkpoints. This module owns deterministic selection, terminal-safe rollout
capture, camera placement, H.264 encoding, labels, and manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from gait_aqa.logging_utils import get_logger, setup_logging

DEFAULT_RUNTIME_REPO = Path("_references/mujoco-bipedal-joystick-walker")
DEFAULT_OUTPUT_DIR = Path("MUJOCO_videos_better")
FALL_HEIGHT_RATIO = 0.60
FALL_TORSO_UP_THRESHOLD = 0.25
MIN_RENDER_FRAMES = 2
REFERENCE_MODEL_EXTENT_M = 2.2
REFERENCE_LOOKAT_FLOOR_M = 0.70
MIN_CAMERA_DISTANCE_M = 0.8
MAX_CAMERA_DISTANCE_M = 6.5
EVAL_MARKER = "eval |"
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


@dataclass(frozen=True)
class Scenario:
    """One deterministic joystick command and reset seed."""

    name: str
    command_x: float
    command_y: float
    command_yaw: float
    seed: int

    @property
    def command(self) -> np.ndarray:
        """Return the command vector expected by walker environments."""
        return np.asarray(
            [self.command_x, self.command_y, self.command_yaw],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class CameraPreset:
    """A stable free-camera view used for every policy."""

    name: str
    distance: float
    azimuth: float
    elevation: float


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("forward_slow", 0.15, 0.0, 0.0, 7),
    Scenario("forward", 0.35, 0.0, 0.0, 11),
    Scenario("forward_fast", 0.65, 0.0, 0.0, 13),
    Scenario("lateral", 0.0, 0.20, 0.0, 17),
    Scenario("turn", 0.0, 0.0, 0.35, 19),
    Scenario("diagonal", 0.25, 0.15, 0.25, 23),
)

CAMERAS: tuple[CameraPreset, ...] = (
    CameraPreset("side", 4.2, 90.0, -15.0),
    CameraPreset("front_oblique", 4.2, 160.0, -20.0),
)


def _scaled_camera_distance(reference_distance: float, model_extent: float) -> float:
    """Scale framing to the physical model while preserving the chosen view."""
    if not math.isfinite(model_extent) or model_extent <= 0.0:
        model_extent = REFERENCE_MODEL_EXTENT_M
    scaled_distance = reference_distance * model_extent / REFERENCE_MODEL_EXTENT_M
    return float(np.clip(scaled_distance, MIN_CAMERA_DISTANCE_M, MAX_CAMERA_DISTANCE_M))


def _scaled_lookat_floor(model_extent: float) -> float:
    """Keep vertical framing proportional across differently sized robots."""
    if not math.isfinite(model_extent) or model_extent <= 0.0:
        model_extent = REFERENCE_MODEL_EXTENT_M
    return REFERENCE_LOOKAT_FLOOR_M * model_extent / REFERENCE_MODEL_EXTENT_M


def _safe_name(value: str, maximum_length: int = 80) -> str:
    normalized = SAFE_NAME_PATTERN.sub("_", value.strip()).strip("_").lower()
    return (normalized or "unnamed")[:maximum_length]


def _parse_training_rewards(log_path: Path) -> dict[int, float]:
    """Read checkpoint-aligned evaluation rewards from a training log."""
    if not log_path.exists():
        return {}
    rewards: dict[int, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if EVAL_MARKER not in line:
            continue
        fields: dict[str, str] = {}
        for part in line.split(EVAL_MARKER, maxsplit=1)[1].split("|"):
            if "=" not in part:
                continue
            name, raw_value = part.strip().split("=", maxsplit=1)
            fields[name] = raw_value.strip()
        try:
            step = int(float(fields["step"]))
            reward = float(fields["reward"])
        except (KeyError, ValueError):
            continue
        if math.isfinite(reward):
            rewards[step] = reward
    return rewards


def _policy_type(run_name: str) -> str:
    return "berkeley" if run_name.startswith("ppo_Berkeley") else "biomechanics"


def _policy_id(index: int, run_name: str, run_config: dict[str, Any]) -> str:
    env_config = run_config.get("env", {})
    if _policy_type(run_name) == "berkeley":
        suffix = "berkeley_flat"
    else:
        reference = env_config.get("reference_gait") or "no_ref"
        profile = env_config.get("command_profile") or "standard"
        xml_path = str(env_config.get("xml_path") or "")
        xml_match = re.search(r"trainfast_v(\d+)", xml_path)
        xml_version = f"v{xml_match.group(1)}" if xml_match else "auto_xml"
        suffix = f"{xml_version}_{profile}_{reference}"
    return f"P{index:02d}_{_safe_name(suffix, maximum_length=48)}"


def discover_policies(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Select the best logged checkpoint from every successful run."""
    root = Path(runs_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Successful-runs directory not found: {root}")
    policies: list[dict[str, Any]] = []
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for index, run_dir in enumerate(run_dirs, start=1):
        checkpoint_root = run_dir / "checkpoints"
        checkpoints = sorted(
            (
                path
                for path in checkpoint_root.glob("*")
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )
        if not checkpoints:
            continue
        config_path = run_dir / "config.json"
        run_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        checkpoint_by_step = {int(path.name): path.resolve() for path in checkpoints}
        rewards = _parse_training_rewards(run_dir / "train.log")
        eligible_rewards = {
            step: reward
            for step, reward in rewards.items()
            if step in checkpoint_by_step
        }
        if eligible_rewards:
            checkpoint_step = max(
                eligible_rewards,
                key=lambda step: (eligible_rewards[step], step),
            )
            training_reward: float | None = eligible_rewards[checkpoint_step]
            selection_reason = "max_logged_eval_reward"
        else:
            checkpoint_step = max(checkpoint_by_step)
            training_reward = None
            selection_reason = "latest_checkpoint_fallback"
        env_config = run_config.get("env", {})
        policies.append(
            {
                "policy_id": _policy_id(index, run_dir.name, run_config),
                "run_name": run_dir.name,
                "policy_type": _policy_type(run_dir.name),
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": str(checkpoint_by_step[checkpoint_step]),
                "training_reward": training_reward,
                "selection_reason": selection_reason,
                "command_profile": str(env_config.get("command_profile", "standard")),
                "reference_gait": str(env_config.get("reference_gait", "none")),
                "run_config_path": str(config_path.resolve()),
            }
        )
    if not policies:
        raise ValueError(f"No checkpoint-bearing runs found under {root}")
    return policies


def build_rollout_tasks(
    policies: Iterable[dict[str, Any]],
    output_dir: str | Path,
    scenarios: Iterable[Scenario] = SCENARIOS,
    cameras: Iterable[CameraPreset] = CAMERAS,
) -> list[dict[str, Any]]:
    """Build unique rollout records; camera views share one rollout identity."""
    root = Path(output_dir).resolve()
    tasks: list[dict[str, Any]] = []
    seen_rollouts: set[str] = set()
    seen_clips: set[str] = set()
    camera_rows = [asdict(camera) for camera in cameras]
    for policy in policies:
        for scenario in scenarios:
            rollout_id = (
                f"{policy['policy_id']}__ckpt{int(policy['checkpoint_step']):012d}"
                f"__{scenario.name}__seed{scenario.seed}"
            )
            if rollout_id in seen_rollouts:
                raise ValueError(f"Duplicate rollout identity: {rollout_id}")
            seen_rollouts.add(rollout_id)
            video_paths: dict[str, str] = {}
            for camera in camera_rows:
                clip_id = f"{rollout_id}__{camera['name']}"
                if clip_id in seen_clips:
                    raise ValueError(f"Duplicate clip identity: {clip_id}")
                seen_clips.add(clip_id)
                video_paths[camera["name"]] = str(
                    root / str(policy["policy_id"]) / scenario.name / f"{clip_id}.mp4"
                )
            tasks.append(
                {
                    **policy,
                    "rollout_id": rollout_id,
                    "scenario": scenario.name,
                    "command_x": scenario.command_x,
                    "command_y": scenario.command_y,
                    "command_yaw": scenario.command_yaw,
                    "seed": scenario.seed,
                    "cameras": camera_rows,
                    "video_paths": video_paths,
                }
            )
    return tasks


def _scenario_is_in_distribution(command_profile: str, task: dict[str, Any]) -> bool:
    if command_profile in {"forward", "forward_slow", "walk"}:
        return (
            float(task["command_x"]) >= 0.0
            and float(task["command_y"]) == 0.0
            and float(task["command_yaw"]) == 0.0
        )
    return True


def _load_walker_analysis(runtime_repo: Path) -> ModuleType:
    """Load the external runtime without vendoring its environment or policy."""
    module_path = runtime_repo / "walking_analysis.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Walker analysis runtime not found: {module_path}")
    sys.path.insert(0, str(runtime_repo))
    spec = importlib.util.spec_from_file_location(
        "gait_aqa_walker_runtime", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import walker runtime: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state_snapshot(model: Any, state: Any) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(state.data.qpos, dtype=np.float64).reshape(-1)
    qvel = np.asarray(state.data.qvel, dtype=np.float64).reshape(-1)
    if qpos.size < model.nq or qvel.size < model.nv:
        raise ValueError(
            f"State/model mismatch: qpos={qpos.size}/{model.nq}, "
            f"qvel={qvel.size}/{model.nv}"
        )
    return qpos[: model.nq].copy(), qvel[: model.nv].copy()


def _torso_up(state: Any, torso_body_id: int, policy_type: str) -> float:
    if torso_body_id < 0:
        return float("nan")
    rotation = np.asarray(state.data.xmat[torso_body_id], dtype=np.float64).reshape(
        3, 3
    )
    torso_axis = 1 if policy_type == "biomechanics" else 2
    return float(rotation[2, torso_axis])


def _measured_command(
    state: Any,
    torso_body_id: int,
    policy_type: str,
) -> np.ndarray:
    qvel = np.asarray(state.data.qvel, dtype=np.float64)
    if torso_body_id >= 0:
        rotation = np.asarray(state.data.xmat[torso_body_id], dtype=np.float64).reshape(
            3, 3
        )
    else:
        rotation = np.eye(3)
    local_linear_velocity = rotation.T @ qvel[:3]
    local_angular_velocity = rotation.T @ qvel[3:6]
    if policy_type == "biomechanics":
        return np.asarray(
            [
                local_linear_velocity[0],
                local_linear_velocity[2],
                local_angular_velocity[1],
            ]
        )
    return np.asarray(
        [
            local_linear_velocity[0],
            local_linear_velocity[1],
            local_angular_velocity[2],
        ]
    )


def _scalar_metric(metrics: Any, name: str) -> float:
    if not isinstance(metrics, dict) or name not in metrics:
        return float("nan")
    try:
        value = np.asarray(metrics[name])
    except (TypeError, ValueError):
        return float("nan")
    if value.size != 1:
        return float("nan")
    return float(value.reshape(-1)[0])


def _termination_reason(
    env_done: bool,
    root_height: float,
    minimum_height: float,
    torso_up: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        reasons.append("invalid_state")
    if math.isfinite(root_height) and root_height < minimum_height:
        reasons.append("low_height")
    if math.isfinite(torso_up) and torso_up < FALL_TORSO_UP_THRESHOLD:
        reasons.append("tipped")
    if env_done and not reasons:
        reasons.append("env_done")
    return bool(reasons), "+".join(reasons) if reasons else "none"


def _summarize_rollout(
    task: dict[str, Any],
    step_rows: list[dict[str, float]],
    requested_steps: int,
    dt: float,
    snapshots: list[tuple[np.ndarray, np.ndarray]],
    terminated: bool,
    termination_reason: str,
) -> dict[str, Any]:
    metrics = pd.DataFrame(step_rows)
    actual_steps = len(metrics)
    if actual_steps == 0:
        raise RuntimeError("Rollout produced no control steps")
    survival = min(actual_steps / max(requested_steps, 1), 1.0)
    return {
        "rollout_id": task["rollout_id"],
        "policy_id": task["policy_id"],
        "run_name": task["run_name"],
        "policy_type": task["policy_type"],
        "checkpoint_step": int(task["checkpoint_step"]),
        "checkpoint_path": task["checkpoint_path"],
        "training_reward": task.get("training_reward"),
        "selection_reason": task["selection_reason"],
        "command_profile": task["command_profile"],
        "reference_gait": task["reference_gait"],
        "scenario": task["scenario"],
        "scenario_in_distribution": _scenario_is_in_distribution(
            str(task["command_profile"]), task
        ),
        "command_x": float(task["command_x"]),
        "command_y": float(task["command_y"]),
        "command_yaw": float(task["command_yaw"]),
        "seed": int(task["seed"]),
        "requested_steps": requested_steps,
        "steps": actual_steps,
        "simulated_seconds": actual_steps * dt,
        "frames": len(snapshots),
        "terminated": int(terminated),
        "ended_done": int(terminated),
        "resets": 0,
        "termination_reason": termination_reason,
        "fall_label": int(
            any(
                token in termination_reason
                for token in ("low_height", "tipped", "invalid_state", "env_done")
            )
        ),
        "mean_first_fall_survival_fraction": survival,
        "total_reward": float(metrics["reward"].sum()),
        "mean_reward": float(metrics["reward"].mean()),
        "tracking_rmse": float(
            np.sqrt(np.mean(np.square(metrics["command_error_norm"])))
        ),
        "command_failure_rate": float(np.mean(metrics["command_error_norm"] > 0.25)),
        "mean_torso_up": float(metrics["torso_up"].mean()),
        "min_torso_up": float(metrics["torso_up"].min()),
        "mean_root_height": float(metrics["root_height"].mean()),
        "min_root_height": float(metrics["root_height"].min()),
        "mean_action_norm": float(metrics["action_norm"].mean()),
        "mean_action_rate_norm": float(metrics["action_rate_norm"].mean()),
        "mean_foot_slip_speed": float(metrics["foot_slip"].mean()),
    }


def _rollout_once(
    runtime: ModuleType,
    loaded: Any,
    metadata: Any,
    task: dict[str, Any],
    duration_seconds: float,
    fps: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    import jax

    command = np.asarray(
        [task["command_x"], task["command_y"], task["command_yaw"]],
        dtype=np.float32,
    )
    if loaded.policy_type == "biomechanics":
        runtime.biomechanics_eval.clip_command(command)
    else:
        runtime.berkeley_eval.clip_command(command)

    rng = jax.random.PRNGKey(int(task["seed"]))
    state = loaded.reset_fn(loaded.env, rng, command)
    model = loaded.env.mj_model
    dt = float(loaded.env.dt)
    requested_steps = max(1, math.ceil(duration_seconds / dt))
    initial_qpos = np.asarray(state.data.qpos, dtype=np.float64)
    minimum_height = FALL_HEIGHT_RATIO * float(initial_qpos[2])
    snapshots = [_state_snapshot(model, state)]
    last_frame_index = 0
    previous_action: np.ndarray | None = None
    step_rows: list[dict[str, float]] = []
    terminated = False
    termination_reason = "none"

    for step in range(requested_steps):
        rng, action_key = jax.random.split(rng)
        next_state, action = loaded.step_fn(
            loaded.env,
            loaded.policy,
            state,
            action_key,
            command,
        )
        jax.block_until_ready(action)
        qpos = np.asarray(next_state.data.qpos, dtype=np.float64)
        qvel = np.asarray(next_state.data.qvel, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        torso_up = _torso_up(next_state, metadata.torso_body_id, loaded.policy_type)
        measured = _measured_command(
            next_state,
            metadata.torso_body_id,
            loaded.policy_type,
        )
        action_rate = (
            0.0
            if previous_action is None
            else float(np.linalg.norm(action_array - previous_action) / dt)
        )
        step_rows.append(
            {
                "reward": float(np.asarray(next_state.reward)),
                "command_error_norm": float(np.linalg.norm(measured - command)),
                "torso_up": torso_up,
                "root_height": float(qpos[2]),
                "action_norm": float(np.linalg.norm(action_array)),
                "action_rate_norm": action_rate,
                "foot_slip": _scalar_metric(next_state.metrics, "foot_slip"),
            }
        )
        previous_action = action_array
        terminated, termination_reason = _termination_reason(
            bool(np.asarray(next_state.done)),
            float(qpos[2]),
            minimum_height,
            torso_up,
            qpos,
            qvel,
        )
        if terminated:
            # Record terminal metrics, but never render the terminal/fallen pose.
            break
        state = next_state
        simulated_time = (step + 1) * dt
        frame_index = math.floor(simulated_time * fps + 1e-9)
        if frame_index > last_frame_index:
            snapshots.append(_state_snapshot(model, state))
            last_frame_index = frame_index

    summary = _summarize_rollout(
        task,
        step_rows,
        requested_steps,
        dt,
        snapshots,
        terminated,
        termination_reason,
    )
    return snapshots, summary


class _FfmpegWriter:
    """Atomic raw-RGB to H.264 stream."""

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("FFmpeg is required for rollout rendering")
        self.path = path
        self.partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path.unlink(missing_ok=True)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.partial_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            self.partial_path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg failed ({return_code}): {stderr.strip()}")
        self.partial_path.replace(self.path)

    def abort(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        self.process.kill()
        self.process.wait()
        self.partial_path.unlink(missing_ok=True)


def _render_views(
    model: Any,
    snapshots: list[tuple[np.ndarray, np.ndarray]],
    cameras: list[dict[str, Any]],
    video_paths: dict[str, str],
    width: int,
    height: int,
    fps: int,
    crf: int,
    overwrite: bool,
) -> dict[str, float]:
    import mujoco

    model_extent = float(model.stat.extent)
    lookat_floor = _scaled_lookat_floor(model_extent)
    camera_distances = {
        str(preset["name"]): _scaled_camera_distance(
            float(preset["distance"]), model_extent
        )
        for preset in cameras
    }
    active_cameras: list[tuple[Any, Any, _FfmpegWriter]] = []
    for preset in cameras:
        path = Path(video_paths[str(preset["name"])])
        if _video_is_valid(path) and not overwrite:
            continue
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = camera_distances[str(preset["name"])]
        camera.azimuth = float(preset["azimuth"])
        camera.elevation = float(preset["elevation"])
        writer = _FfmpegWriter(path, width, height, fps, crf)
        active_cameras.append((renderer, camera, writer))
    if not active_cameras:
        return camera_distances

    data = mujoco.MjData(model)
    try:
        for qpos, qvel in snapshots:
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            root = np.asarray(data.qpos[:3], dtype=np.float64)
            lookat = np.asarray([root[0], root[1], max(lookat_floor, root[2] * 0.55)])
            for renderer, camera, writer in active_cameras:
                camera.lookat[:] = lookat
                renderer.update_scene(data, camera=camera)
                writer.write(renderer.render())
        for _renderer, _camera, writer in active_cameras:
            writer.close()
    except Exception:
        for _renderer, _camera, writer in active_cameras:
            if writer.process.poll() is None:
                writer.abort()
        raise
    finally:
        for renderer, _camera, _writer in active_cameras:
            renderer.close()
    return camera_distances


def _worker(task_file: Path) -> int:
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    runtime_repo = Path(payload["runtime_repo"]).resolve()
    runtime = _load_walker_analysis(runtime_repo)
    policy = payload["policy"]
    loaded = runtime.load_policy(pd.Series(policy))
    metadata = runtime.build_model_metadata(loaded.env.mj_model)
    results: list[dict[str, Any]] = []
    logger = get_logger(__name__)

    for task in payload["tasks"]:
        started = time.perf_counter()
        try:
            snapshots, summary = _rollout_once(
                runtime,
                loaded,
                metadata,
                task,
                float(payload["duration_seconds"]),
                int(payload["fps"]),
            )
            if len(snapshots) < MIN_RENDER_FRAMES:
                raise RuntimeError(
                    f"Only {len(snapshots)} safe frames before termination"
                )
            camera_distances = _render_views(
                loaded.env.mj_model,
                snapshots,
                task["cameras"],
                task["video_paths"],
                int(payload["width"]),
                int(payload["height"]),
                int(payload["fps"]),
                int(payload["crf"]),
                bool(payload["overwrite"]),
            )
            for camera in task["cameras"]:
                name = str(camera["name"])
                results.append(
                    {
                        **summary,
                        "clip_id": f"{task['rollout_id']}__{name}",
                        "camera": name,
                        "camera_distance_m": camera_distances[name],
                        "video_path": task["video_paths"][name],
                        "fps": int(payload["fps"]),
                        "width": int(payload["width"]),
                        "height": int(payload["height"]),
                        "status": "recorded",
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": "",
                    }
                )
            logger.info(
                "Recorded rollout={} frames={} terminated={} reason={}",
                task["rollout_id"],
                len(snapshots),
                summary["terminated"],
                summary["termination_reason"],
            )
        except Exception as exc:  # one bad scenario must not discard a policy
            logger.exception("Failed rollout={}", task["rollout_id"])
            for camera in task["cameras"]:
                name = str(camera["name"])
                results.append(
                    _failure_video_row(
                        task,
                        name,
                        int(payload["fps"]),
                        int(payload["width"]),
                        int(payload["height"]),
                        f"{type(exc).__name__}: {exc}",
                        time.perf_counter() - started,
                    )
                )
    result_path = Path(payload["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def _failure_video_row(
    task: dict[str, Any],
    camera: str,
    fps: int,
    width: int,
    height: int,
    error: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build one flat failure row without serializing nested task structures."""
    scalar_keys = (
        "rollout_id",
        "policy_id",
        "run_name",
        "policy_type",
        "checkpoint_step",
        "checkpoint_path",
        "training_reward",
        "selection_reason",
        "command_profile",
        "reference_gait",
        "scenario",
        "command_x",
        "command_y",
        "command_yaw",
        "seed",
    )
    return {
        **{key: task.get(key) for key in scalar_keys},
        "clip_id": f"{task['rollout_id']}__{camera}",
        "camera": camera,
        "video_path": task["video_paths"][camera],
        "fps": fps,
        "width": width,
        "height": height,
        "status": "failed",
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }


def _min_max_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    minimum = finite.min()
    maximum = finite.max()
    if pd.isna(minimum) or pd.isna(maximum) or math.isclose(minimum, maximum):
        score = pd.Series(1.0, index=series.index)
    else:
        score = (finite - minimum) / (maximum - minimum)
    return score if higher_is_better else 1.0 - score


def apply_reference_scores(manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply the upstream scenario-wise score formula without view duplication."""
    result = manifest.copy()
    score_columns = [
        "score_stability",
        "score_tracking",
        "score_upright",
        "score_smoothness",
        "composite_score",
    ]
    result[score_columns] = np.nan
    recorded = result[result["status"] == "recorded"].drop_duplicates("rollout_id")
    scored_groups: list[pd.DataFrame] = []
    for _, group in recorded.groupby("scenario", sort=False):
        group = group.copy()
        group["score_stability"] = _min_max_score(
            group["mean_first_fall_survival_fraction"], True
        )
        group["score_tracking"] = _min_max_score(group["tracking_rmse"], False)
        group["score_upright"] = _min_max_score(group["mean_torso_up"], True)
        group["score_smoothness"] = _min_max_score(
            group["mean_action_rate_norm"], False
        )
        group["composite_score"] = (
            0.40 * group["score_stability"]
            + 0.30 * group["score_tracking"]
            + 0.20 * group["score_upright"]
            + 0.10 * group["score_smoothness"]
        )
        scored_groups.append(group[["rollout_id", *score_columns]])
    if scored_groups:
        scores = pd.concat(scored_groups, ignore_index=True)
        result = result.drop(columns=score_columns).merge(
            scores, on="rollout_id", how="left", validate="many_to_one"
        )
    cohort_payload = "\n".join(sorted(recorded["rollout_id"].astype(str)))
    result["label_cohort_sha256"] = hashlib.sha256(
        cohort_payload.encode("utf-8")
    ).hexdigest()
    result["label_formula"] = (
        "scenario min-max: 0.40 stability + 0.30 tracking + "
        "0.20 upright + 0.10 smoothness"
    )
    result["split_group"] = (
        result["policy_id"].astype(str)
        + ":"
        + pd.to_numeric(result["checkpoint_step"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    return result.sort_values(
        ["policy_id", "scenario", "seed", "camera"], na_position="last"
    ).reset_index(drop=True)


def _video_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return True
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    manifest = apply_reference_scores(pd.DataFrame(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    manifest.to_csv(temporary, index=False)
    temporary.replace(path)
    return manifest


def _planned_video_rows(tasks: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for camera in task["cameras"]:
            name = str(camera["name"])
            rows.append(
                {
                    "rollout_id": task["rollout_id"],
                    "clip_id": f"{task['rollout_id']}__{name}",
                    "policy_id": task["policy_id"],
                    "run_name": task["run_name"],
                    "checkpoint_step": task["checkpoint_step"],
                    "checkpoint_path": task["checkpoint_path"],
                    "scenario": task["scenario"],
                    "seed": task["seed"],
                    "camera": name,
                    "video_path": task["video_paths"][name],
                }
            )
    return pd.DataFrame(rows)


def _run_batch(args: argparse.Namespace) -> int:
    logger = setup_logging(args.log_file, args.log_level)
    runtime_repo = args.runtime_repo.resolve()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    policies = discover_policies(runs_dir)
    if args.limit_policies is not None:
        policies = policies[: args.limit_policies]
    scenarios: tuple[Scenario, ...] = SCENARIOS
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]
    tasks = build_rollout_tasks(policies, output_dir, scenarios=scenarios)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _planned_video_rows(tasks)
    plan.to_csv(output_dir / "render_plan.csv", index=False)
    logger.info(
        "Planned policies={} rollouts={} videos={} output={}",
        len(policies),
        len(tasks),
        len(plan),
        output_dir,
    )
    if args.dry_run:
        print(plan.to_string(index=False))
        return 0

    manifest_path = output_dir / "manifest.csv"
    previous = (
        pd.read_csv(manifest_path).to_dict(orient="records")
        if manifest_path.exists()
        else []
    )
    previous_by_rollout: dict[str, list[dict[str, Any]]] = {}
    for row in previous:
        previous_by_rollout.setdefault(str(row["rollout_id"]), []).append(row)
    planned_ids = {str(task["rollout_id"]) for task in tasks}
    rows = [row for row in previous if str(row["rollout_id"]) in planned_ids]
    completed_ids = {
        rollout_id
        for rollout_id, group in previous_by_rollout.items()
        if len(group) == len(CAMERAS)
        and all(
            row.get("status") == "recorded"
            and _video_is_valid(Path(str(row["video_path"])))
            for row in group
        )
    }

    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    for policy_index, policy in enumerate(policies, start=1):
        policy_tasks = [
            task
            for task in tasks
            if task["policy_id"] == policy["policy_id"]
            and (args.overwrite or task["rollout_id"] not in completed_ids)
        ]
        if not policy_tasks:
            logger.info("Skipping completed policy={}", policy["policy_id"])
            continue
        task_path = work_dir / f"{policy['policy_id']}.json"
        result_path = work_dir / f"{policy['policy_id']}__result.json"
        task_payload = {
            "runtime_repo": str(runtime_repo),
            "policy": policy,
            "tasks": policy_tasks,
            "duration_seconds": args.duration,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "crf": args.crf,
            "overwrite": args.overwrite,
            "result_path": str(result_path),
        }
        task_path.write_text(json.dumps(task_payload, indent=2), encoding="utf-8")
        result_path.unlink(missing_ok=True)
        logger.info(
            "Rendering policy {}/{}: {} rollouts={}",
            policy_index,
            len(policies),
            policy["policy_id"],
            len(policy_tasks),
        )
        environment = os.environ.copy()
        source_root = str((Path(__file__).resolve().parents[2]).resolve())
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [source_root, environment.get("PYTHONPATH", "")])
        )
        environment["JAX_PLATFORM_NAME"] = "cpu"
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        environment["JAX_COMPILATION_CACHE_DIR"] = str(work_dir / "jax_cache")
        log_path = work_dir / f"{policy['policy_id']}.log"
        command = [
            sys.executable,
            "-m",
            "gait_aqa.reference_videos.render_walker_rollouts",
            "--worker-task",
            str(task_path),
        ]
        process = subprocess.run(
            command,
            cwd=runtime_repo,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(
            process.stdout + "\n--- STDERR ---\n" + process.stderr,
            encoding="utf-8",
        )
        policy_ids = {str(task["rollout_id"]) for task in policy_tasks}
        rows = [row for row in rows if str(row["rollout_id"]) not in policy_ids]
        if process.returncode == 0 and result_path.exists():
            rows.extend(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            error = (
                process.stderr.strip()
                or process.stdout.strip()
                or f"worker exit code {process.returncode}"
            )
            logger.error("Policy failed={} error={}", policy["policy_id"], error[-500:])
            for task in policy_tasks:
                for camera in task["cameras"]:
                    name = str(camera["name"])
                    rows.append(
                        _failure_video_row(
                            task,
                            name,
                            args.fps,
                            args.width,
                            args.height,
                            error[-2000:],
                        )
                    )
        manifest = _write_manifest(manifest_path, rows)
        recorded = int((manifest["status"] == "recorded").sum())
        logger.info("Progress recorded={}/{}", recorded, len(plan))

    manifest = _write_manifest(manifest_path, rows)
    recorded = manifest[manifest["status"] == "recorded"]
    duplicate_clips = int(recorded["clip_id"].duplicated().sum())
    duplicate_paths = int(recorded["video_path"].duplicated().sum())
    invalid_videos = sum(
        not _video_is_valid(Path(str(path))) for path in recorded["video_path"]
    )
    rollout_rows = recorded.drop_duplicates("rollout_id")
    termination_counts = {
        str(reason): int(count)
        for reason, count in rollout_rows["termination_reason"].value_counts().items()
    }
    summary = {
        "policies": len(policies),
        "rollouts": len(tasks),
        "planned_videos": len(plan),
        "recorded_videos": len(recorded),
        "failed_videos": int((manifest["status"] != "recorded").sum()),
        "duplicate_clip_ids": duplicate_clips,
        "duplicate_video_paths": duplicate_paths,
        "invalid_videos": invalid_videos,
        "termination_reason_rollouts": termination_counts,
        "min_simulated_seconds": float(recorded["simulated_seconds"].min()),
        "max_simulated_seconds": float(recorded["simulated_seconds"].max()),
        "cameras": [camera.name for camera in CAMERAS],
        "scenarios": [scenario.name for scenario in scenarios],
        "camera_framing": "distance and look-at height scaled by model extent",
        "terminal_frame_policy": "terminal metrics retained; terminal pose not rendered",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not invalid_videos else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the public batch-renderer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-repo", type=Path, default=DEFAULT_RUNTIME_REPO)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNTIME_REPO / "runs" / "successful",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--limit-policies", type=int)
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "render.log",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the selected successful-policy dataset."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_task is not None:
        setup_logging(args.worker_task.with_suffix(".worker.log"), "INFO")
        return _worker(args.worker_task.resolve())
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--fps, --width, and --height must be positive")
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
