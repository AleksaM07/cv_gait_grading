"""Retarget CMU BVH motion onto a MuJoCo model and render side-view MP4s.

The playback is deliberately kinematic: BVH poses are written to ``qpos`` and
MuJoCo performs forward kinematics and rendering. This is the correct mode for
reference-motion videos because a controller or contact instability cannot
distort the source motion.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from gait_aqa.reference_videos.bvh import BvhMotion, load_bvh

LOGGER = logging.getLogger(__name__)

# CMU: X=lateral, Y=up, Z=forward. Model: X=forward, Y=lateral, Z=up.
CMU_TO_MUJOCO = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

SOURCE_BODY_JOINTS = {
    "thorax": "Chest2",
    "head": "Head",
    "abdomen": "Chest",
    "pelvis": "Hips",
    "left_thigh": "LeftHip",
    "left_shank": "LeftKnee",
    "left_foot": "LeftAnkle",
    "right_thigh": "RightHip",
    "right_shank": "RightKnee",
    "right_foot": "RightAnkle",
    "left_upper_arm": "LeftShoulder",
    "left_forearm": "LeftElbow",
    "left_hand": "LeftWrist",
    "right_upper_arm": "RightShoulder",
    "right_forearm": "RightElbow",
    "right_hand": "RightWrist",
}

BODY_PARENTS = {
    "head": "thorax",
    "abdomen": "thorax",
    "pelvis": "abdomen",
    "left_thigh": "pelvis",
    "left_shank": "left_thigh",
    "left_foot": "left_shank",
    "right_thigh": "pelvis",
    "right_shank": "right_thigh",
    "right_foot": "right_shank",
    "left_upper_arm": "thorax",
    "left_forearm": "left_upper_arm",
    "left_hand": "left_forearm",
    "right_upper_arm": "thorax",
    "right_forearm": "right_upper_arm",
    "right_hand": "right_forearm",
}

# Axis order here is also the order of same-body MuJoCo hinge composition.
BODY_JOINTS = {
    "head": ("XYZ", ("head_x", "head_y", "head_z")),
    "abdomen": ("XYZ", ("abdomen_x", "abdomen_y", "abdomen_z")),
    "pelvis": ("XYZ", ("pelvis_x", "pelvis_y", "pelvis_z")),
    "left_thigh": (
        "XYZ",
        ("left_hip_x", "left_hip_y", "left_hip_z"),
    ),
    "left_shank": ("Z", ("left_knee_z",)),
    "left_foot": ("YZ", ("left_ankle_y", "left_ankle_z")),
    "right_thigh": (
        "XYZ",
        ("right_hip_x", "right_hip_y", "right_hip_z"),
    ),
    "right_shank": ("Z", ("right_knee_z",)),
    "right_foot": ("YZ", ("right_ankle_y", "right_ankle_z")),
    "left_upper_arm": (
        "XYZ",
        ("left_shoulder_x", "left_shoulder_y", "left_shoulder_z"),
    ),
    "left_forearm": ("Z", ("left_elbow_z",)),
    "left_hand": ("YZ", ("left_wrist_y", "left_wrist_z")),
    "right_upper_arm": (
        "XYZ",
        ("right_shoulder_x", "right_shoulder_y", "right_shoulder_z"),
    ),
    "right_forearm": ("Z", ("right_elbow_z",)),
    "right_hand": ("YZ", ("right_wrist_y", "right_wrist_z")),
}


@dataclass(frozen=True)
class RenderSettings:
    """Serializable batch rendering settings."""

    model_path: Path
    fps: float = 30.0
    width: int = 640
    height: int = 480
    crf: int = 20
    preset: str = "veryfast"
    max_duration: float | None = None
    clip_joints: bool = True
    overwrite: bool = False
    camera_distance: float = 3.2
    camera_elevation: float = -5.0
    foot_clearance: float = 0.005
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str | None = None


@dataclass(frozen=True)
class RetargetedClip:
    """MuJoCo generalized positions plus camera metadata for one BVH clip."""

    qpos: np.ndarray
    fps: float
    scale: float
    camera_azimuth: float


@dataclass(frozen=True)
class RenderResult:
    """One row in the resumable render manifest."""

    source_path: str
    output_path: str
    status: str
    frame_count: int
    duration_seconds: float
    scale: float | None
    elapsed_seconds: float
    error: str


class MujocoRetargeter:
    """Retarget parsed CMU motion into the supplied humanoid model."""

    def __init__(self, model_path: Path, clip_joints: bool, foot_clearance: float):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                'MuJoCo is not installed. Run: python -m pip install -e ".[render]"'
            ) from exc

        self.mujoco = mujoco
        try:
            self.model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load MuJoCo model: {model_path}") from exc
        self.data = mujoco.MjData(self.model)
        self.clip_joints = clip_joints
        self.foot_clearance = foot_clearance

        self._body_ids = {
            name: self._named_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in SOURCE_BODY_JOINTS
        }
        self._joint_ids = {
            name: self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            for _, names in BODY_JOINTS.values()
            for name in names
        }
        self._joint_qpos_addresses = {
            name: int(self.model.jnt_qposadr[joint_id])
            for name, joint_id in self._joint_ids.items()
        }
        self._foot_geom_ids = tuple(
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_foot_sole", "right_foot_sole")
        )

        root_quaternion = np.asarray(self.model.qpos0[3:7], dtype=np.float64)
        self._model_rest_rotation = _mujoco_quaternion_matrix(root_quaternion)
        mujoco.mj_forward(self.model, self.data)
        left_thigh = self.data.xpos[self._body_ids["left_thigh"]]
        left_foot = self.data.xpos[self._body_ids["left_foot"]]
        self._model_leg_length = float(np.linalg.norm(left_foot - left_thigh))

    def _named_id(self, object_type: Any, name: str) -> int:
        object_id = self.mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing required object: {name}")
        return int(object_id)

    def retarget(
        self,
        motion: BvhMotion,
        fps: float,
        max_duration: float | None,
    ) -> RetargetedClip:
        """Convert a BVH clip to model qpos samples."""
        source_indices = motion.joint_index
        missing = sorted(set(SOURCE_BODY_JOINTS.values()) - set(source_indices))
        if missing:
            raise ValueError(f"BVH is missing required joints: {missing}")

        frame_indices = motion.sample_indices(fps, max_duration)
        positions, rotations = motion.forward_kinematics(frame_indices)
        frame_count = len(frame_indices)
        desired_rotations: dict[str, np.ndarray] = {}

        for body_name, source_name in SOURCE_BODY_JOINTS.items():
            source_rotation = rotations[:, source_indices[source_name]]
            desired_rotations[body_name] = (
                CMU_TO_MUJOCO
                @ source_rotation
                @ CMU_TO_MUJOCO.T
                @ self._model_rest_rotation
            )

        qpos = np.broadcast_to(
            np.asarray(self.model.qpos0, dtype=np.float64),
            (frame_count, self.model.nq),
        ).copy()
        qpos[:, 3:7] = _matrices_to_mujoco_quaternions(desired_rotations["thorax"])

        for body_name, (axes, joint_names) in BODY_JOINTS.items():
            parent_name = BODY_PARENTS[body_name]
            parent_rotation = desired_rotations[parent_name]
            body_rotation = desired_rotations[body_name]
            relative_rotation = np.swapaxes(parent_rotation, -1, -2) @ body_rotation
            angles = _project_rotation(relative_rotation, axes)
            angles = np.unwrap(angles, axis=0)
            for column, joint_name in enumerate(joint_names):
                qpos[:, self._joint_qpos_addresses[joint_name]] = angles[:, column]

        if self.clip_joints:
            self._clip_hinge_ranges(qpos)

        scale = self._calculate_scale(motion)
        thorax_position = positions[:, source_indices["Chest2"]]
        mapped_position = (CMU_TO_MUJOCO @ thorax_position.T).T * scale
        qpos[:, :2] = mapped_position[:, :2] - mapped_position[0, :2]
        qpos[:, 2] = mapped_position[:, 2] - mapped_position[0, 2]
        qpos[:, 2] += self._root_height_for_floor(qpos[0])

        camera_azimuth = self._side_camera_azimuth(desired_rotations["pelvis"])
        return RetargetedClip(
            qpos=qpos,
            fps=fps,
            scale=scale,
            camera_azimuth=camera_azimuth,
        )

    def _calculate_scale(self, motion: BvhMotion) -> float:
        source_indices = motion.joint_index
        source_leg_length = sum(
            float(np.linalg.norm(motion.joints[source_indices[name]].offset))
            for name in ("LeftKnee", "LeftAnkle")
        )
        if source_leg_length <= 0:
            raise ValueError(f"BVH has an invalid leg length: {motion.path}")
        return self._model_leg_length / source_leg_length

    def _clip_hinge_ranges(self, qpos: np.ndarray) -> None:
        for joint_name, joint_id in self._joint_ids.items():
            if not self.model.jnt_limited[joint_id]:
                continue
            address = self._joint_qpos_addresses[joint_name]
            lower, upper = self.model.jnt_range[joint_id]
            qpos[:, address] = np.clip(qpos[:, address], lower, upper)

    def _root_height_for_floor(self, first_qpos: np.ndarray) -> float:
        self.data.qpos[:] = first_qpos
        self.data.qpos[2] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        bottoms: list[float] = []
        for geom_id in self._foot_geom_ids:
            center = self.data.geom_xpos[geom_id]
            axis = self.data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
            radius = float(self.model.geom_size[geom_id, 0])
            half_length = float(self.model.geom_size[geom_id, 1])
            endpoint_z = min(
                center[2] - axis[2] * half_length,
                center[2] + axis[2] * half_length,
            )
            bottoms.append(endpoint_z - radius)
        return self.foot_clearance - min(bottoms)

    @staticmethod
    def _side_camera_azimuth(pelvis_rotations: np.ndarray) -> float:
        sample_count = min(len(pelvis_rotations), 30)
        forward_vectors = pelvis_rotations[:sample_count, :, 0]
        mean_forward = np.mean(forward_vectors[:, :2], axis=0)
        if np.linalg.norm(mean_forward) < 1e-6:
            heading = 0.0
        else:
            heading = float(np.degrees(np.arctan2(mean_forward[1], mean_forward[0])))
        return heading + 90.0


class RenderRuntime:
    """Reusable MuJoCo/OpenGL state for a serial process or worker process."""

    def __init__(self, settings: RenderSettings):
        self.settings = settings
        self.retargeter = MujocoRetargeter(
            settings.model_path,
            settings.clip_joints,
            settings.foot_clearance,
        )
        mujoco = self.retargeter.mujoco
        try:
            self.renderer = mujoco.Renderer(
                self.retargeter.model,
                height=settings.height,
                width=settings.width,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not create a MuJoCo offscreen renderer. On a headless "
                "machine, configure a supported EGL/OSMesa backend."
            ) from exc
        self.camera = mujoco.MjvCamera()
        self.scene_option = mujoco.MjvOption()
        self.scene_option.sitegroup[:] = 0

    def close(self) -> None:
        """Release the renderer's OpenGL resources."""
        self.renderer.close()

    def render(self, source_path: Path, output_path: Path) -> RenderResult:
        """Render one BVH file, writing the final MP4 atomically."""
        started = time.perf_counter()
        if (
            output_path.exists()
            and not self.settings.overwrite
            and _video_is_valid(output_path, self.settings.ffprobe_path)
        ):
            return RenderResult(
                source_path=str(source_path.resolve()),
                output_path=str(output_path.resolve()),
                status="skipped",
                frame_count=0,
                duration_seconds=0.0,
                scale=None,
                elapsed_seconds=time.perf_counter() - started,
                error="",
            )

        motion = load_bvh(source_path)
        clip = self.retargeter.retarget(
            motion,
            self.settings.fps,
            self.settings.max_duration,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(f"{output_path.stem}.partial.mp4")
        if partial_path.exists():
            partial_path.unlink()

        process = _start_ffmpeg(partial_path, self.settings)
        try:
            self._stream_frames(process, clip)
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                stderr = _read_stderr(process)
                raise RuntimeError(
                    f"FFmpeg exited with code {return_code}: {stderr[-1000:]}"
                )
            if not _video_is_valid(partial_path, self.settings.ffprobe_path):
                raise RuntimeError("FFmpeg output failed validation")
            os.replace(partial_path, output_path)
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            if partial_path.exists():
                partial_path.unlink()
            raise

        elapsed = time.perf_counter() - started
        frame_count = len(clip.qpos)
        return RenderResult(
            source_path=str(source_path.resolve()),
            output_path=str(output_path.resolve()),
            status="rendered",
            frame_count=frame_count,
            duration_seconds=frame_count / clip.fps,
            scale=clip.scale,
            elapsed_seconds=elapsed,
            error="",
        )

    def _stream_frames(
        self,
        process: subprocess.Popen[bytes],
        clip: RetargetedClip,
    ) -> None:
        data = self.retargeter.data
        model = self.retargeter.model
        mujoco = self.retargeter.mujoco
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = clip.camera_azimuth
        self.camera.elevation = self.settings.camera_elevation
        self.camera.distance = self.settings.camera_distance

        if process.stdin is None:
            raise RuntimeError("FFmpeg stdin was not created")
        for qpos in clip.qpos:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            thorax_position = data.xpos[self.retargeter._body_ids["thorax"]]
            self.camera.lookat[:] = thorax_position
            self.camera.lookat[2] = max(0.8, thorax_position[2] - 0.6)
            self.renderer.update_scene(
                data,
                camera=self.camera,
                scene_option=self.scene_option,
            )
            frame = self.renderer.render()
            try:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"FFmpeg stopped accepting frames: {_read_stderr(process)[-1000:]}"
                ) from exc


def _project_rotation(matrices: np.ndarray, axes: str) -> np.ndarray:
    if axes == "XYZ":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Rotation.from_matrix(matrices).as_euler("XYZ")
    if axes == "YZ":
        y_angle = np.arcsin(np.clip(matrices[:, 0, 2], -1.0, 1.0))
        z_angle = np.arctan2(matrices[:, 1, 0], matrices[:, 1, 1])
        return np.column_stack((y_angle, z_angle))
    if axes == "Z":
        z_angle = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
        return z_angle[:, np.newaxis]
    raise ValueError(f"Unsupported target joint-axis sequence: {axes}")


def _mujoco_quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _matrices_to_mujoco_quaternions(matrices: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrices).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def _start_ffmpeg(
    output_path: Path,
    settings: RenderSettings,
) -> subprocess.Popen[bytes]:
    command = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{settings.width}x{settings.height}",
        "-framerate",
        f"{settings.fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start FFmpeg: {settings.ffmpeg_path}") from exc


def _read_stderr(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    try:
        return process.stderr.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _video_is_valid(path: Path, ffprobe_path: str | None) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    if ffprobe_path is None:
        return True
    try:
        completed = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _resolve_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg or run: "
            "python -m pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _result_for_error(
    source: Path,
    output: Path,
    started: float,
    exc: Exception,
) -> RenderResult:
    return RenderResult(
        source_path=str(source.resolve()),
        output_path=str(output.resolve()),
        status="failed",
        frame_count=0,
        duration_seconds=0.0,
        scale=None,
        elapsed_seconds=time.perf_counter() - started,
        error=f"{type(exc).__name__}: {exc}",
    )


_WORKER_RUNTIME: RenderRuntime | None = None


def _initialize_worker(settings: RenderSettings) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = RenderRuntime(settings)


def _render_worker(task: tuple[Path, Path]) -> RenderResult:
    source, output = task
    started = time.perf_counter()
    try:
        if _WORKER_RUNTIME is None:
            raise RuntimeError("Render worker was not initialized")
        return _WORKER_RUNTIME.render(source, output)
    except Exception as exc:
        return _result_for_error(source, output, started, exc)


def render_batch(
    tasks: list[tuple[Path, Path]],
    settings: RenderSettings,
    workers: int,
    manifest_path: Path,
) -> list[RenderResult]:
    """Render tasks serially or in reusable worker processes."""
    results: list[RenderResult] = []
    previous_results = _load_previous_results(manifest_path)
    if workers == 1:
        runtime = RenderRuntime(settings)
        try:
            for index, (source, output) in enumerate(tasks, start=1):
                started = time.perf_counter()
                try:
                    result = runtime.render(source, output)
                except Exception as exc:
                    result = _result_for_error(source, output, started, exc)
                result = _with_previous_metadata(result, previous_results)
                results.append(result)
                _write_manifest(manifest_path, results)
                _log_result(index, len(tasks), result)
        finally:
            runtime.close()
        return results

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(settings,),
    ) as executor:
        future_tasks = {executor.submit(_render_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_tasks), start=1):
            source, output = future_tasks[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _result_for_error(source, output, time.perf_counter(), exc)
            result = _with_previous_metadata(result, previous_results)
            results.append(result)
            _write_manifest(manifest_path, results)
            _log_result(completed, len(tasks), result)
    return results


def _load_previous_results(path: Path) -> dict[str, RenderResult]:
    if not path.is_file():
        return {}
    previous: dict[str, RenderResult] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                result = RenderResult(
                    source_path=row["source_path"],
                    output_path=row["output_path"],
                    status=row["status"],
                    frame_count=int(row["frame_count"]),
                    duration_seconds=float(row["duration_seconds"]),
                    scale=float(row["scale"]) if row["scale"] else None,
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    error=row["error"],
                )
                previous[result.source_path] = result
    except (KeyError, OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable previous manifest %s: %s", path, exc)
        return {}
    return previous


def _with_previous_metadata(
    result: RenderResult,
    previous_results: dict[str, RenderResult],
) -> RenderResult:
    if result.status != "skipped":
        return result
    previous = previous_results.get(result.source_path)
    if previous is None:
        return result
    return replace(
        result,
        frame_count=previous.frame_count,
        duration_seconds=previous.duration_seconds,
        scale=previous.scale,
    )


def _log_result(index: int, total: int, result: RenderResult) -> None:
    message = (
        f"[{index}/{total}] {result.status}: {Path(result.output_path).name} "
        f"({result.elapsed_seconds:.1f}s)"
    )
    if result.status == "failed":
        LOGGER.error("%s - %s", message, result.error)
    else:
        LOGGER.info("%s", message)


def _write_manifest(path: Path, results: list[RenderResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (
        list(asdict(results[0]).keys())
        if results
        else list(RenderResult.__dataclass_fields__)
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.source_path):
            writer.writerow(asdict(result))
    os.replace(temporary, path)


def _write_training_manifest(
    path: Path,
    input_dir: Path,
    results: list[RenderResult],
    settings: RenderSettings,
) -> None:
    metadata_path = input_dir.parent / "walking_manifest.csv"
    metadata_by_filename: dict[str, dict[str, str]] = {}
    if metadata_path.is_file():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                filename = Path(row.get("flat_relative_path", "")).name
                if filename:
                    metadata_by_filename[filename] = row

    fieldnames = [
        "clip_id",
        "title",
        "slug",
        "tier",
        "source_bvh",
        "video_path",
        "camera",
        "fps",
        "width",
        "height",
        "frame_count",
        "duration_seconds",
        "retarget_scale",
        "status",
    ]
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: item.source_path):
        if result.status == "failed":
            continue
        source_path = Path(result.source_path)
        metadata = metadata_by_filename.get(source_path.name, {})
        clip_id = metadata.get("clip_id") or source_path.stem.split("__", 1)[0]
        rows.append(
            {
                "clip_id": clip_id,
                "title": metadata.get("title", ""),
                "slug": metadata.get("slug", ""),
                "tier": metadata.get("tier", ""),
                "source_bvh": result.source_path,
                "video_path": result.output_path,
                "camera": "side",
                "fps": settings.fps,
                "width": settings.width,
                "height": settings.height,
                "frame_count": result.frame_count,
                "duration_seconds": result.duration_seconds,
                "retarget_scale": result.scale,
                "status": result.status,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _build_tasks(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"BVH input directory does not exist: {input_dir}")
    sources = sorted(input_dir.glob(args.glob))
    sources = [source for source in sources if source.is_file()]
    if args.start_index:
        sources = sources[args.start_index :]
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError(f"No files matched {args.glob!r} under {input_dir}")
    return [
        (source, args.output_dir / f"{source.stem}__side.mp4") for source in sources
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("CMU_reference_videos/walking_flat"),
        help="Directory containing flattened CMU BVH files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "CMU_reference_videos/models/"
            "human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
        ),
        help="MuJoCo MJCF/XML model used for rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CMU_reference_videos/mujoco_render"),
    )
    parser.add_argument("--glob", default="*.bvh", help="Input filename pattern.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    default_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel renderer processes (default: {default_workers}).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-elevation", type=float, default=-5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-clip-joints",
        action="store_true",
        help="Do not constrain retargeted angles to the XML joint limits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the batch without rendering.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Output manifest path; defaults to OUTPUT_DIR/render_manifest.csv.",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Training manifest path; defaults to OUTPUT_DIR/training_manifest.csv.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the BVH-to-MuJoCo batch renderer."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.width < 2 or args.height < 2:
        raise ValueError("Video width and height must be at least 2")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 yuv420p output requires even width and height")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
    args.output_dir = args.output_dir.resolve()
    tasks = _build_tasks(args)
    manifest_path = (args.manifest or args.output_dir / "render_manifest.csv").resolve()

    LOGGER.info("Discovered %d BVH files", len(tasks))
    LOGGER.info("Model: %s", model_path)
    LOGGER.info("Output: %s", args.output_dir)
    if args.dry_run:
        total_bytes = sum(source.stat().st_size for source, _ in tasks)
        print(f"Would render {len(tasks)} BVH files")
        print(f"Input size: {total_bytes / (1024**2):.1f} MiB")
        print(f"First: {tasks[0][0].name}")
        print(f"Last:  {tasks[-1][0].name}")
        return 0

    ffmpeg_path = _resolve_ffmpeg()
    settings = RenderSettings(
        model_path=model_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
        max_duration=args.max_duration,
        clip_joints=not args.no_clip_joints,
        overwrite=args.overwrite,
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=shutil.which("ffprobe"),
    )
    started = time.perf_counter()
    results = render_batch(tasks, settings, args.workers, manifest_path)
    training_manifest_path = (
        args.training_manifest or args.output_dir / "training_manifest.csv"
    ).resolve()
    _write_training_manifest(
        training_manifest_path,
        args.input_dir.resolve(),
        results,
        settings,
    )
    failures = [result for result in results if result.status == "failed"]
    rendered = sum(result.status == "rendered" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    elapsed = time.perf_counter() - started
    print(
        f"Finished {len(results)} files in {elapsed / 60:.1f} min: "
        f"rendered={rendered}, skipped={skipped}, failed={len(failures)}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Training manifest: {training_manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
