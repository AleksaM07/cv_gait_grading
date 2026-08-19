"""Small, dependency-light parser and forward kinematics for BVH motion files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BvhJoint:
    """One joint in a BVH hierarchy."""

    name: str
    parent: int | None
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int


@dataclass(frozen=True)
class BvhMotion:
    """Parsed BVH hierarchy and motion samples."""

    path: Path
    joints: tuple[BvhJoint, ...]
    values: np.ndarray
    frame_time: float

    @property
    def frame_count(self) -> int:
        """Return the number of source motion frames."""
        return int(self.values.shape[0])

    @property
    def duration(self) -> float:
        """Return the duration from the first to the last sample in seconds."""
        return max(0.0, (self.frame_count - 1) * self.frame_time)

    @property
    def joint_index(self) -> dict[str, int]:
        """Return a case-sensitive joint-name lookup."""
        return {joint.name: index for index, joint in enumerate(self.joints)}

    def sample_indices(
        self,
        fps: float,
        max_duration: float | None = None,
    ) -> np.ndarray:
        """Choose source frames nearest to a constant-rate output time grid."""
        if fps <= 0:
            raise ValueError(f"Output FPS must be positive, got {fps}")

        duration = self.duration
        if max_duration is not None:
            if max_duration <= 0:
                raise ValueError("Maximum duration must be positive")
            duration = min(duration, max_duration)

        # BVH frame times are commonly rounded decimal representations such as
        # 0.0083333 for 1/120 s. The small tolerance prevents an exact source
        # interval from disappearing due only to that text rounding.
        output_count = max(1, int(np.floor(duration * fps + 1e-3)) + 1)
        output_times = np.arange(output_count, dtype=np.float64) / fps
        indices = np.rint(output_times / self.frame_time).astype(np.int64)
        return np.clip(indices, 0, self.frame_count - 1)

    def forward_kinematics(
        self,
        frame_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate BVH joint positions and orientations in world coordinates.

        Args:
            frame_indices: One-dimensional array of source frame indices.

        Returns:
            Tuple ``(positions, rotations)`` with shapes ``(F, J, 3)`` and
            ``(F, J, 3, 3)``.
        """
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices.ndim != 1:
            raise ValueError("frame_indices must be one-dimensional")
        if frame_indices.size == 0:
            raise ValueError("At least one frame must be selected")
        if frame_indices.min() < 0 or frame_indices.max() >= self.frame_count:
            raise IndexError("A requested BVH frame is out of range")

        selected = self.values[frame_indices]
        frame_count = len(frame_indices)
        joint_count = len(self.joints)
        positions = np.empty((frame_count, joint_count, 3), dtype=np.float64)
        rotations = np.empty(
            (frame_count, joint_count, 3, 3),
            dtype=np.float64,
        )

        for joint_index, joint in enumerate(self.joints):
            local_position = np.broadcast_to(joint.offset, (frame_count, 3)).copy()
            rotation_channels: list[str] = []
            rotation_values: list[np.ndarray] = []

            for channel_offset, channel in enumerate(joint.channels):
                values = selected[:, joint.channel_start + channel_offset]
                axis = channel[0].upper()
                if channel.lower().endswith("position"):
                    local_position[:, "XYZ".index(axis)] += values
                elif channel.lower().endswith("rotation"):
                    rotation_channels.append(axis)
                    rotation_values.append(values)
                else:
                    raise ValueError(
                        f"Unsupported BVH channel {channel!r} in {self.path}"
                    )

            if rotation_channels:
                angles = np.column_stack(rotation_values)
                local_rotation = Rotation.from_euler(
                    "".join(rotation_channels),
                    angles,
                    degrees=True,
                ).as_matrix()
            else:
                local_rotation = np.broadcast_to(
                    np.eye(3),
                    (frame_count, 3, 3),
                )

            if joint.parent is None:
                positions[:, joint_index] = local_position
                rotations[:, joint_index] = local_rotation
                continue

            parent_rotation = rotations[:, joint.parent]
            positions[:, joint_index] = positions[:, joint.parent] + np.einsum(
                "fij,fj->fi", parent_rotation, local_position
            )
            rotations[:, joint_index] = np.einsum(
                "fij,fjk->fik",
                parent_rotation,
                local_rotation,
            )

        return positions, rotations


def load_bvh(path: Path) -> BvhMotion:
    """Load a BVH file with strict validation and useful error messages."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OSError(f"Could not read BVH file: {path}") from exc

    try:
        hierarchy_text, motion_text = text.split("MOTION", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"BVH file has no MOTION section: {path}") from exc

    joints = _parse_hierarchy(hierarchy_text, path)
    values, frame_time = _parse_motion(motion_text, joints, path)
    return BvhMotion(
        path=path,
        joints=tuple(joints),
        values=values,
        frame_time=frame_time,
    )


def _parse_hierarchy(hierarchy_text: str, path: Path) -> list[BvhJoint]:
    mutable_joints: list[dict[str, object]] = []
    stack: list[int | None] = []
    pending: int | None = None
    channel_cursor = 0

    for raw_line in hierarchy_text.splitlines():
        line = raw_line.strip()
        if not line or line == "HIERARCHY":
            continue
        if line.startswith(("ROOT ", "JOINT ")):
            name = line.split(maxsplit=1)[1]
            parent = next((item for item in reversed(stack) if item is not None), None)
            mutable_joints.append(
                {
                    "name": name,
                    "parent": parent,
                    "offset": np.zeros(3, dtype=np.float64),
                    "channels": (),
                    "channel_start": channel_cursor,
                }
            )
            pending = len(mutable_joints) - 1
            continue
        if line.startswith("End Site"):
            pending = None
            continue
        if line == "{":
            stack.append(pending)
            pending = None
            continue
        if line == "}":
            if not stack:
                raise ValueError(f"Unbalanced hierarchy braces in {path}")
            stack.pop()
            continue
        if line.startswith("OFFSET "):
            if not stack or stack[-1] is None:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid OFFSET line in {path}: {line}")
            mutable_joints[stack[-1]]["offset"] = np.asarray(
                [float(value) for value in parts[1:]],
                dtype=np.float64,
            )
            continue
        if line.startswith("CHANNELS "):
            if not stack or stack[-1] is None:
                raise ValueError(f"CHANNELS outside a joint in {path}")
            parts = line.split()
            count = int(parts[1])
            channels = tuple(parts[2:])
            if len(channels) != count:
                raise ValueError(f"Invalid CHANNELS count in {path}: {line}")
            mutable_joints[stack[-1]]["channels"] = channels
            mutable_joints[stack[-1]]["channel_start"] = channel_cursor
            channel_cursor += count

    if stack:
        raise ValueError(f"Unbalanced hierarchy braces in {path}")
    if not mutable_joints:
        raise ValueError(f"BVH file contains no joints: {path}")

    return [
        BvhJoint(
            name=str(joint["name"]),
            parent=joint["parent"] if isinstance(joint["parent"], int) else None,
            offset=np.asarray(joint["offset"], dtype=np.float64),
            channels=cast(tuple[str, ...], joint["channels"]),
            channel_start=cast(int, joint["channel_start"]),
        )
        for joint in mutable_joints
    ]


def _parse_motion(
    motion_text: str,
    joints: list[BvhJoint],
    path: Path,
) -> tuple[np.ndarray, float]:
    lines = [line.strip() for line in motion_text.splitlines() if line.strip()]
    if len(lines) < 3 or not lines[0].startswith("Frames:"):
        raise ValueError(f"Invalid BVH motion header in {path}")
    if not lines[1].startswith("Frame Time:"):
        raise ValueError(f"BVH motion header has no frame time in {path}")

    frame_count = int(lines[0].split(":", maxsplit=1)[1])
    frame_time = float(lines[1].split(":", maxsplit=1)[1])
    if frame_count <= 0 or frame_time <= 0:
        raise ValueError(f"Invalid frame count or frame time in {path}")

    channel_count = sum(len(joint.channels) for joint in joints)
    values = np.fromstring(" ".join(lines[2:]), sep=" ", dtype=np.float64)
    expected = frame_count * channel_count
    if values.size != expected:
        raise ValueError(
            f"BVH motion size mismatch in {path}: expected {expected} values, "
            f"found {values.size}"
        )
    return values.reshape(frame_count, channel_count), frame_time
