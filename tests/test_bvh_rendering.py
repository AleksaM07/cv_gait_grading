"""Tests for BVH parsing and MuJoCo rotation projection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gait_aqa.reference_videos.bvh import load_bvh
from gait_aqa.reference_videos.render_mujoco import _project_rotation

MINIMAL_BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 0 0 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Knee
    {
        OFFSET 1 0 0
        CHANNELS 3 Zrotation Yrotation Xrotation
        End Site
        {
            OFFSET 0 -1 0
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.0083333
0 0 0 0 0 0 0 0 0
2 3 4 90 0 0 0 0 0
"""


class BvhParserTests(unittest.TestCase):
    def test_load_and_forward_kinematics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.bvh"
            path.write_text(MINIMAL_BVH, encoding="utf-8")
            motion = load_bvh(path)

        self.assertEqual(motion.frame_count, 2)
        self.assertEqual(motion.values.shape, (2, 9))
        self.assertEqual(motion.joint_index, {"Hips": 0, "Knee": 1})

        positions, rotations = motion.forward_kinematics(np.asarray([1]))
        np.testing.assert_allclose(positions[0, 0], [2, 3, 4], atol=1e-8)
        np.testing.assert_allclose(positions[0, 1], [2, 4, 4], atol=1e-8)
        np.testing.assert_allclose(
            rotations[0, 0],
            Rotation.from_euler("Z", 90, degrees=True).as_matrix(),
            atol=1e-8,
        )

    def test_output_sampling_uses_nearest_source_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.bvh"
            path.write_text(MINIMAL_BVH, encoding="utf-8")
            motion = load_bvh(path)

        np.testing.assert_array_equal(motion.sample_indices(120), [0, 1])
        np.testing.assert_array_equal(motion.sample_indices(30), [0])


class RotationProjectionTests(unittest.TestCase):
    def test_xyz_projection_round_trip(self) -> None:
        angles = np.asarray([[0.2, -0.3, 0.4], [-0.5, 0.1, -0.2]])
        matrices = Rotation.from_euler("XYZ", angles).as_matrix()
        np.testing.assert_allclose(_project_rotation(matrices, "XYZ"), angles)

    def test_yz_projection_round_trip(self) -> None:
        angles = np.asarray([[0.2, -0.3], [-0.5, 0.1]])
        matrices = Rotation.from_euler("YZ", angles).as_matrix()
        np.testing.assert_allclose(_project_rotation(matrices, "YZ"), angles)

    def test_z_projection_round_trip(self) -> None:
        angles = np.asarray([[0.2], [-0.5]])
        matrices = Rotation.from_euler("Z", angles).as_matrix()
        np.testing.assert_allclose(_project_rotation(matrices, "Z"), angles)


if __name__ == "__main__":
    unittest.main()
