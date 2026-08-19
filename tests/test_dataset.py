import unittest

import numpy as np

from gait_aqa.vision.preprocessing import resample_frames


class DatasetTests(unittest.TestCase):
    def test_frame_rate_resampling_preserves_duration(self) -> None:
        frames = np.zeros((31, 4, 4, 3), dtype=np.uint8)
        frames[:, 0, 0, 0] = np.arange(31)
        resampled = resample_frames(frames, source_fps=30.0, target_fps=20.0)
        self.assertEqual(resampled.shape[0], 21)
        self.assertEqual(int(resampled[-1, 0, 0, 0]), 30)


if __name__ == "__main__":
    unittest.main()
