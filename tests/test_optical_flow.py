import unittest

import numpy as np

from gait_aqa.vision.optical_flow import compute_dense_flow


class OpticalFlowTests(unittest.TestCase):
    def test_flow_shape(self) -> None:
        frames = np.zeros((5, 16, 16), dtype=np.float32)
        frames[:, 5:9, 5:9] = 1.0
        frames[1:, 5:9, 6:10] = 1.0
        flow = compute_dense_flow(frames, mode="absolute_flow")
        self.assertEqual(flow.shape, (4, 16, 16, 2))

    def test_short_video_errors(self) -> None:
        with self.assertRaises(ValueError):
            compute_dense_flow(np.zeros((1, 16, 16), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
