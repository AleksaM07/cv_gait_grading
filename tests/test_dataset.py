import unittest

import numpy as np

from gait_aqa.data.video_dataset import deterministic_indices


class DatasetTests(unittest.TestCase):
    def test_deterministic_indices(self) -> None:
        first = deterministic_indices(10, 4)
        second = deterministic_indices(10, 4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
