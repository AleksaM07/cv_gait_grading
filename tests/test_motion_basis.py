import unittest

import numpy as np

from gait_aqa.vision.motion_basis import MotionBasis


class MotionBasisTests(unittest.TestCase):
    def test_basis_fit_transform_shape(self) -> None:
        rng = np.random.default_rng(0)
        train_flow = rng.normal(size=(4, 8, 8, 2)).astype(np.float32)
        test_flow = rng.normal(size=(3, 8, 8, 2)).astype(np.float32)
        basis = MotionBasis(n_components=3).fit([train_flow])
        coefficients = basis.transform(test_flow)
        self.assertEqual(coefficients.shape, (3, 3))

    def test_transform_before_fit_errors(self) -> None:
        with self.assertRaises(RuntimeError):
            MotionBasis(n_components=2).transform(np.zeros((2, 4, 4, 2)))


if __name__ == "__main__":
    unittest.main()
