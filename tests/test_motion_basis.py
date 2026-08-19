import unittest

import numpy as np

from gait_aqa.vision.motion_basis import MotionBasis
from gait_aqa.vision.temporal_features import (
    align_feature_dict,
    coefficient_features,
    merge_feature_dicts,
)


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

    def test_dominant_frequency_is_reported_in_hz(self) -> None:
        sample_rate = 20.0
        time = np.arange(100) / sample_rate
        coefficients = np.sin(2.0 * np.pi * 2.0 * time)[:, None]
        features = coefficient_features(coefficients, sample_rate_hz=sample_rate)
        self.assertAlmostEqual(features["pca00_dominant_freq_hz"], 2.0)

    def test_incremental_basis_is_centered_and_orthonormal(self) -> None:
        rng = np.random.default_rng(4)
        flows = [
            rng.normal(size=(6, 4, 4, 2)).astype(np.float32),
            rng.normal(size=(7, 4, 4, 2)).astype(np.float32),
        ]
        basis = MotionBasis(n_components=3).fit(flows)
        expected_mean = np.concatenate(
            [flow.reshape(flow.shape[0], -1) for flow in flows]
        ).mean(axis=0)
        np.testing.assert_allclose(basis.mean_, expected_mean, atol=1e-6)
        np.testing.assert_allclose(
            basis.components_ @ basis.components_.T, np.eye(3), atol=1e-6
        )

    def test_feature_schema_mismatch_is_not_silently_zero_filled(self) -> None:
        with self.assertRaises(ValueError):
            merge_feature_dicts([{"a": 1.0}, {"b": 2.0}])
        with self.assertRaises(ValueError):
            align_feature_dict({"a": 1.0}, ["a", "missing"])


if __name__ == "__main__":
    unittest.main()
