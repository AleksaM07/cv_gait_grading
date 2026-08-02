import tempfile
import unittest
from pathlib import Path

import numpy as np

from gait_aqa.models.classical_regressor import fit_classical_model
from gait_aqa.models.model_io import load_model, save_model


class ModelIoTests(unittest.TestCase):
    def test_save_load_consistency(self) -> None:
        x = np.arange(20, dtype=float).reshape(5, 4)
        y_scores = np.tile(np.linspace(10, 90, 5)[:, None], (1, 7))
        y_labels = np.zeros((5, 8), dtype=float)
        model = fit_classical_model(x, y_scores, y_labels, [f"f{i}" for i in range(4)])
        expected = model.predict(x)["scores"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            save_model(model, path)
            loaded = load_model(path)
        np.testing.assert_allclose(expected, loaded.predict(x)["scores"])


if __name__ == "__main__":
    unittest.main()
