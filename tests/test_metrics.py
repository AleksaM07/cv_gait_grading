import unittest

import pandas as pd

from gait_aqa.evaluation.metrics import classification_table, regression_table


class MetricsTests(unittest.TestCase):
    def test_regression_table_reports_train_mean_baseline(self) -> None:
        predictions = pd.DataFrame(
            {
                "split": ["train", "train", "test", "test"],
                "true_overall_score": [10.0, 30.0, 15.0, 25.0],
                "pred_overall_score": [10.0, 30.0, 16.0, 24.0],
            }
        )
        table = regression_table(predictions)
        self.assertAlmostEqual(table.loc[0, "baseline_mae"], 5.0)
        self.assertGreater(table.loc[0, "mae_skill_vs_mean"], 0.0)

    def test_classification_table_flags_one_class_split(self) -> None:
        predictions = pd.DataFrame(
            {
                "split": ["test", "test"],
                "true_fall_label": [1.0, 1.0],
                "pred_fall_label": [0.9, 0.8],
            }
        )
        table = classification_table(predictions)
        self.assertEqual(table.loc[0, "classes_present"], 1.0)
        self.assertEqual(table.loc[0, "positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
