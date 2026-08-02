import unittest

from gait_aqa.labels.score_components import score_from_severities


class LabelTests(unittest.TestCase):
    def test_scores_are_in_range(self) -> None:
        scores = score_from_severities(
            {
                "sliding": 0.2,
                "asymmetry": 0.3,
                "jitter": 0.1,
                "hopping": 0.0,
                "micro_stepping": 0.2,
                "toe_dragging": 0.1,
                "fall": 0.0,
                "command_ignoring": 0.4,
            }
        )
        for key, value in scores.items():
            if key == "irregularities":
                continue
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_more_sliding_lowers_contact_score(self) -> None:
        low = score_from_severities({"sliding": 0.1})["foot_contact_quality"]
        high = score_from_severities({"sliding": 0.8})["foot_contact_quality"]
        self.assertLess(high, low)

    def test_more_asymmetry_lowers_symmetry_score(self) -> None:
        low = score_from_severities({"asymmetry": 0.1})["left_right_symmetry"]
        high = score_from_severities({"asymmetry": 0.8})["left_right_symmetry"]
        self.assertLess(high, low)


if __name__ == "__main__":
    unittest.main()
