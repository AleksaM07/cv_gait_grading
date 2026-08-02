import unittest

import pandas as pd

from gait_aqa.data.split_dataset import assert_no_group_overlap, grouped_split


class SplitTests(unittest.TestCase):
    def test_grouped_split_has_no_overlap(self) -> None:
        manifest = pd.DataFrame(
            {
                "clip_id": [f"c{i}" for i in range(12)],
                "split_group": [f"g{i // 2}" for i in range(12)],
            }
        )
        split = grouped_split(manifest, seed=3)
        assert_no_group_overlap(split)
        self.assertIn("test", set(split["split"]))


if __name__ == "__main__":
    unittest.main()
