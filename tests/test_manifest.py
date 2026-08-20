import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gait_aqa.data.build_manifest import prepare_real_video_manifest


class ManifestTests(unittest.TestCase):
    def test_real_manifest_keeps_only_supported_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "P01" / "forward" / "clip.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            source = pd.DataFrame(
                [
                    {
                        "video_path": "old/checkout/clip.mp4",
                        "policy_id": "P01",
                        "checkpoint_step": 10,
                        "scenario": "forward",
                        "seed": 7,
                        "camera": "side",
                        "composite_score": 0.75,
                        "tracking_rmse": 0.2,
                        "mean_torso_up": 0.95,
                        # Optional telemetry is unavailable in some MuJoCo
                        # environments and must not invalidate core labels.
                        "mean_foot_slip_speed": np.nan,
                        "mean_first_fall_survival_fraction": 1.0,
                        "status": "recorded",
                        "frames": 150,
                        "simulated_seconds": 5.0,
                        "resets": 0,
                        "ended_done": 0,
                    }
                ]
            )
            input_manifest = root / "manifest.csv"
            output_manifest = root / "prepared.csv"
            source.to_csv(input_manifest, index=False)

            prepared = prepare_real_video_manifest(
                input_manifest,
                output_manifest,
                dataset_root=root,
                camera="side",
            )

            self.assertEqual(len(prepared), 1)
            self.assertEqual(prepared.loc[0, "overall_score"], 75.0)
            self.assertEqual(prepared.loc[0, "stability_score"], 100.0)
            self.assertTrue(np.isnan(prepared.loc[0, "symmetry_score"]))
            self.assertEqual(Path(prepared.loc[0, "video_path"]), video)


if __name__ == "__main__":
    unittest.main()
