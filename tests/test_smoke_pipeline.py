import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SmokePipelineTests(unittest.TestCase):
    def test_cli_generate_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gait_aqa.cli",
                    "generate-synthetic",
                    "--output-dir",
                    str(root / "raw"),
                    "--manifest",
                    str(root / "manifest.csv"),
                    "--clip-count",
                    "4",
                ],
                cwd=Path.cwd(),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Wrote 4 synthetic clips", result.stdout)
            self.assertTrue((root / "manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
