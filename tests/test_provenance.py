import csv
import unittest
from pathlib import Path


class ProvenanceTests(unittest.TestCase):
    def test_dataset_catalog_documents_local_payloads(self) -> None:
        catalog = Path("datasets/README.md").read_text(encoding="utf-8")
        for required_entry in (
            "MUJOCO_videos_better/",
            "CMU_reference_videos/walking_flat/",
            "datasets/disabled_gait/videos/",
            "datasets/gahu/videos/",
            "data/interim/flow/",
            "output/models/",
            "https://github.com/ShiqiYu/OpenGait",
        ):
            self.assertIn(required_entry, catalog)

    def test_provenance_schema(self) -> None:
        path = Path("reports/assets/docs/provenance.csv")
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.assertEqual(
                reader.fieldnames,
                [
                    "local_file",
                    "local_symbol",
                    "source_repository",
                    "source_commit",
                    "source_file",
                    "source_symbol",
                    "license",
                    "reuse_type",
                    "modifications",
                    "citation_key",
                ],
            )
            self.assertGreater(len(list(reader)), 0)


if __name__ == "__main__":
    unittest.main()
