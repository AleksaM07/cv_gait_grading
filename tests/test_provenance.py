import csv
import unittest
from pathlib import Path


class ProvenanceTests(unittest.TestCase):
    def test_provenance_schema(self) -> None:
        path = Path("docs/provenance.csv")
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
