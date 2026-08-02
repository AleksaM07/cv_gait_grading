"""Validation helpers for manifests and labels."""

from __future__ import annotations

import pandas as pd


SCORE_COLUMNS = [
    "overall_score",
    "stability_score",
    "contact_score",
    "symmetry_score",
    "periodicity_score",
    "smoothness_score",
    "tracking_score",
]


def validate_score_ranges(manifest: pd.DataFrame) -> None:
    """Ensure all score columns are present and within `[0, 100]`."""
    missing = [column for column in SCORE_COLUMNS if column not in manifest]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")
    for column in SCORE_COLUMNS:
        invalid = ~manifest[column].between(0.0, 100.0)
        if invalid.any():
            raise AssertionError(f"Score column out of range: {column}")
