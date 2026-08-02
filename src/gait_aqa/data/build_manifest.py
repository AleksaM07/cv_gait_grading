"""Manifest construction helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest CSV."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)
