"""Report table helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Save a CSV report table."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return output
