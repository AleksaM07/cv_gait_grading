"""Irregularity label helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IrregularityInterval:
    """A temporal interval where an irregularity was detected."""

    name: str
    start_frame: int
    end_frame: int
    severity: float


def temporal_intervals(
    telemetry: pd.DataFrame,
    label_columns: list[str],
    threshold: float = 0.5,
) -> list[IrregularityInterval]:
    """Extract contiguous high-severity intervals from telemetry."""
    intervals: list[IrregularityInterval] = []
    for column in label_columns:
        if column not in telemetry:
            continue
        active = telemetry[column].fillna(0.0).to_numpy() >= threshold
        start: int | None = None
        for index, is_active in enumerate(active):
            if is_active and start is None:
                start = index
            if (not is_active or index == len(active) - 1) and start is not None:
                end = index if not is_active else index + 1
                severity = float(telemetry[column].iloc[start:end].max())
                intervals.append(IrregularityInterval(column, start, end, severity))
                start = None
    return intervals
