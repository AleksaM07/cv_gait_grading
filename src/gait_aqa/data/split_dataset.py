"""Leakage-safe grouped dataset splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_split(
    manifest: pd.DataFrame,
    group_column: str = "split_group",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 13,
) -> pd.DataFrame:
    """Assign train/val/test splits without breaking groups."""
    if group_column not in manifest:
        raise ValueError(f"Missing group column: {group_column}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    groups = np.asarray(sorted(manifest[group_column].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_end = max(1, round(len(groups) * train_fraction))
    val_end = min(len(groups), train_end + round(len(groups) * val_fraction))
    split_by_group = {
        group: ("train" if index < train_end else "val" if index < val_end else "test")
        for index, group in enumerate(groups)
    }
    result = manifest.copy()
    result["split"] = result[group_column].astype(str).map(split_by_group)
    if result["split"].nunique() < 2 and len(groups) > 1:
        last_group = groups[-1]
        result.loc[result[group_column].astype(str) == last_group, "split"] = "test"
    return result


def assert_no_group_overlap(
    manifest: pd.DataFrame, group_column: str = "split_group"
) -> None:
    """Raise if any group appears in more than one split."""
    if "split" not in manifest:
        raise ValueError("Manifest has no split column")
    counts = manifest.groupby(group_column)["split"].nunique()
    overlapping = counts[counts > 1]
    if not overlapping.empty:
        raise AssertionError(
            f"Groups overlap across splits: {overlapping.index.tolist()}"
        )
