"""Robust percentile scaling for metrics."""

from __future__ import annotations

import numpy as np


class RobustPercentileScaler:
    """Map training percentiles to `[0, 100]` with optional inversion."""

    def __init__(self, lower: float = 5.0, upper: float = 95.0, higher_is_better: bool = True) -> None:
        self.lower = lower
        self.upper = upper
        self.higher_is_better = higher_is_better
        self.low_: float | None = None
        self.high_: float | None = None

    def fit(self, values: np.ndarray) -> "RobustPercentileScaler":
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Cannot fit scaler on empty finite values")
        self.low_ = float(np.percentile(finite, self.lower))
        self.high_ = float(np.percentile(finite, self.upper))
        if np.isclose(self.low_, self.high_):
            self.high_ = self.low_ + 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.low_ is None or self.high_ is None:
            raise RuntimeError("Scaler has not been fit")
        scaled = (np.asarray(values, dtype=float) - self.low_) / (self.high_ - self.low_)
        if not self.higher_is_better:
            scaled = 1.0 - scaled
        return np.clip(100.0 * scaled, 0.0, 100.0)
