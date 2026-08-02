"""SVD/PCA motion basis for stacked dense flow fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionBasis:
    """Low-rank basis learned from normalized dense flow fields."""

    n_components: int = 8
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, flows: list[np.ndarray]) -> "MotionBasis":
        """Fit the PCA/SVD basis on training flow fields only."""
        matrix = _stack_flows(flows)
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        count = min(self.n_components, vt.shape[0])
        self.components_ = vt[:count]
        variance = singular_values**2
        total = float(variance.sum())
        self.explained_variance_ratio_ = variance[:count] / total if total > 0 else np.zeros(count)
        return self

    def transform(self, flow: np.ndarray) -> np.ndarray:
        """Project one clip's flow fields onto the learned basis."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("MotionBasis has not been fit")
        matrix = _stack_flows([flow])
        return (matrix - self.mean_) @ self.components_.T

    def fit_transform(self, flows: list[np.ndarray]) -> list[np.ndarray]:
        """Fit on flows and return per-clip coefficient sequences."""
        self.fit(flows)
        return [self.transform(flow) for flow in flows]


def _stack_flows(flows: list[np.ndarray]) -> np.ndarray:
    if not flows:
        raise ValueError("At least one flow clip is required")
    rows = []
    for flow in flows:
        if flow.ndim != 4 or flow.shape[-1] != 2:
            raise ValueError("Each flow clip must have shape T,H,W,2")
        rows.append(flow.reshape(flow.shape[0], -1))
    return np.concatenate(rows, axis=0).astype(np.float64)
