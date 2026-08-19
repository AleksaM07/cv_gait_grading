"""SVD/PCA motion basis for stacked dense flow fields."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class MotionBasis:
    """Low-rank basis learned from normalized dense flow fields.

    Incremental PCA implements the centered SVD construction described for
    learned motion models by Szeliski (Sec. 8.2.2) without materializing every
    training flow field in one multi-gigabyte matrix.
    """

    n_components: int = 8
    explained_variance: float | None = None
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, flows: Iterable[np.ndarray]) -> MotionBasis:
        """Fit the PCA/SVD basis on training flow fields only."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.explained_variance is not None and not (
            0.0 < self.explained_variance <= 1.0
        ):
            raise ValueError("explained_variance must be in (0, 1]")

        fitted_batches = 0
        expected_width: int | None = None
        sample_count = 0
        running_mean: np.ndarray | None = None
        singular_values: np.ndarray | None = None
        components: np.ndarray | None = None
        total_sum_squares = 0.0
        for flow in flows:
            matrix = _flow_matrix(flow)
            expected_width = expected_width or matrix.shape[1]
            if matrix.shape[1] != expected_width:
                raise ValueError("All flow clips must have the same spatial shape")
            if matrix.shape[0] < self.n_components:
                raise ValueError(
                    "Each flow clip needs at least n_components frames for "
                    "incremental PCA"
                )
            batch_count = matrix.shape[0]
            batch_mean = matrix.mean(axis=0)
            centered = matrix - batch_mean
            if running_mean is None:
                stacked = centered
                running_mean = batch_mean
                total_sum_squares = float(np.sum(centered**2))
            else:
                if singular_values is None or components is None:
                    raise RuntimeError("Incremental SVD state is incomplete")
                combined_count = sample_count + batch_count
                mean_delta = running_mean - batch_mean
                correction_scale = np.sqrt(sample_count * batch_count / combined_count)
                mean_correction = correction_scale * mean_delta
                stacked = np.vstack(
                    [
                        singular_values[:, None] * components,
                        centered,
                        mean_correction[None, :],
                    ]
                )
                total_sum_squares += float(
                    np.sum(centered**2)
                    + (sample_count * batch_count / combined_count)
                    * np.sum(mean_delta**2)
                )
                running_mean = (
                    sample_count * running_mean + batch_count * batch_mean
                ) / combined_count

            _, updated_singular_values, updated_components = np.linalg.svd(
                stacked, full_matrices=False
            )
            keep = min(self.n_components, updated_components.shape[0])
            singular_values = updated_singular_values[:keep]
            components = updated_components[:keep]
            sample_count += batch_count
            fitted_batches += 1
        if fitted_batches == 0:
            raise ValueError("At least one flow clip is required")

        if running_mean is None or singular_values is None or components is None:
            raise RuntimeError("Incremental SVD did not produce a basis")
        count = components.shape[0]
        ratios = (
            singular_values**2 / total_sum_squares
            if total_sum_squares > 0.0
            else np.zeros_like(singular_values)
        )
        if self.explained_variance is not None:
            reached = np.flatnonzero(np.cumsum(ratios) >= self.explained_variance)
            if reached.size:
                count = int(reached[0]) + 1
        self.mean_ = np.asarray(running_mean, dtype=np.float64)
        self.components_ = np.asarray(components[:count], dtype=np.float64)
        self.explained_variance_ratio_ = ratios[:count]
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
        rows.append(_flow_matrix(flow))
    return np.concatenate(rows, axis=0).astype(np.float64)


def _flow_matrix(flow: np.ndarray) -> np.ndarray:
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError("Each flow clip must have shape T,H,W,2")
    return flow.reshape(flow.shape[0], -1).astype(np.float64, copy=False)
