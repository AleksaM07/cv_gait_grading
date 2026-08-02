"""Classical NumPy regressors for gait quality scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCORE_TARGETS = [
    "overall_score",
    "stability_score",
    "contact_score",
    "symmetry_score",
    "periodicity_score",
    "smoothness_score",
    "tracking_score",
]

IRREGULARITY_TARGETS = [
    "foot_sliding_label",
    "hopping_label",
    "micro_stepping_label",
    "asymmetry_label",
    "torso_instability_label",
    "toe_dragging_label",
    "fall_label",
    "command_ignoring_label",
]


@dataclass
class StandardScaler:
    """Small feature standardizer."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class NumpyRidgeRegressor:
    """Closed-form multi-output Ridge regression."""

    alpha: float = 1.0
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidgeRegressor":
        design = np.c_[np.ones(x.shape[0]), x]
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Regressor has not been fit")
        design = np.c_[np.ones(x.shape[0]), x]
        return design @ self.coef_


@dataclass
class ClassicalGaitModel:
    """Self-contained score and irregularity predictor."""

    scaler: StandardScaler
    score_regressor: NumpyRidgeRegressor
    irregularity_regressor: NumpyRidgeRegressor
    feature_schema: list[str]
    metadata: dict[str, Any]

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Predict scores and irregularity probabilities."""
        scaled = self.scaler.transform(x)
        scores = np.clip(self.score_regressor.predict(scaled), 0.0, 100.0)
        logits = self.irregularity_regressor.predict(scaled)
        irregularities = 1.0 / (1.0 + np.exp(-logits))
        return {"scores": scores, "irregularities": irregularities}


def fit_classical_model(
    x: np.ndarray,
    y_scores: np.ndarray,
    y_irregularities: np.ndarray,
    feature_schema: list[str],
    alpha: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> ClassicalGaitModel:
    """Fit a classical gait model."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    score_regressor = NumpyRidgeRegressor(alpha=alpha).fit(x_scaled, y_scores)
    # Fit binary labels as least-squares logits around the class probability.
    clipped = np.clip(y_irregularities, 1e-3, 1.0 - 1e-3)
    logits = np.log(clipped / (1.0 - clipped))
    irregularity_regressor = NumpyRidgeRegressor(alpha=alpha).fit(x_scaled, logits)
    return ClassicalGaitModel(
        scaler=scaler,
        score_regressor=score_regressor,
        irregularity_regressor=irregularity_regressor,
        feature_schema=feature_schema,
        metadata=metadata or {},
    )
