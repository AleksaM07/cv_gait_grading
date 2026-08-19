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

    def fit(self, x: np.ndarray) -> StandardScaler:
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> NumpyRidgeRegressor:
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        design = np.c_[np.ones(x.shape[0]), x]
        if self.alpha > 0.0:
            penalty = np.sqrt(self.alpha) * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            design = np.vstack([design, penalty])
            target_shape = (penalty.shape[0],) + y.shape[1:]
            y = np.concatenate([y, np.zeros(target_shape, dtype=float)], axis=0)
        # Solving the augmented least-squares system is more stable than
        # forming normal equations, which square the condition number.
        self.coef_ = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Regressor has not been fit")
        design = np.c_[np.ones(x.shape[0]), x]
        return design @ self.coef_


@dataclass
class IndependentRidgeRegressor:
    """Fit one Ridge head per target while preserving genuinely missing labels."""

    alpha: float = 1.0
    models_: list[NumpyRidgeRegressor | None] | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> IndependentRidgeRegressor:
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 2 or targets.shape[0] != x.shape[0]:
            raise ValueError("y must have shape samples,targets and match x")
        self.models_ = []
        for index in range(targets.shape[1]):
            available = np.isfinite(targets[:, index])
            if not available.any():
                self.models_.append(None)
                continue
            model = NumpyRidgeRegressor(alpha=self.alpha).fit(
                x[available], targets[available, index]
            )
            self.models_.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.models_ is None:
            raise RuntimeError("Regressor has not been fit")
        output = np.full((x.shape[0], len(self.models_)), np.nan, dtype=float)
        for index, model in enumerate(self.models_):
            if model is not None:
                output[:, index] = model.predict(x)
        return output


@dataclass
class ClassicalGaitModel:
    """Self-contained score and irregularity predictor."""

    scaler: StandardScaler
    score_regressor: IndependentRidgeRegressor | NumpyRidgeRegressor
    irregularity_regressor: IndependentRidgeRegressor | NumpyRidgeRegressor
    feature_schema: list[str]
    metadata: dict[str, Any]

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Predict scores and irregularity probabilities."""
        scaled = self.scaler.transform(x)
        scores = np.clip(self.score_regressor.predict(scaled), 0.0, 100.0)
        logits = self.irregularity_regressor.predict(scaled)
        irregularities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
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
    score_regressor = IndependentRidgeRegressor(alpha=alpha).fit(x_scaled, y_scores)
    # Fit binary labels as least-squares logits around the class probability.
    clipped = np.clip(y_irregularities, 1e-3, 1.0 - 1e-3)
    logits = np.log(clipped / (1.0 - clipped))
    irregularity_regressor = IndependentRidgeRegressor(alpha=alpha).fit(
        x_scaled, logits
    )
    return ClassicalGaitModel(
        scaler=scaler,
        score_regressor=score_regressor,
        irregularity_regressor=irregularity_regressor,
        feature_schema=feature_schema,
        metadata=metadata or {},
    )
