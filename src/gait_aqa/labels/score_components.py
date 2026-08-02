"""Simulation-derived gait score construction."""

from __future__ import annotations

from typing import Mapping, Any

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "stability": 0.25,
    "foot_contact_quality": 0.20,
    "left_right_symmetry": 0.15,
    "periodicity": 0.15,
    "smoothness": 0.10,
    "command_tracking": 0.15,
}


def clamp_score(value: float) -> float:
    """Clamp a score to `[0, 100]`."""
    return float(np.clip(value, 0.0, 100.0))


def weighted_overall(
    component_scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute the configured weighted overall score."""
    active_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        raise ValueError("At least one positive score weight is required")
    value = sum(component_scores[name] * weight for name, weight in active_weights.items())
    return clamp_score(value / total_weight)


def score_from_severities(values: Mapping[str, Any]) -> dict[str, Any]:
    """Create component scores from normalized synthetic irregularity severities."""
    sliding = _severity(values, "sliding", "foot_sliding")
    asymmetry = _severity(values, "asymmetry", "left_right_asymmetry")
    jitter = _severity(values, "jitter", "torso_instability")
    hopping = _severity(values, "hopping")
    micro = _severity(values, "micro_stepping")
    toe = _severity(values, "toe_dragging")
    fall = _severity(values, "fall", "fall_or_near_fall")
    ignoring = _severity(values, "command_ignoring")

    components = {
        "stability": clamp_score(100.0 * (1.0 - 0.75 * fall - 0.25 * jitter)),
        "foot_contact_quality": clamp_score(100.0 * (1.0 - 0.80 * sliding - 0.20 * toe)),
        "left_right_symmetry": clamp_score(100.0 * (1.0 - asymmetry)),
        "periodicity": clamp_score(100.0 * (1.0 - 0.65 * hopping - 0.35 * micro)),
        "smoothness": clamp_score(100.0 * (1.0 - 0.85 * jitter - 0.15 * hopping)),
        "command_tracking": clamp_score(100.0 * (1.0 - ignoring)),
    }
    return {
        **components,
        "overall_score": weighted_overall(components),
        "irregularities": {
            "foot_sliding": sliding,
            "hopping": hopping,
            "micro_stepping": micro,
            "left_right_asymmetry": asymmetry,
            "torso_instability": jitter,
            "toe_dragging": toe,
            "fall_or_near_fall": fall,
            "command_ignoring": ignoring,
        },
    }


def score_from_walker_trial(trial: pd.Series | Mapping[str, Any]) -> dict[str, Any]:
    """Build scores from fields exported by the MuJoCo walker analysis."""
    get = trial.get if isinstance(trial, Mapping) else trial.get
    survival = _finite(get("survival_fraction", 1.0), 1.0)
    falls = _finite(get("falls", 0.0), 0.0)
    torso_up = _finite(get("mean_torso_up", 0.8), 0.8)
    tracking_rmse = _finite(get("tracking_rmse", 0.5), 0.5)
    command_failure = _finite(get("command_failure_rate", 0.0), 0.0)
    slip = _finite(get("mean_foot_slip_speed", 0.0), 0.0)
    action_rate = _finite(get("mean_action_rate_norm", 0.0), 0.0)

    fall_severity = float(np.clip((1.0 - survival) + 0.20 * falls, 0.0, 1.0))
    slip_severity = float(np.clip(slip / 1.0, 0.0, 1.0))
    torso_severity = float(np.clip((0.95 - torso_up) / 0.50, 0.0, 1.0))
    tracking_severity = float(np.clip(0.6 * tracking_rmse + command_failure, 0.0, 1.0))
    action_severity = float(np.clip(action_rate / 60.0, 0.0, 1.0))
    values = {
        "sliding": slip_severity,
        "asymmetry": 0.0,
        "jitter": max(torso_severity, action_severity * 0.5),
        "hopping": 0.0,
        "micro_stepping": 0.0,
        "toe_dragging": 0.0,
        "fall": fall_severity,
        "command_ignoring": tracking_severity,
    }
    return score_from_severities(values)


def _severity(values: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in values:
            return float(np.clip(float(values[name]), 0.0, 1.0))
    return 0.0


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number
