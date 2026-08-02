"""Temporal features from PCA coefficients and regional flow."""

from __future__ import annotations

import numpy as np

from gait_aqa.vision.body_regions import regional_flow_magnitudes


def coefficient_features(coefficients: np.ndarray, prefix: str = "pca") -> dict[str, float]:
    """Summarize temporal PCA coefficient sequences."""
    if coefficients.ndim != 2:
        raise ValueError("Coefficients must have shape T,C")
    features: dict[str, float] = {}
    for component in range(coefficients.shape[1]):
        values = coefficients[:, component]
        name = f"{prefix}{component:02d}"
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_std"] = float(values.std())
        features[f"{name}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{name}_range"] = float(np.percentile(values, 95) - np.percentile(values, 5))
        delta = np.diff(values)
        features[f"{name}_delta_rms"] = float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
        features[f"{name}_dominant_freq"] = _dominant_frequency(values)
        features[f"{name}_spectral_entropy"] = _spectral_entropy(values)
        features[f"{name}_autocorr_peak"] = _autocorr_peak(values)
        features[f"{name}_zero_crossings"] = float(np.count_nonzero(np.diff(np.signbit(values - values.mean()))))
    return features


def flow_features(flow: np.ndarray) -> dict[str, float]:
    """Summarize dense and regional flow magnitudes."""
    magnitude = np.linalg.norm(flow, axis=-1)
    features = {
        "flow_mean": float(magnitude.mean()),
        "flow_std": float(magnitude.std()),
        "flow_p95": float(np.percentile(magnitude, 95)),
        "flow_high_freq_energy": _high_frequency_energy(magnitude.mean(axis=(1, 2))),
    }
    regions = regional_flow_magnitudes(flow)
    for name, values in regions.items():
        features[f"region_{name}_mean"] = float(values.mean())
        features[f"region_{name}_std"] = float(values.std())
    if "left_foot" in regions and "right_foot" in regions:
        left = regions["left_foot"]
        right = regions["right_foot"]
        denom = max(float(left.std() * right.std()), 1e-8)
        features["left_right_flow_corr"] = float(np.mean((left - left.mean()) * (right - right.mean())) / denom)
        features["left_right_flow_absdiff"] = float(np.mean(np.abs(left - right)))
    return features


def merge_feature_dicts(dicts: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    """Convert feature dictionaries into a matrix with a stable schema."""
    schema = sorted({key for item in dicts for key in item})
    matrix = np.asarray([[item.get(key, 0.0) for key in schema] for item in dicts], dtype=np.float64)
    return matrix, schema


def _dominant_frequency(values: np.ndarray) -> float:
    centered = values - values.mean()
    if centered.size < 3 or np.allclose(centered, 0.0):
        return 0.0
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 1:
        return 0.0
    return float(np.argmax(spectrum[1:]) + 1)


def _spectral_entropy(values: np.ndarray) -> float:
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    probs = spectrum / total
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _autocorr_peak(values: np.ndarray) -> float:
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0.0 or centered.size < 4:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :] / denom
    return float(np.max(corr[1:])) if corr.size > 1 else 0.0


def _high_frequency_energy(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    midpoint = spectrum.size // 2
    total = float(spectrum.sum())
    return float(spectrum[midpoint:].sum() / total) if total > 0 else 0.0
