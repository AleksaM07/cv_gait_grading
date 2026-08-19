"""Temporal features from PCA coefficients and regional flow."""

from __future__ import annotations

import numpy as np


def coefficient_features(
    coefficients: np.ndarray,
    sample_rate_hz: float,
    prefix: str = "pca",
) -> dict[str, float]:
    """Summarize temporal PCA coefficient sequences."""
    if coefficients.ndim != 2:
        raise ValueError("Coefficients must have shape T,C")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    features: dict[str, float] = {}
    for component in range(coefficients.shape[1]):
        values = coefficients[:, component]
        name = f"{prefix}{component:02d}"
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_std"] = float(values.std())
        features[f"{name}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{name}_range"] = float(
            np.percentile(values, 95) - np.percentile(values, 5)
        )
        delta = np.diff(values)
        features[f"{name}_velocity_rms"] = (
            float(np.sqrt(np.mean((delta * sample_rate_hz) ** 2)))
            if delta.size
            else 0.0
        )
        features[f"{name}_dominant_freq_hz"] = _dominant_frequency(
            values, sample_rate_hz
        )
        features[f"{name}_spectral_entropy"] = _spectral_entropy(values)
        features[f"{name}_autocorr_peak"] = _autocorr_peak(values)
        duration_seconds = max((values.size - 1) / sample_rate_hz, 1.0 / sample_rate_hz)
        features[f"{name}_zero_crossing_rate_hz"] = float(
            np.count_nonzero(np.diff(np.signbit(values - values.mean())))
            / duration_seconds
        )
    return features


def flow_features(
    flow: np.ndarray,
    sample_rate_hz: float,
    high_frequency_hz: float = 3.0,
) -> dict[str, float]:
    """Summarize dense residual-flow magnitudes over time."""
    magnitude = np.linalg.norm(flow, axis=-1)
    features = {
        "flow_mean": float(magnitude.mean()),
        "flow_std": float(magnitude.std()),
        "flow_p95": float(np.percentile(magnitude, 95)),
        "flow_high_freq_energy": _high_frequency_energy(
            magnitude.mean(axis=(1, 2)), sample_rate_hz, high_frequency_hz
        ),
    }
    return features


def merge_feature_dicts(dicts: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    """Convert feature dictionaries into a matrix with a stable schema."""
    if not dicts:
        raise ValueError("At least one feature dictionary is required")
    expected = set(dicts[0])
    for index, item in enumerate(dicts[1:], start=1):
        if set(item) != expected:
            missing = sorted(expected - set(item))
            extra = sorted(set(item) - expected)
            raise ValueError(
                f"Feature schema mismatch at row {index}: missing={missing}, extra={extra}"
            )
    schema = sorted(expected)
    matrix = np.asarray(
        [[item[key] for key in schema] for item in dicts], dtype=np.float64
    )
    return matrix, schema


def align_feature_dict(
    features: dict[str, float], expected_schema: list[str]
) -> np.ndarray:
    """Build one feature row and reject incompatible model schemas."""
    missing = sorted(set(expected_schema) - set(features))
    if missing:
        raise ValueError(
            f"Model expects features unavailable in this pipeline version: {missing}"
        )
    return np.asarray([[features[name] for name in expected_schema]], dtype=float)


def _dominant_frequency(values: np.ndarray, sample_rate_hz: float) -> float:
    centered = values - values.mean()
    if centered.size < 3 or np.allclose(centered, 0.0):
        return 0.0
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 1:
        return 0.0
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    return float(frequencies[np.argmax(spectrum[1:]) + 1])


def _spectral_entropy(values: np.ndarray) -> float:
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    probs = spectrum / total
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    maximum = np.log2(probs.size) if probs.size > 1 else 1.0
    return entropy / maximum


def _autocorr_peak(values: np.ndarray) -> float:
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0.0 or centered.size < 4:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :] / denom
    return float(np.max(corr[1:])) if corr.size > 1 else 0.0


def _high_frequency_energy(
    values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
) -> float:
    if values.size < 4:
        return 0.0
    if not 0.0 < cutoff_hz < sample_rate_hz / 2.0:
        raise ValueError("high-frequency cutoff must be between 0 and Nyquist")
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate_hz)
    total = float(spectrum.sum())
    return float(spectrum[frequencies >= cutoff_hz].sum() / total) if total > 0 else 0.0
