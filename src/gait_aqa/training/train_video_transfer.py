"""Train a leakage-safe frozen R3D-18 gait-quality transfer baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait_aqa.config import load_config
from gait_aqa.data.split_dataset import assert_no_group_overlap
from gait_aqa.data.video_io import read_video
from gait_aqa.evaluation.metrics import regression_metrics
from gait_aqa.logging_utils import get_logger, setup_logging
from gait_aqa.models.classical_regressor import NumpyRidgeRegressor, StandardScaler
from gait_aqa.training.reproducibility import file_sha256, git_commit

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class VideoTransferConfig:
    """Small validated configuration for frozen-backbone transfer learning."""

    clip_frames: int = 16
    clip_windows: int = 2
    clip_duration_seconds: float = 2.0
    pca_components: int = 64
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    device: str = "auto"
    embedding_cache: str = "data/interim/embeddings/r3d18_kinetics400"
    torch_home: str = "output/torch_cache"
    seed: int = 13

    @classmethod
    def from_yaml(cls, path: str | Path) -> VideoTransferConfig:
        values = load_config(path)
        clip = values.get("clip", {})
        representation = values.get("representation", {})
        regression = values.get("regression", {})
        runtime = values.get("runtime", {})
        config = cls(
            clip_frames=int(clip.get("frames", 16)),
            clip_windows=int(clip.get("windows", 2)),
            clip_duration_seconds=float(clip.get("duration_seconds", 2.0)),
            pca_components=int(representation.get("pca_components", 64)),
            ridge_alphas=tuple(
                float(value)
                for value in regression.get(
                    "ridge_alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                )
            ),
            device=str(runtime.get("device", "auto")),
            embedding_cache=str(
                runtime.get(
                    "embedding_cache",
                    "data/interim/embeddings/r3d18_kinetics400",
                )
            ),
            torch_home=str(runtime.get("torch_home", "output/torch_cache")),
            seed=int(runtime.get("seed", 13)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.clip_frames < 2:
            raise ValueError("clip.frames must be at least 2")
        if self.clip_windows < 1:
            raise ValueError("clip.windows must be positive")
        if self.clip_duration_seconds <= 0.0:
            raise ValueError("clip.duration_seconds must be positive")
        if self.pca_components < 1:
            raise ValueError("representation.pca_components must be positive")
        if not self.ridge_alphas or any(alpha < 0.0 for alpha in self.ridge_alphas):
            raise ValueError("regression.ridge_alphas must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime.device must be auto, cpu, or cuda")


@dataclass(frozen=True)
class WhiteningBasis:
    """PCA projection learned without quality labels."""

    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        centered = np.asarray(features, dtype=np.float64) - self.mean
        return (centered @ self.components.T) / self.scales


def _sample_window_indices(
    frame_count: int,
    fps: float,
    clip_frames: int,
    clip_windows: int,
    duration_seconds: float,
) -> list[np.ndarray]:
    """Return deterministic, time-normalized indices for multiple clip windows."""
    if frame_count < 2:
        raise ValueError("A clip needs at least two decoded frames")
    safe_fps = fps if math.isfinite(fps) and fps > 0.0 else 30.0
    span = min(frame_count, max(clip_frames, round(safe_fps * duration_seconds)))
    centers = np.linspace(0.0, frame_count - 1.0, clip_windows + 2)[1:-1]
    windows: list[np.ndarray] = []
    for center in centers:
        start = round(center - (span - 1) / 2.0)
        start = max(0, min(start, frame_count - span))
        stop = start + span - 1
        indices = np.linspace(start, stop, clip_frames).round().astype(np.int64)
        windows.append(np.clip(indices, 0, frame_count - 1))
    return windows


def _fit_whitening_basis(features: np.ndarray, components: int) -> WhiteningBasis:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA features must have shape samples,features")
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    count = min(components, right_vectors.shape[0], right_vectors.shape[1])
    variances = singular_values**2 / max(values.shape[0] - 1, 1)
    total_variance = max(float(variances.sum()), 1e-12)
    selected_variances = variances[:count]
    scales = np.sqrt(np.maximum(selected_variances, 1e-12))
    return WhiteningBasis(
        mean=mean,
        components=right_vectors[:count],
        scales=scales,
        explained_variance_ratio=selected_variances / total_variance,
    )


def _resolve_video(path: str, workspace: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video does not exist: {resolved}")
    return resolved


def _embedding_cache_path(
    row: pd.Series,
    video_path: Path,
    cache_root: Path,
    config: VideoTransferConfig,
) -> Path:
    stat = video_path.stat()
    payload = {
        "video": str(video_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "frames": config.clip_frames,
        "windows": config.clip_windows,
        "duration_seconds": config.clip_duration_seconds,
        "backbone": "r3d_18_kinetics400_v1",
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    dataset = SAFE_NAME.sub("_", str(row.get("dataset", "unknown")))
    clip_id = SAFE_NAME.sub("_", str(row["clip_id"]))[:140]
    return cache_root / dataset / f"{clip_id}__{key}.npy"


def _load_backbone(config: VideoTransferConfig) -> tuple[Any, Any, Any, str]:
    os.environ["TORCH_HOME"] = str(Path(config.torch_home).resolve())
    try:
        import torch
        from torchvision.models.video import R3D_18_Weights, r3d_18
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Video transfer training requires the optional torch and torchvision "
            "dependencies"
        ) from exc

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = (
        "cuda"
        if config.device == "cuda"
        or (config.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms(), torch, device


def _extract_one_embedding(
    video_path: Path,
    model: Any,
    preprocess: Any,
    torch: Any,
    device: str,
    config: VideoTransferConfig,
) -> np.ndarray:
    frames, fps = read_video(video_path)
    indices = _sample_window_indices(
        len(frames),
        fps,
        config.clip_frames,
        config.clip_windows,
        config.clip_duration_seconds,
    )
    tensors = []
    for window in indices:
        selected = torch.from_numpy(frames[window]).permute(0, 3, 1, 2)
        tensors.append(preprocess(selected))
    batch = torch.stack(tensors).to(device, non_blocking=True)
    autocast_enabled = device == "cuda"
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device, dtype=torch.float16, enabled=autocast_enabled
        ),
    ):
        window_embeddings = model(batch)
    embedding = window_embeddings.float().mean(dim=0).cpu().numpy()
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(embedding).all() or norm <= 1e-12:
        raise ValueError(f"Invalid R3D embedding for {video_path}")
    return np.asarray(embedding / norm, dtype=np.float32)


def _extract_manifest_embeddings(
    manifest: pd.DataFrame,
    workspace: Path,
    cache_root: Path,
    model: Any,
    preprocess: Any,
    torch: Any,
    device: str,
    config: VideoTransferConfig,
    label: str,
) -> np.ndarray:
    logger = get_logger(__name__)
    embeddings: list[np.ndarray] = []
    started = time.perf_counter()
    cache_hits = 0
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        video_path = _resolve_video(str(row["video_path"]), workspace)
        cache_path = _embedding_cache_path(row, video_path, cache_root, config)
        if cache_path.exists():
            embedding = np.asarray(np.load(cache_path), dtype=np.float32)
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                cache_path.unlink()
            else:
                cache_hits += 1
                embeddings.append(embedding)
                continue
        embedding = _extract_one_embedding(
            video_path, model, preprocess, torch, device, config
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_name(f"{cache_path.stem}.partial.npy")
        np.save(partial, embedding)
        partial.replace(cache_path)
        embeddings.append(embedding)
        if position % 25 == 0 or position == len(manifest):
            elapsed = time.perf_counter() - started
            logger.info(
                "R3D embeddings {}: {}/{} cache_hits={} elapsed={:.1f}s",
                label,
                position,
                len(manifest),
                cache_hits,
                elapsed,
            )
    return np.stack(embeddings)


def _select_ridge(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[StandardScaler, NumpyRidgeRegressor, float, dict[str, float]]:
    train = splits == "train"
    validation = splits == "val"
    if not train.any() or not validation.any():
        raise ValueError("Supervised manifest needs non-empty train and val splits")
    scaler = StandardScaler().fit(features[train])
    transformed = scaler.transform(features)
    validation_mae: dict[str, float] = {}
    best_alpha = alphas[0]
    best_mae = float("inf")
    best_model: NumpyRidgeRegressor | None = None
    for alpha in alphas:
        candidate = NumpyRidgeRegressor(alpha=alpha).fit(
            transformed[train], target[train]
        )
        prediction = candidate.predict(transformed[validation])
        mae = float(np.mean(np.abs(prediction - target[validation])))
        validation_mae[str(alpha)] = mae
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = candidate
    if best_model is None:
        raise RuntimeError("Ridge selection did not produce a model")
    return scaler, best_model, best_alpha, validation_mae


def _group_validation_mae(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[str, float]:
    """Score each alpha with leave-one-policy-group-out validation."""
    unique_groups = np.unique(groups.astype(str))
    if unique_groups.size < 3:
        raise ValueError("Group validation needs at least three independent groups")
    absolute_errors: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    for validation_group in unique_groups:
        validation = groups.astype(str) == validation_group
        train = ~validation
        scaler = StandardScaler().fit(features[train])
        transformed_train = scaler.transform(features[train])
        transformed_validation = scaler.transform(features[validation])
        for alpha in alphas:
            model = NumpyRidgeRegressor(alpha=alpha).fit(
                transformed_train, target[train]
            )
            prediction = model.predict(transformed_validation)
            absolute_errors[alpha].extend(
                np.abs(prediction - target[validation]).tolist()
            )
    return {
        str(alpha): float(np.mean(errors)) for alpha, errors in absolute_errors.items()
    }


def _nested_group_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, float]]:
    """Predict every policy once using alpha selection on other policies only."""
    string_groups = groups.astype(str)
    predictions = np.full(target.shape, np.nan, dtype=float)
    selected_alphas: dict[str, float] = {}
    for test_group in np.unique(string_groups):
        test = string_groups == test_group
        train = ~test
        inner_mae = _group_validation_mae(
            features[train], target[train], string_groups[train], alphas
        )
        alpha = float(min(inner_mae, key=lambda key: inner_mae[key]))
        scaler = StandardScaler().fit(features[train])
        model = NumpyRidgeRegressor(alpha=alpha).fit(
            scaler.transform(features[train]), target[train]
        )
        predictions[test] = model.predict(scaler.transform(features[test]))
        selected_alphas[test_group] = alpha
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested group validation left non-finite predictions")
    return np.clip(predictions, 0.0, 100.0), selected_alphas


def _group_average_metrics(
    target: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Evaluate policy means after every clip prediction is already held out."""
    string_groups = groups.astype(str)
    rows: list[dict[str, float | str]] = []
    for group in np.unique(string_groups):
        selected = string_groups == group
        rows.append(
            {
                "split_group": group,
                "true_policy_score": float(np.mean(target[selected])),
                "pred_policy_score": float(np.mean(predictions[selected])),
                "scenario_count": float(selected.sum()),
            }
        )
    true_policy_scores = np.asarray(
        [row["true_policy_score"] for row in rows], dtype=float
    )
    predicted_policy_scores = np.asarray(
        [row["pred_policy_score"] for row in rows], dtype=float
    )
    return regression_metrics(true_policy_scores, predicted_policy_scores), rows


def _leave_one_group_mean_predictions(
    target: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Predict each held-out policy with the mean of all remaining policies."""
    string_groups = groups.astype(str)
    predictions = np.full(target.shape, np.nan, dtype=float)
    for test_group in np.unique(string_groups):
        test = string_groups == test_group
        predictions[test] = float(np.mean(target[~test]))
    return predictions


def _smoke_subset(manifest: pd.DataFrame, role: str) -> pd.DataFrame:
    if role == "representation":
        return (
            manifest.groupby("dataset", group_keys=False).head(4).reset_index(drop=True)
        )
    if role == "supervised":
        return (
            manifest.groupby("split", group_keys=False).head(2).reset_index(drop=True)
        )
    return manifest.head(4).reset_index(drop=True)


def train_video_transfer(
    representation_manifest: str | Path,
    supervised_manifest: str | Path,
    ood_manifest: str | Path,
    config_path: str | Path,
    model_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    ood_output_path: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Extract frozen embeddings, fit unsupervised PCA, and train Ridge AQA."""
    logger = get_logger(__name__)
    workspace = Path.cwd()
    config = VideoTransferConfig.from_yaml(config_path)
    _set_determinism(config.seed)
    representation = pd.read_csv(representation_manifest)
    supervised = pd.read_csv(supervised_manifest)
    ood = pd.read_csv(ood_manifest)
    assert_no_group_overlap(supervised)
    if smoke:
        representation = _smoke_subset(representation, "representation")
        supervised = _smoke_subset(supervised, "supervised")
        ood = _smoke_subset(ood, "ood")
    _validate_manifests(representation, supervised, ood)

    model, preprocess, torch, device = _load_backbone(config)
    logger.info(
        "Loaded frozen R3D-18 Kinetics-400 backbone device={} cuda={} config={}",
        device,
        torch.version.cuda,
        asdict(config),
    )
    cache_root = Path(config.embedding_cache)
    representation_embeddings = _extract_manifest_embeddings(
        representation,
        workspace,
        cache_root,
        model,
        preprocess,
        torch,
        device,
        config,
        "representation",
    )
    supervised_embeddings = _extract_manifest_embeddings(
        supervised,
        workspace,
        cache_root,
        model,
        preprocess,
        torch,
        device,
        config,
        "supervised",
    )
    ood_embeddings = _extract_manifest_embeddings(
        ood,
        workspace,
        cache_root,
        model,
        preprocess,
        torch,
        device,
        config,
        "ood",
    )

    basis = _fit_whitening_basis(
        representation_embeddings,
        min(config.pca_components, len(representation_embeddings) - 1),
    )
    supervised_features = basis.transform(supervised_embeddings)
    ood_features = basis.transform(ood_embeddings)
    target = pd.to_numeric(supervised["overall_score"], errors="raise").to_numpy()
    splits = supervised["split"].astype(str).to_numpy()
    scaler, ridge, selected_alpha, validation_mae = _select_ridge(
        supervised_features, target, splits, config.ridge_alphas
    )
    predictions = np.clip(
        ridge.predict(scaler.transform(supervised_features)), 0.0, 100.0
    )
    groups = supervised["split_group"].astype(str).to_numpy()
    nested_predictions, nested_alphas = _nested_group_predictions(
        supervised_features, target, groups, config.ridge_alphas
    )
    nested_policy_metrics, nested_policy_rows = _group_average_metrics(
        target, nested_predictions, groups
    )
    nested_mean_predictions = _leave_one_group_mean_predictions(target, groups)
    nested_policy_mean_metrics, _ = _group_average_metrics(
        target, nested_mean_predictions, groups
    )
    deployment_cv_mae = _group_validation_mae(
        supervised_features, target, groups, config.ridge_alphas
    )
    deployment_alpha = float(
        min(deployment_cv_mae, key=lambda key: deployment_cv_mae[key])
    )
    deployment_scaler = StandardScaler().fit(supervised_features)
    deployment_ridge = NumpyRidgeRegressor(alpha=deployment_alpha).fit(
        deployment_scaler.transform(supervised_features), target
    )
    prediction_table = supervised[
        [
            "clip_id",
            "video_path",
            "policy_run",
            "scenario",
            "split_group",
            "split",
        ]
    ].copy()
    prediction_table["true_overall_score"] = target
    prediction_table["pred_overall_score"] = predictions
    prediction_table["nested_group_cv_pred_overall_score"] = nested_predictions
    prediction_file = Path(predictions_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(prediction_file, index=False)

    train_mask = splits == "train"
    test_mask = splits == "test"
    validation_mask = splits == "val"
    mean_baseline = np.full(test_mask.sum(), target[train_mask].mean())
    test_metrics = regression_metrics(target[test_mask], predictions[test_mask])
    baseline_metrics = regression_metrics(target[test_mask], mean_baseline)
    test_policy_metrics, test_policy_rows = _group_average_metrics(
        target[test_mask], predictions[test_mask], groups[test_mask]
    )

    transformed_supervised = deployment_scaler.transform(supervised_features)
    transformed_ood = deployment_scaler.transform(ood_features)
    train_distance = np.sqrt(np.mean(transformed_supervised**2, axis=1))
    ood_distance = np.sqrt(np.mean(transformed_ood**2, axis=1))
    ood_table = ood[["clip_id", "video_path", "split_group", "variant"]].copy()
    ood_table["domain_distance"] = ood_distance
    ood_table["training_distance_p95"] = float(np.percentile(train_distance, 95))
    ood_table["quality_score_valid"] = False
    ood_file = Path(ood_output_path)
    ood_file.parent.mkdir(parents=True, exist_ok=True)
    ood_table.to_csv(ood_file, index=False)

    weights_path = (
        Path(config.torch_home) / "hub/checkpoints/r3d_18-b3b3357e.pth"
    ).resolve()
    checkpoint = {
        "format_version": 1,
        "backbone": "r3d_18",
        "weights": "KINETICS400_V1",
        "backbone_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "pca_mean": basis.mean,
        "pca_components": basis.components,
        "pca_scales": basis.scales,
        "feature_mean": deployment_scaler.mean_,
        "feature_scale": deployment_scaler.scale_,
        "ridge_coef": deployment_ridge.coef_,
        "evaluation_feature_mean": scaler.mean_,
        "evaluation_feature_scale": scaler.scale_,
        "evaluation_ridge_coef": ridge.coef_,
        "config": asdict(config),
        "metadata": {
            "project_commit": git_commit("."),
            "representation_manifest_hash": file_sha256(representation_manifest),
            "supervised_manifest_hash": file_sha256(supervised_manifest),
            "weights_sha256": file_sha256(weights_path),
            "quality_semantics": "simulator-derived research score; not clinical",
            "deployment_selected_ridge_alpha": deployment_alpha,
            "holdout_selected_ridge_alpha": selected_alpha,
            "training_domain_distance_p95": float(np.percentile(train_distance, 95)),
        },
    }
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_file)

    report = {
        "status": "smoke" if smoke else "trained",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "representation_clips": len(representation),
        "supervised_clips": len(supervised),
        "supervised_split_counts": supervised["split"].value_counts().to_dict(),
        "ood_clips": len(ood),
        "pca_components": int(basis.components.shape[0]),
        "pca_explained_variance": float(basis.explained_variance_ratio.sum()),
        "selected_ridge_alpha": selected_alpha,
        "validation_mae_by_alpha": validation_mae,
        "validation_metrics": regression_metrics(
            target[validation_mask], predictions[validation_mask]
        ),
        "test_metrics": test_metrics,
        "test_mean_baseline_metrics": baseline_metrics,
        "test_mae_skill_vs_mean": (
            1.0 - test_metrics["mae"] / baseline_metrics["mae"]
            if baseline_metrics["mae"] > 0.0
            else 0.0
        ),
        "nested_policy_cv_metrics": regression_metrics(target, nested_predictions),
        "nested_policy_aggregate_metrics": nested_policy_metrics,
        "nested_policy_aggregate_predictions": nested_policy_rows,
        "nested_policy_aggregate_mean_baseline_metrics": nested_policy_mean_metrics,
        "test_policy_aggregate_metrics": test_policy_metrics,
        "test_policy_aggregate_predictions": test_policy_rows,
        "nested_policy_cv_selected_alphas": nested_alphas,
        "deployment_selected_ridge_alpha": deployment_alpha,
        "deployment_group_cv_mae_by_alpha": deployment_cv_mae,
        "train_domain_distance_p95": float(np.percentile(train_distance, 95)),
        "disabled_ood_distance_median": float(np.median(ood_distance)),
        "disabled_ood_distance_p95": float(np.percentile(ood_distance, 95)),
        "model_path": str(model_file),
        "predictions_path": str(prediction_file),
        "ood_output_path": str(ood_file),
        "limitations": [
            "Only nine independent supervised policy/checkpoint groups.",
            "Continuous labels are simulator-derived, not expert clinical labels.",
            "DisabledGait is frontal OOD data and receives no fabricated score.",
        ],
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved transfer model={} report={}", model_file, report_file)
    return report


class VideoTransferScorer:
    """Reusable scorer that loads the backbone and deployment head once."""

    def __init__(self, model_path: str | Path) -> None:
        """Load a self-contained transfer checkpoint for repeated inference."""
        try:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transfer inference requires torch and torchvision"
            ) from exc

        self.model_path = Path(model_path).resolve()
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.config = VideoTransferConfig(**checkpoint["config"])
        _set_determinism(self.config.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = r3d_18(weights=None)
        self.model.fc = torch.nn.Identity()
        self.model.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.preprocess = R3D_18_Weights.KINETICS400_V1.transforms()
        self.torch = torch
        self.pca_mean = np.asarray(checkpoint["pca_mean"], dtype=np.float64)
        self.pca_components = np.asarray(checkpoint["pca_components"], dtype=np.float64)
        self.pca_scales = np.asarray(checkpoint["pca_scales"], dtype=np.float64)
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
        self.feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float64)
        self.ridge_coef = np.asarray(checkpoint["ridge_coef"], dtype=np.float64)
        self.domain_threshold = float(
            checkpoint["metadata"]["training_domain_distance_p95"]
        )

    def score(self, video_path: str | Path, camera: str = "side") -> dict[str, Any]:
        """Score one video and expose camera and feature-domain warnings."""
        resolved_video = Path(video_path).resolve()
        embedding = _extract_one_embedding(
            resolved_video,
            self.model,
            self.preprocess,
            self.torch,
            self.device,
            self.config,
        ).astype(np.float64)
        centered = embedding - self.pca_mean
        features = (centered @ self.pca_components.T) / self.pca_scales
        scaled = (features - self.feature_mean) / self.feature_scale
        predicted_score = float(
            np.clip(np.r_[1.0, scaled] @ self.ridge_coef, 0.0, 100.0)
        )
        distance = float(np.sqrt(np.mean(scaled**2)))
        camera_supported = camera.lower() == "side"
        return {
            "video_path": str(resolved_video),
            "scenario": resolved_video.parent.name,
            "model_path": str(self.model_path),
            "predicted_research_score": predicted_score,
            "camera": camera,
            "camera_supported": camera_supported,
            "domain_distance": distance,
            "training_domain_distance_p95": self.domain_threshold,
            "distribution_warning": distance > self.domain_threshold,
            "score_valid": camera_supported and distance <= self.domain_threshold,
            "clinical_score_valid": False,
            "score_semantics": "MuJoCo simulator-derived gait-quality proxy",
        }


def score_video_transfer(
    video_path: str | Path,
    model_path: str | Path,
    camera: str = "side",
) -> dict[str, Any]:
    """Score one video with the deployment head and expose domain warnings."""
    return VideoTransferScorer(model_path).score(video_path, camera)


def _aggregate_policy_scores(
    video_results: list[dict[str, Any]],
    expected_video_count: int,
) -> dict[str, Any]:
    """Aggregate valid scenario scores without hiding missing or OOD clips."""
    if expected_video_count < 1:
        raise ValueError("expected_video_count must be positive")
    if not video_results:
        raise ValueError("At least one video result is required")

    valid_results = [result for result in video_results if result["score_valid"]]
    discovered_scenarios = sorted(
        {str(result.get("scenario", "")) for result in video_results}
    )
    valid_scenarios = {str(result.get("scenario", "")) for result in valid_results}
    scores = np.asarray(
        [result["predicted_research_score"] for result in valid_results],
        dtype=np.float64,
    )
    scenario_set_complete = len(discovered_scenarios) == expected_video_count and len(
        discovered_scenarios
    ) == len(video_results)
    policy_score_valid = (
        scenario_set_complete
        and len(valid_scenarios) == expected_video_count
        and len(valid_scenarios) == len(video_results)
    )
    diagnostic_mean = float(scores.mean()) if len(scores) else None
    return {
        "policy_research_score": diagnostic_mean if policy_score_valid else None,
        "diagnostic_mean_of_valid_scenarios": diagnostic_mean,
        "scenario_score_std": float(scores.std()) if len(scores) else None,
        "scenario_score_min": float(scores.min()) if len(scores) else None,
        "scenario_score_max": float(scores.max()) if len(scores) else None,
        "videos_discovered": len(video_results),
        "valid_videos": len(valid_results),
        "expected_videos": expected_video_count,
        "discovered_scenarios": discovered_scenarios,
        "scenario_set_complete": scenario_set_complete,
        "policy_score_valid": policy_score_valid,
        "distribution_warning": any(
            bool(result["distribution_warning"]) for result in video_results
        ),
        "clinical_score_valid": False,
        "score_semantics": "Mean MuJoCo simulator-derived gait-quality proxy",
    }


def score_policy_transfer(
    video_paths: list[str | Path],
    model_path: str | Path,
    expected_video_count: int = 6,
) -> dict[str, Any]:
    """Score a policy across standardized side-view scenario videos."""
    if not video_paths:
        raise ValueError("No policy videos were provided")
    resolved_paths = [Path(path).resolve() for path in video_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Policy video paths must be unique")
    policy_ids = {path.parents[1].name for path in resolved_paths}
    if len(policy_ids) != 1:
        raise ValueError("All videos must belong to the same policy directory")

    scorer = VideoTransferScorer(model_path)
    video_results = [scorer.score(path, camera="side") for path in resolved_paths]
    aggregate = _aggregate_policy_scores(video_results, expected_video_count)
    return {
        "policy_id": next(iter(policy_ids)),
        "model_path": str(Path(model_path).resolve()),
        **aggregate,
        "scenario_results": video_results,
    }


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_manifests(
    representation: pd.DataFrame,
    supervised: pd.DataFrame,
    ood: pd.DataFrame,
) -> None:
    for name, manifest, required in (
        (
            "representation",
            representation,
            {"dataset", "clip_id", "video_path", "split_group"},
        ),
        (
            "supervised",
            supervised,
            {
                "clip_id",
                "video_path",
                "split_group",
                "split",
                "overall_score",
            },
        ),
        (
            "ood",
            ood,
            {"clip_id", "video_path", "split_group", "variant"},
        ),
    ):
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
        if manifest.empty or manifest["clip_id"].duplicated().any():
            raise ValueError(f"{name} manifest is empty or has duplicate clip IDs")
    if set(supervised["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("Supervised manifest must contain train, val, and test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation-manifest",
        type=Path,
        default=Path("data/manifests/side_representation_pretrain.csv"),
    )
    parser.add_argument(
        "--supervised-manifest",
        type=Path,
        default=Path("data/manifests/side_aqa_supervised.csv"),
    )
    parser.add_argument(
        "--ood-manifest",
        type=Path,
        default=Path("data/manifests/front_disabled_ood.csv"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/video_transfer.yaml")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("output/models/r3d18_transfer_side.pt")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("output/predictions/r3d18_transfer_side.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/predictions/r3d18_transfer_side_report.json"),
    )
    parser.add_argument(
        "--ood-output",
        type=Path,
        default=Path("output/predictions/disabled_ood_distances.csv"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("output/logs/video_transfer.log"),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_file, args.log_level)
    report = train_video_transfer(
        args.representation_manifest,
        args.supervised_manifest,
        args.ood_manifest,
        args.config,
        args.model,
        args.predictions,
        args.report,
        args.ood_output,
        smoke=args.smoke,
    )
    print(json.dumps(report, indent=2))


def score_main() -> None:
    parser = argparse.ArgumentParser(description="Score one video with transfer model")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path("output/models/r3d18_transfer_side.pt")
    )
    parser.add_argument(
        "--camera", choices=["side", "front", "front_oblique"], default="side"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/predictions/video_transfer_score.json"),
    )
    args = parser.parse_args()
    result = score_video_transfer(args.video, args.model, args.camera)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def score_policy_main() -> None:
    """Score all side-view scenario videos under one policy directory."""
    parser = argparse.ArgumentParser(
        description="Score a MuJoCo policy from standardized side-view rollouts"
    )
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path("output/models/r3d18_transfer_side.pt")
    )
    parser.add_argument("--expected-videos", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/predictions/policy_transfer_score.json"),
    )
    args = parser.parse_args()
    if not args.policy_dir.is_dir():
        parser.error(f"Policy directory does not exist: {args.policy_dir}")
    video_paths = sorted(args.policy_dir.rglob("*__side.mp4"))
    if not video_paths:
        parser.error(f"No *__side.mp4 videos found under: {args.policy_dir}")
    result = score_policy_transfer(
        video_paths,
        args.model,
        expected_video_count=args.expected_videos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
