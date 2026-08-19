"""Train the Szeliski-aligned classical baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait_aqa.config import load_config
from gait_aqa.data.split_dataset import assert_no_group_overlap, grouped_split
from gait_aqa.data.video_io import read_video
from gait_aqa.evaluation.metrics import classification_table, regression_table
from gait_aqa.logging_utils import get_logger
from gait_aqa.models.classical_regressor import (
    IRREGULARITY_TARGETS,
    SCORE_TARGETS,
    fit_classical_model,
)
from gait_aqa.models.model_io import save_model
from gait_aqa.training.reproducibility import file_sha256, git_commit
from gait_aqa.vision.motion_basis import MotionBasis
from gait_aqa.vision.optical_flow import (
    cache_flow,
    compute_dense_flow,
    load_cached_flow,
)
from gait_aqa.vision.preprocessing import preprocess_frames, resample_frames
from gait_aqa.vision.temporal_features import (
    coefficient_features,
    flow_features,
    merge_feature_dicts,
)


@dataclass(frozen=True)
class ClassicalTrainingConfig:
    """Validated settings used by feature extraction and model fitting."""

    target_fps: float = 20.0
    frame_width: int = 96
    frame_height: int = 96
    flow_mode: str = "body_centered_residual_flow"
    flow_cache_dir: str = "data/interim/flow/classical"
    max_components: int = 8
    explained_variance: float | None = 0.95
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    high_frequency_hz: float = 3.0
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 13

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> ClassicalTrainingConfig:
        """Load and validate the small classical-training YAML schema."""
        values = load_config(path)
        flow = values.get("flow", {})
        basis = values.get("motion_basis", {})
        model = values.get("model", {})
        temporal = values.get("temporal", {})
        split = values.get("split", {})
        size = flow.get("size", [96, 96])
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("flow.size must be [width, height]")
        config = cls(
            target_fps=float(flow.get("target_fps", 20.0)),
            frame_width=int(size[0]),
            frame_height=int(size[1]),
            flow_mode=str(flow.get("mode", "body_centered_residual_flow")),
            flow_cache_dir=str(flow.get("cache_dir", "data/interim/flow/classical")),
            max_components=int(
                basis.get("max_components", basis.get("n_components", 8))
            ),
            explained_variance=(
                None
                if basis.get("explained_variance") is None
                else float(basis["explained_variance"])
            ),
            ridge_alphas=tuple(
                float(value)
                for value in model.get("ridge_alphas", [model.get("ridge_alpha", 1.0)])
            ),
            high_frequency_hz=float(temporal.get("high_frequency_hz", 3.0)),
            train_fraction=float(split.get("train_fraction", 0.70)),
            val_fraction=float(split.get("val_fraction", 0.15)),
            seed=int(split.get("random_seed", 13)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise a clear error for unsafe or inconsistent settings."""
        if self.target_fps <= 0.0:
            raise ValueError("target_fps must be positive")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("flow image dimensions must be positive")
        if self.flow_mode not in {"absolute_flow", "body_centered_residual_flow"}:
            raise ValueError(f"Unsupported flow mode: {self.flow_mode}")
        if self.max_components <= 0:
            raise ValueError("max_components must be positive")
        if self.explained_variance is not None and not (
            0.0 < self.explained_variance <= 1.0
        ):
            raise ValueError("explained_variance must be in (0, 1]")
        if not self.ridge_alphas or any(alpha < 0.0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain non-negative values")
        if not 0.0 < self.high_frequency_hz < self.target_fps / 2.0:
            raise ValueError("high_frequency_hz must be below the Nyquist frequency")


def train_classical(
    manifest_path: str | Path,
    model_path: str | Path = "output/models/classical_side.pkl",
    predictions_path: str | Path = "output/predictions/classical_side.csv",
    config_path: str | Path | None = "configs/classical.yaml",
) -> tuple[Any, pd.DataFrame]:
    """Train the classical model, then save clip predictions and metrics."""
    logger = get_logger(__name__)
    manifest_file = Path(manifest_path)
    config = ClassicalTrainingConfig.from_yaml(config_path)
    manifest = pd.read_csv(manifest_file)
    _validate_manifest(manifest, manifest_file)
    if "split" not in manifest:
        logger.info("Manifest has no split column; creating grouped split")
        manifest = grouped_split(
            manifest,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )
    assert_no_group_overlap(manifest)
    train_indices = np.flatnonzero(manifest["split"].to_numpy() == "train")
    if train_indices.size == 0:
        raise ValueError("Training split is empty")
    logger.info(
        "Training from manifest={} rows={} splits={} config={}",
        manifest_file,
        len(manifest),
        manifest["split"].value_counts().to_dict(),
        asdict(config),
    )

    cache_paths = _prepare_flow_cache(manifest, manifest_file, config)
    basis = MotionBasis(
        n_components=config.max_components,
        explained_variance=config.explained_variance,
    ).fit(load_cached_flow(cache_paths[index]) for index in train_indices)
    if basis.components_ is None or basis.explained_variance_ratio_ is None:
        raise RuntimeError("Motion basis fit did not produce components")
    logger.info(
        "Fitted incremental motion basis on {} train clips: components={} "
        "explained_variance={:.4f}",
        train_indices.size,
        basis.components_.shape[0],
        float(np.sum(basis.explained_variance_ratio_)),
    )

    feature_dicts: list[dict[str, float]] = []
    for cache_path in cache_paths:
        flow = load_cached_flow(cache_path)
        coefficients = basis.transform(flow)
        feature_dicts.append(
            {
                **flow_features(
                    flow,
                    sample_rate_hz=config.target_fps,
                    high_frequency_hz=config.high_frequency_hz,
                ),
                **coefficient_features(coefficients, sample_rate_hz=config.target_fps),
            }
        )
    x, feature_schema = merge_feature_dicts(feature_dicts)
    y_scores = manifest[SCORE_TARGETS].to_numpy(dtype=float)
    y_irregularities = manifest[IRREGULARITY_TARGETS].to_numpy(dtype=float)

    metadata: dict[str, Any] = {
        "project_commit": git_commit("."),
        "dataset_manifest_hash": file_sha256(manifest_file),
        "feature_schema": feature_schema,
        "training_config": asdict(config),
        "book_sections_used": ["Szeliski Sec. 8.2.2", "Szeliski Sec. 8.4"],
        "label_formula": (
            "source composite = 0.40 stability + 0.30 tracking + "
            "0.20 upright + 0.10 smoothness"
        ),
        "third_party_sources": [
            "datasets/README.md",
            "reports/assets/docs/provenance.csv",
        ],
        "motion_basis": basis,
        "available_score_targets": [
            name
            for index, name in enumerate(SCORE_TARGETS)
            if np.isfinite(y_scores[:, index]).any()
        ],
        "available_irregularity_targets": [
            name
            for index, name in enumerate(IRREGULARITY_TARGETS)
            if np.isfinite(y_irregularities[:, index]).any()
        ],
    }
    if config_path is not None:
        metadata["training_config_hash"] = file_sha256(config_path)

    model = _select_ridge_model(
        x,
        y_scores,
        y_irregularities,
        manifest["split"].to_numpy(),
        feature_schema,
        config.ridge_alphas,
        metadata,
    )
    scaled_train = model.scaler.transform(x[train_indices])
    train_distances = np.sqrt(np.mean(scaled_train**2, axis=1))
    model.metadata["training_distance_p95"] = float(np.percentile(train_distances, 95))

    predictions = model.predict(x)
    output = manifest[["clip_id", "split"]].copy()
    for index, target in enumerate(SCORE_TARGETS):
        output[f"true_{target}"] = y_scores[:, index]
        output[f"pred_{target}"] = predictions["scores"][:, index]
    for index, target in enumerate(IRREGULARITY_TARGETS):
        output[f"true_{target}"] = y_irregularities[:, index]
        output[f"pred_{target}"] = predictions["irregularities"][:, index]

    _add_validation_calibration(model, output)
    metrics = regression_table(output, split="test")
    output.attrs["metrics"] = metrics
    save_model(model, model_path)
    output_file = Path(predictions_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    metrics.to_csv(
        output_file.with_name(output_file.stem + "_metrics.csv"), index=False
    )
    classification_table(output, split="test").to_csv(
        output_file.with_name(output_file.stem + "_classification_metrics.csv"),
        index=False,
    )
    logger.info("Saved classical model: {}", model_path)
    logger.info("Saved predictions: {}", output_file)
    return model, output


def _prepare_flow_cache(
    manifest: pd.DataFrame,
    manifest_path: Path,
    config: ClassicalTrainingConfig,
) -> list[Path]:
    logger = get_logger(__name__)
    cache_dir = Path(config.flow_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for position, row in manifest.reset_index(drop=True).iterrows():
        video_path = _resolve_video_path(str(row["video_path"]), manifest_path)
        cache_path = (
            cache_dir / f"{row['clip_id']}__{_cache_key(video_path, config)}.npz"
        )
        if not cache_path.exists():
            frames, source_fps = read_video(video_path)
            sampled = resample_frames(frames, source_fps, config.target_fps)
            gray = preprocess_frames(
                sampled,
                size=(config.frame_width, config.frame_height),
                grayscale=True,
            )
            flow = compute_dense_flow(gray, mode=config.flow_mode)
            cache_flow(cache_path, flow)
        paths.append(cache_path)
        if (position + 1) % 10 == 0 or position + 1 == len(manifest):
            logger.info("Flow cache ready: {}/{}", position + 1, len(manifest))
    return paths


def _resolve_video_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    candidates = [path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Video from manifest does not exist: {raw_path}")


def _cache_key(path: Path, config: ClassicalTrainingConfig) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "target_fps": config.target_fps,
        "frame_size": [config.frame_width, config.frame_height],
        "flow_mode": config.flow_mode,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:12]


def _validate_manifest(manifest: pd.DataFrame, manifest_path: Path) -> None:
    required = {
        "clip_id",
        "video_path",
        "split_group",
        *SCORE_TARGETS,
        *IRREGULARITY_TARGETS,
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    if manifest["clip_id"].duplicated().any():
        duplicates = manifest.loc[manifest["clip_id"].duplicated(), "clip_id"].tolist()
        raise ValueError(f"Manifest has duplicate clip IDs: {duplicates[:5]}")
    for target in SCORE_TARGETS:
        values = pd.to_numeric(manifest[target], errors="coerce")
        invalid = values.notna() & ~values.between(0.0, 100.0)
        if invalid.any():
            raise ValueError(f"Score target is outside [0, 100]: {target}")


def _add_validation_calibration(model: Any, predictions: pd.DataFrame) -> None:
    validation = predictions[predictions["split"] == "val"]
    if validation.empty:
        return
    true = validation["true_overall_score"].to_numpy(dtype=float)
    pred = validation["pred_overall_score"].to_numpy(dtype=float)
    available = np.isfinite(true) & np.isfinite(pred)
    if available.sum() < 10:
        model.metadata["confidence_calibration"] = (
            "unavailable: fewer than 10 finite validation clips"
        )
        return
    residual = np.abs(pred[available] - true[available])
    model.metadata["validation_overall_mae"] = float(residual.mean())
    model.metadata["validation_overall_error_p90"] = float(np.percentile(residual, 90))


def _select_ridge_model(
    x: np.ndarray,
    y_scores: np.ndarray,
    y_irregularities: np.ndarray,
    splits: np.ndarray,
    feature_schema: list[str],
    alphas: tuple[float, ...],
    metadata: dict[str, Any],
) -> Any:
    """Choose Ridge regularization on validation overall-score MAE only."""
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "val")
    if validation_indices.size == 0:
        selected = alphas[0]
        metadata["ridge_selection"] = "first configured alpha; no validation split"
        metadata["selected_ridge_alpha"] = selected
        return fit_classical_model(
            x[train_indices],
            y_scores[train_indices],
            y_irregularities[train_indices],
            feature_schema,
            alpha=selected,
            metadata=metadata,
        )

    validation_results: dict[str, float] = {}
    best_model: Any = None
    best_alpha = alphas[0]
    best_mae = float("inf")
    for alpha in alphas:
        candidate = fit_classical_model(
            x[train_indices],
            y_scores[train_indices],
            y_irregularities[train_indices],
            feature_schema,
            alpha=alpha,
            metadata=metadata,
        )
        prediction = candidate.predict(x[validation_indices])["scores"][:, 0]
        truth = y_scores[validation_indices, 0]
        available = np.isfinite(truth) & np.isfinite(prediction)
        mae = float(np.mean(np.abs(prediction[available] - truth[available])))
        validation_results[str(alpha)] = mae
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = candidate
    if best_model is None:
        raise RuntimeError("Ridge validation did not produce a model")
    metadata["ridge_validation_overall_mae"] = validation_results
    metadata["selected_ridge_alpha"] = best_alpha
    get_logger(__name__).info(
        "Selected Ridge alpha={} from validation MAE grid={}",
        best_alpha,
        validation_results,
    )
    return best_model
