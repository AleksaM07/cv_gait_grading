"""Command-line interface for visual gait quality assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from gait_aqa.data.build_manifest import prepare_real_video_manifest
from gait_aqa.data.import_walker_outputs import import_walker_outputs
from gait_aqa.data.split_dataset import grouped_split
from gait_aqa.data.video_io import read_video
from gait_aqa.evaluation.metrics import regression_table
from gait_aqa.logging_utils import get_logger, setup_logging
from gait_aqa.models.classical_regressor import (
    IRREGULARITY_TARGETS,
    SCORE_TARGETS,
    ClassicalGaitModel,
)
from gait_aqa.models.model_io import load_model
from gait_aqa.training.train_classical import ClassicalTrainingConfig, train_classical
from gait_aqa.vision.flow_visualization import flow_to_rgb
from gait_aqa.vision.optical_flow import cache_flow, compute_dense_flow
from gait_aqa.vision.preprocessing import preprocess_frames, resample_frames
from gait_aqa.vision.temporal_features import (
    align_feature_dict,
    coefficient_features,
    flow_features,
)
from gait_aqa.visualization.annotate_video import annotate_video
from gait_aqa.visualization.plots import plot_heatmap


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("output/logs/gait_aqa.log"),
        help="Path where CLI logs are appended/rotated.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console and file logging level.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_book = subparsers.add_parser(
        "map-book", help="Create/update book mapping notes."
    )
    map_book.add_argument("--book", type=Path, required=False)

    split = subparsers.add_parser(
        "split-dataset", help="Assign leakage-safe grouped splits."
    )
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)

    prepare_real = subparsers.add_parser(
        "prepare-real-manifest",
        help="Convert rendered walker video manifest into training schema.",
    )
    prepare_real.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("MUJOCO_videos_better/manifest.csv"),
    )
    prepare_real.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("MUJOCO_videos_better"),
    )
    prepare_real.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/better_videos_side_split.csv"),
    )
    prepare_real.add_argument(
        "--camera",
        default="side",
        help="Camera view to keep; use 'all' to retain every camera.",
    )

    import_walker = subparsers.add_parser(
        "import-walker", help="Import walker CSV metrics."
    )
    import_walker.add_argument("--walker-repo", type=Path, required=True)
    import_walker.add_argument("--output", type=Path, default=Path("data/processed"))

    extract_flow = subparsers.add_parser(
        "extract-flow", help="Compute and cache dense flow."
    )
    extract_flow.add_argument("--manifest", type=Path, required=True)
    extract_flow.add_argument(
        "--output-dir", type=Path, default=Path("data/interim/flow")
    )
    extract_flow.add_argument(
        "--config", type=Path, default=Path("configs/classical.yaml")
    )

    train = subparsers.add_parser(
        "train-classical", help="Train the classical baseline."
    )
    train.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/better_videos_side_split.csv"),
    )
    train.add_argument(
        "--model", type=Path, default=Path("output/models/classical_side.pkl")
    )
    train.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    train.add_argument(
        "--predictions",
        type=Path,
        default=Path("output/predictions/classical_side.csv"),
    )

    evaluate = subparsers.add_parser("evaluate", help="Evaluate saved predictions.")
    evaluate.add_argument(
        "--predictions",
        type=Path,
        default=Path("output/predictions/classical_side.csv"),
    )
    evaluate.add_argument("--split", default="test")

    score = subparsers.add_parser(
        "score-video", help="Score one RGB video or `.npz` clip."
    )
    score.add_argument("--video", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument(
        "--json-output", type=Path, default=Path("output/predictions/score.json")
    )
    score.add_argument(
        "--csv-output", type=Path, default=Path("output/predictions/score.csv")
    )
    score.add_argument(
        "--annotated-output", type=Path, default=Path("output/videos/scored.mp4")
    )

    args = parser.parse_args(argv)
    logger = setup_logging(args.log_file, args.log_level)
    logger.info("Running command: {}", args.command)
    if args.command == "map-book":
        _map_book(args.book)
    elif args.command == "split-dataset":
        logger.info("Splitting manifest: {}", args.manifest)
        manifest = grouped_split(pd.read_csv(args.manifest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output, index=False)
        logger.info("Split manifest written: {}", args.output)
        print(f"Wrote split manifest to {args.output}")
    elif args.command == "prepare-real-manifest":
        logger.info(
            "Preparing real video manifest: input={} root={}",
            args.input_manifest,
            args.dataset_root,
        )
        manifest = prepare_real_video_manifest(
            args.input_manifest,
            args.output,
            args.dataset_root,
            camera=None if args.camera.lower() == "all" else args.camera,
        )
        logger.info("Prepared real manifest: {} rows={}", args.output, len(manifest))
        print(f"Wrote real video manifest with {len(manifest)} rows to {args.output}")
    elif args.command == "import-walker":
        logger.info("Importing walker outputs from {}", args.walker_repo)
        imported = import_walker_outputs(args.walker_repo, args.output)
        logger.info("Imported walker metric rows: {}", len(imported))
        print(f"Imported {len(imported)} walker metric rows")
    elif args.command == "extract-flow":
        _extract_flow(args.manifest, args.output_dir, args.config)
    elif args.command == "train-classical":
        logger.info(
            "Training classical model: manifest={} model={}", args.manifest, args.model
        )
        _, predictions = train_classical(
            args.manifest,
            args.model,
            predictions_path=args.predictions,
            config_path=args.config,
        )
        logger.info("Classical training complete: predictions={}", len(predictions))
        print(f"Trained model: {args.model}")
        print(regression_table(predictions).to_string(index=False))
    elif args.command == "evaluate":
        logger.info("Evaluating predictions: {} split={}", args.predictions, args.split)
        predictions = pd.read_csv(args.predictions)
        table = regression_table(predictions, split=args.split)
        logger.info("Evaluation complete: {} metric rows", len(table))
        print(table.to_string(index=False))
    elif args.command == "score-video":
        logger.info("Scoring video: {}", args.video)
        result = score_video(
            args.video,
            args.model,
            args.json_output,
            args.csv_output,
            args.annotated_output,
        )
        logger.info(
            "Score complete: overall={:.3f} json={}",
            result["overall_score"],
            args.json_output,
        )
        print(json.dumps(result, indent=2))


def score_video(
    video_path: Path,
    model_path: Path,
    json_output: Path,
    csv_output: Path,
    annotated_output: Path,
) -> dict[str, Any]:
    """Score one video and write structured outputs."""
    logger = get_logger()
    logger.info("Loading model: {}", model_path)
    model = load_model(model_path)
    frames, fps = read_video(video_path)
    logger.info("Read video: {} frames={} fps={}", video_path, frames.shape[0], fps)
    features, flow = _extract_features_for_model(frames, fps, model)
    predictions = model.predict(features)
    scores = predictions["scores"][0]
    irregularity_probs = predictions["irregularities"][0]
    component_scores = {
        target.removesuffix("_score"): _finite_or_none(scores[index])
        for index, target in enumerate(SCORE_TARGETS)
        if target != "overall_score"
    }
    irregularities = {
        target.removesuffix("_label"): _finite_or_none(irregularity_probs[index])
        for index, target in enumerate(IRREGULARITY_TARGETS)
    }
    result = {
        "overall_score": float(scores[0]),
        "component_scores": component_scores,
        "irregularities": irregularities,
        "confidence": _prediction_confidence(model, features),
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"video_path": str(video_path), **_flatten_result(result)}]).to_csv(
        csv_output, index=False
    )
    clip_scores = np.full(frames.shape[0], scores[0], dtype=float)
    annotate_video(frames, clip_scores, annotated_output, fps)
    motion_energy = np.linalg.norm(flow, axis=-1).mean(axis=(1, 2))
    scale = float(np.percentile(motion_energy, 95))
    normalized_motion = motion_energy / max(scale, 1e-8)
    plot_heatmap(
        np.clip(normalized_motion, 0.0, 1.0),
        csv_output.with_name(csv_output.stem + "_motion_heatmap.png"),
        title="Temporal motion energy",
    )
    flow_image = Image.fromarray(flow_to_rgb(flow)).resize(
        (384, 384), Image.Resampling.BILINEAR
    )
    flow_image.save(csv_output.with_name(csv_output.stem + "_flow.png"))
    logger.info(
        "Wrote scoring artifacts: json={} csv={} annotated={}",
        json_output,
        csv_output,
        annotated_output,
    )
    return result


def _extract_features_for_model(
    frames: np.ndarray,
    source_fps: float,
    model: ClassicalGaitModel,
) -> tuple[np.ndarray, np.ndarray]:
    settings = model.metadata.get("training_config", {})
    target_fps = float(settings.get("target_fps", 20.0))
    frame_width = int(settings.get("frame_width", 96))
    frame_height = int(settings.get("frame_height", 96))
    flow_mode = str(settings.get("flow_mode", "body_centered_residual_flow"))
    high_frequency_hz = float(settings.get("high_frequency_hz", 3.0))
    sampled = resample_frames(frames, source_fps, target_fps)
    gray = preprocess_frames(sampled, size=(frame_width, frame_height), grayscale=True)
    flow = compute_dense_flow(gray, mode=flow_mode)
    basis = model.metadata["motion_basis"]
    coefficients = basis.transform(flow)
    feature_dict = {
        **flow_features(flow, target_fps, high_frequency_hz),
        **coefficient_features(coefficients, target_fps),
    }
    aligned = align_feature_dict(feature_dict, model.feature_schema)
    return aligned, flow


def _extract_flow(
    manifest_path: Path,
    output_dir: Path,
    config_path: Path,
) -> None:
    logger = get_logger()
    manifest = pd.read_csv(manifest_path)
    config = ClassicalTrainingConfig.from_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in manifest.iterrows():
        frames, source_fps = read_video(row["video_path"])
        sampled = resample_frames(frames, source_fps, config.target_fps)
        gray = preprocess_frames(
            sampled,
            size=(config.frame_width, config.frame_height),
            grayscale=True,
        )
        flow = compute_dense_flow(gray, mode=config.flow_mode)
        cache_flow(output_dir / f"{row['clip_id']}.npz", flow)
        logger.debug("Cached flow for clip {}", row["clip_id"])
    logger.info("Cached flow for {} clips in {}", len(manifest), output_dir)
    print(f"Cached flow for {len(manifest)} clips in {output_dir}")


def _map_book(book: Path | None) -> None:
    if book is None or not book.exists():
        print("Book PDF not found; pass --book to verify the documented page mapping.")
        return
    print(
        "Book present:"
        f" {book}. Please verify printed/PDF pages in "
        "reports/assets/docs/book_mapping.md."
    )


def _flatten_result(result: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {
        "overall_score": float(result["overall_score"]),
    }
    confidence = result.get("confidence")
    if confidence is not None:
        flat["confidence"] = float(confidence)
    for group in ("component_scores", "irregularities"):
        values = result[group]
        if isinstance(values, dict):
            for key, value in values.items():
                if value is not None:
                    flat[f"{group}.{key}"] = float(value)
    return flat


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _prediction_confidence(
    model: ClassicalGaitModel, features: np.ndarray
) -> float | None:
    """Return a validation- and familiarity-based confidence heuristic."""
    metadata = model.metadata
    error_p90 = metadata.get("validation_overall_error_p90")
    distance_p95 = metadata.get("training_distance_p95")
    if error_p90 is None or distance_p95 is None or float(distance_p95) <= 0.0:
        return None
    scaled = model.scaler.transform(features)
    distance = float(np.sqrt(np.mean(scaled**2)))
    excess_distance = max(0.0, distance / float(distance_p95) - 1.0)
    validation_factor = np.clip(1.0 - float(error_p90) / 100.0, 0.0, 1.0)
    familiarity_factor = np.exp(-excess_distance)
    return float(validation_factor * familiarity_factor)


if __name__ == "__main__":
    main()
