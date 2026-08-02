"""Command-line interface for visual gait quality assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gait_aqa.data.import_walker_outputs import import_walker_outputs
from gait_aqa.data.split_dataset import grouped_split
from gait_aqa.data.synthetic_motion import generate_synthetic_dataset
from gait_aqa.data.video_io import read_video
from gait_aqa.evaluation.metrics import regression_table
from gait_aqa.models.classical_regressor import IRREGULARITY_TARGETS, SCORE_TARGETS
from gait_aqa.models.model_io import load_model
from gait_aqa.training.train_classical import train_classical
from gait_aqa.vision.flow_visualization import flow_to_rgb
from gait_aqa.vision.optical_flow import cache_flow, compute_dense_flow
from gait_aqa.vision.preprocessing import preprocess_frames
from gait_aqa.vision.temporal_features import coefficient_features, flow_features, merge_feature_dicts
from gait_aqa.visualization.annotate_video import annotate_video
from gait_aqa.visualization.plots import plot_heatmap, plot_score_over_time


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_book = subparsers.add_parser("map-book", help="Create/update book mapping notes.")
    map_book.add_argument("--book", type=Path, required=False)

    synth = subparsers.add_parser("generate-synthetic", help="Generate synthetic smoke dataset.")
    synth.add_argument("--output-dir", type=Path, default=Path("data/raw/synthetic"))
    synth.add_argument("--manifest", type=Path, default=Path("data/manifests/synthetic.csv"))
    synth.add_argument("--clip-count", type=int, default=24)

    split = subparsers.add_parser("split-dataset", help="Assign leakage-safe grouped splits.")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)

    import_walker = subparsers.add_parser("import-walker", help="Import walker CSV metrics.")
    import_walker.add_argument("--walker-repo", type=Path, required=True)
    import_walker.add_argument("--output", type=Path, default=Path("data/processed"))

    extract_flow = subparsers.add_parser("extract-flow", help="Compute and cache dense flow.")
    extract_flow.add_argument("--manifest", type=Path, required=True)
    extract_flow.add_argument("--output-dir", type=Path, default=Path("data/interim/flow"))

    train = subparsers.add_parser("train-classical", help="Train the classical baseline.")
    train.add_argument("--manifest", type=Path, default=Path("data/manifests/synthetic_split.csv"))
    train.add_argument("--model", type=Path, default=Path("outputs/models/classical.pkl"))

    evaluate = subparsers.add_parser("evaluate", help="Evaluate saved predictions.")
    evaluate.add_argument("--predictions", type=Path, default=Path("outputs/predictions/classical_predictions.csv"))
    evaluate.add_argument("--split", default="test")

    score = subparsers.add_parser("score-video", help="Score one RGB video or `.npz` clip.")
    score.add_argument("--video", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--json-output", type=Path, default=Path("outputs/predictions/score.json"))
    score.add_argument("--csv-output", type=Path, default=Path("outputs/predictions/score.csv"))
    score.add_argument("--annotated-output", type=Path, default=Path("outputs/videos/scored.mp4"))

    smoke = subparsers.add_parser("reproduce-smoke", help="Run a small end-to-end demo.")

    args = parser.parse_args(argv)
    if args.command == "map-book":
        _map_book(args.book)
    elif args.command == "generate-synthetic":
        manifest = generate_synthetic_dataset(args.output_dir, args.manifest, clip_count=args.clip_count)
        print(f"Wrote {len(manifest)} synthetic clips to {args.manifest}")
    elif args.command == "split-dataset":
        manifest = grouped_split(pd.read_csv(args.manifest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output, index=False)
        print(f"Wrote split manifest to {args.output}")
    elif args.command == "import-walker":
        imported = import_walker_outputs(args.walker_repo, args.output)
        print(f"Imported {len(imported)} walker metric rows")
    elif args.command == "extract-flow":
        _extract_flow(args.manifest, args.output_dir)
    elif args.command == "train-classical":
        _, predictions = train_classical(args.manifest, args.model)
        print(f"Trained model: {args.model}")
        print(regression_table(predictions).to_string(index=False))
    elif args.command == "evaluate":
        predictions = pd.read_csv(args.predictions)
        print(regression_table(predictions, split=args.split).to_string(index=False))
    elif args.command == "score-video":
        result = score_video(args.video, args.model, args.json_output, args.csv_output, args.annotated_output)
        print(json.dumps(result, indent=2))
    elif args.command == "reproduce-smoke":
        _reproduce_smoke()


def score_video(
    video_path: Path,
    model_path: Path,
    json_output: Path,
    csv_output: Path,
    annotated_output: Path,
) -> dict[str, object]:
    """Score one video and write structured outputs."""
    model = load_model(model_path)
    frames, fps = read_video(video_path)
    features, flow = _extract_features_for_model(frames, model)
    predictions = model.predict(features)
    scores = predictions["scores"][0]
    irregularity_probs = predictions["irregularities"][0]
    component_scores = {
        "stability": float(scores[1]),
        "foot_contact_quality": float(scores[2]),
        "left_right_symmetry": float(scores[3]),
        "periodicity": float(scores[4]),
        "smoothness": float(scores[5]),
        "command_tracking": float(scores[6]),
    }
    irregularities = {
        "foot_sliding": float(irregularity_probs[0]),
        "hopping": float(irregularity_probs[1]),
        "micro_stepping": float(irregularity_probs[2]),
        "left_right_asymmetry": float(irregularity_probs[3]),
        "torso_instability": float(irregularity_probs[4]),
        "toe_dragging": float(irregularity_probs[5]),
        "fall_or_near_fall": float(irregularity_probs[6]),
        "command_ignoring": float(irregularity_probs[7]),
    }
    result = {
        "overall_score": float(scores[0]),
        "component_scores": component_scores,
        "irregularities": irregularities,
        "confidence": float(np.clip(1.0 - np.std(irregularity_probs), 0.0, 1.0)),
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{"video_path": str(video_path)}, **_flatten_result(result)}]).to_csv(csv_output, index=False)
    temporal_scores = np.full(frames.shape[0], scores[0], dtype=float)
    annotate_video(frames, temporal_scores, annotated_output, fps)
    plot_score_over_time(temporal_scores, csv_output.with_name(csv_output.stem + "_score_over_time.png"))
    plot_heatmap(np.full(frames.shape[0], max(irregularities.values())), csv_output.with_name(csv_output.stem + "_irregularity_heatmap.png"))
    Image.fromarray(flow_to_rgb(flow)).save(csv_output.with_name(csv_output.stem + "_flow.png"))
    return result


def _extract_features_for_model(frames: np.ndarray, model: object) -> tuple[np.ndarray, np.ndarray]:
    gray = preprocess_frames(frames, size=(96, 96), grayscale=True)
    flow = compute_dense_flow(gray)
    basis = model.metadata["motion_basis"]
    coefficients = basis.transform(flow)
    feature_dict = {**flow_features(flow), **coefficient_features(coefficients)}
    x, schema = merge_feature_dicts([feature_dict])
    expected = model.feature_schema
    aligned = np.asarray([[feature_dict.get(name, 0.0) for name in expected]], dtype=float)
    return aligned, flow


def _extract_flow(manifest_path: Path, output_dir: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in manifest.iterrows():
        frames, _ = read_video(row["video_path"])
        gray = preprocess_frames(frames, size=(96, 96), grayscale=True)
        flow = compute_dense_flow(gray)
        cache_flow(output_dir / f"{row['clip_id']}.npz", flow)
    print(f"Cached flow for {len(manifest)} clips in {output_dir}")


def _map_book(book: Path | None) -> None:
    if book is None or not book.exists():
        print("Book PDF not found; docs/book_mapping.md keeps TODO page markers.")
        return
    print(f"Book present: {book}. Please verify printed/PDF pages in docs/book_mapping.md.")


def _reproduce_smoke() -> None:
    manifest = Path("data/manifests/synthetic.csv")
    split_manifest = Path("data/manifests/synthetic_split.csv")
    generate_synthetic_dataset(manifest_path=manifest, clip_count=18)
    split_df = grouped_split(pd.read_csv(manifest))
    split_df.to_csv(split_manifest, index=False)
    train_classical(split_manifest)
    result = score_video(
        Path(split_df.iloc[0]["video_path"]),
        Path("outputs/models/classical.pkl"),
        Path("outputs/predictions/smoke_score.json"),
        Path("outputs/predictions/smoke_score.csv"),
        Path("outputs/videos/smoke_scored.mp4"),
    )
    print(f"Smoke overall score: {result['overall_score']:.2f}")


def _flatten_result(result: dict[str, object]) -> dict[str, float]:
    flat: dict[str, float] = {
        "overall_score": float(result["overall_score"]),
        "confidence": float(result["confidence"]),
    }
    for group in ("component_scores", "irregularities"):
        values = result[group]
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{group}.{key}"] = float(value)
    return flat


if __name__ == "__main__":
    main()
