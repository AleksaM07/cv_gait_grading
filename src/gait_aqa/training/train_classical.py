"""Train the Szeliski-aligned classical baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait_aqa.data.split_dataset import grouped_split
from gait_aqa.data.video_io import read_video
from gait_aqa.evaluation.metrics import regression_table
from gait_aqa.models.classical_regressor import (
    IRREGULARITY_TARGETS,
    SCORE_TARGETS,
    fit_classical_model,
)
from gait_aqa.models.model_io import save_model
from gait_aqa.training.reproducibility import file_sha256, git_commit
from gait_aqa.vision.motion_basis import MotionBasis
from gait_aqa.vision.optical_flow import compute_dense_flow
from gait_aqa.vision.preprocessing import preprocess_frames
from gait_aqa.vision.temporal_features import coefficient_features, flow_features, merge_feature_dicts


def train_classical(
    manifest_path: str | Path,
    model_path: str | Path = "outputs/models/classical.pkl",
    predictions_path: str | Path = "outputs/predictions/classical_predictions.csv",
    n_components: int = 8,
    alpha: float = 1.0,
    seed: int = 13,
) -> tuple[Any, pd.DataFrame]:
    """Train the classical model and save predictions."""
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest:
        manifest = grouped_split(manifest, seed=seed)

    flows: list[np.ndarray] = []
    for path in manifest["video_path"]:
        frames, _ = read_video(path)
        gray = preprocess_frames(frames, size=(96, 96), grayscale=True)
        flows.append(compute_dense_flow(gray))

    train_indices = np.flatnonzero(manifest["split"].to_numpy() == "train")
    basis = MotionBasis(n_components=n_components).fit([flows[i] for i in train_indices])
    feature_dicts = []
    for flow in flows:
        coeffs = basis.transform(flow)
        feature_dicts.append({**flow_features(flow), **coefficient_features(coeffs)})
    x, feature_schema = merge_feature_dicts(feature_dicts)
    y_scores = manifest[SCORE_TARGETS].to_numpy(dtype=float)
    y_irregularities = manifest[IRREGULARITY_TARGETS].to_numpy(dtype=float)

    model = fit_classical_model(
        x[train_indices],
        y_scores[train_indices],
        y_irregularities[train_indices],
        feature_schema,
        alpha=alpha,
        metadata={
            "project_commit": git_commit("."),
            "walker_commit": "33eaa1fb76f2ffb0fb8a821deb9cad27f3989426",
            "dataset_manifest_hash": file_sha256(manifest_path),
            "feature_schema": feature_schema,
            "book_sections_used": ["Szeliski Sec. 8.2.2", "Szeliski Sec. 8.4"],
            "third_party_sources": ["docs/provenance.csv"],
            "motion_basis": basis,
        },
    )
    predictions = model.predict(x)
    output = manifest[["clip_id", "split"]].copy()
    for index, target in enumerate(SCORE_TARGETS):
        output[f"true_{target}"] = y_scores[:, index]
        output[f"pred_{target}"] = predictions["scores"][:, index]
    for index, target in enumerate(IRREGULARITY_TARGETS):
        output[f"true_{target}"] = y_irregularities[:, index]
        output[f"pred_{target}"] = predictions["irregularities"][:, index]
    metrics = regression_table(output, split="test")
    output.attrs["metrics"] = metrics
    save_model(model, model_path)
    output_file = Path(predictions_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    metrics.to_csv(output_file.with_name(output_file.stem + "_metrics.csv"), index=False)
    return model, output
