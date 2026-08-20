"""Manifest construction helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gait_aqa.data.split_dataset import grouped_split

REQUIRED_RENDER_COLUMNS = {
    "video_path",
    "policy_id",
    "checkpoint_step",
    "scenario",
    "seed",
    "camera",
    "composite_score",
    "mean_first_fall_survival_fraction",
}


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest CSV."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def prepare_real_video_manifest(
    input_manifest: str | Path,
    output_manifest: str | Path = "data/manifests/better_videos_side_split.csv",
    dataset_root: str | Path | None = None,
    seed: int = 13,
    camera: str | None = "side",
) -> pd.DataFrame:
    """Convert a rendered walker video manifest into this project's schema.

    The renderer manifest may contain paths from another checkout. This function
    resolves each video by `policy_id/scenario/basename(video_path)` under the
    provided dataset root and derives score columns from the exported walker
    metrics.
    """
    source = load_manifest(input_manifest)
    missing_columns = REQUIRED_RENDER_COLUMNS - set(source.columns)
    if missing_columns:
        raise ValueError(
            f"Rendered manifest is missing required columns: {sorted(missing_columns)}"
        )
    if camera is not None:
        source = source[source["camera"].astype(str) == camera].copy()
        if source.empty:
            raise ValueError(f"No rows match camera={camera!r}")
    root = (
        Path(dataset_root) if dataset_root is not None else Path(input_manifest).parent
    )
    normalized = _source_scores(source)
    rows: list[dict[str, object]] = []
    missing_videos: list[Path] = []

    for index, row in source.iterrows():
        if str(row.get("status", "recorded")) != "recorded":
            continue
        video_path = _resolve_video_path(row, root)
        if not video_path.exists():
            missing_videos.append(video_path)
            continue
        scores = normalized.loc[index]
        clip_id = _clip_id(row, video_path)
        frame_count = int(row.get("frames", 0) or 0)
        duration = float(row.get("simulated_seconds", 0.0) or 0.0)
        fps = frame_count / duration if frame_count > 0 and duration > 0 else 30.0
        rows.append(
            {
                "clip_id": clip_id,
                "video_path": str(video_path),
                "policy_run": row["policy_id"],
                "checkpoint_step": int(row["checkpoint_step"]),
                "xml_model": "",
                "scenario": row["scenario"],
                "seed": int(row["seed"]),
                "camera": row["camera"],
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "split_group": f"{row['policy_id']}:{int(row['checkpoint_step'])}",
                "overall_score": scores["overall_score"],
                "stability_score": scores["stability_score"],
                "contact_score": float("nan"),
                # These properties cannot be recovered from the aggregate
                # renderer manifest. Missing labels are explicit so the model
                # cannot learn fabricated perfect scores.
                "symmetry_score": float("nan"),
                "periodicity_score": float("nan"),
                "smoothness_score": float("nan"),
                "tracking_score": float("nan"),
                "foot_sliding_label": float("nan"),
                "hopping_label": float("nan"),
                "micro_stepping_label": float("nan"),
                "asymmetry_label": float("nan"),
                "torso_instability_label": float("nan"),
                "toe_dragging_label": float("nan"),
                "fall_label": scores["fall_label"],
                "command_ignoring_label": float("nan"),
                "telemetry_path": str(input_manifest),
            }
        )

    if missing_videos:
        examples = ", ".join(str(path) for path in missing_videos[:3])
        raise FileNotFoundError(
            f"{len(missing_videos)} rendered videos are missing under {root}. "
            f"Examples: {examples}"
        )
    if not rows:
        raise ValueError(
            "No usable video rows were found. Check --dataset-root and manifest paths."
        )
    manifest = grouped_split(pd.DataFrame(rows), seed=seed)
    output = Path(output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return manifest


def _resolve_video_path(row: pd.Series, dataset_root: Path) -> Path:
    raw_path = Path(str(row["video_path"]))
    if raw_path.exists():
        return raw_path
    return dataset_root / str(row["policy_id"]) / str(row["scenario"]) / raw_path.name


def _clip_id(row: pd.Series, video_path: Path) -> str:
    stem = video_path.stem
    safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in stem)
    return (
        safe
        or f"{row['policy_id']}_{row['scenario']}_seed{row['seed']}_{row['camera']}"
    )


def _source_scores(source: pd.DataFrame) -> pd.DataFrame:
    """Return only scores directly supported by the renderer export.

    The renderer computes its composite once per rollout from stability,
    tracking, upright posture, and action smoothness across the policy cohort.
    Some environments cannot expose optional diagnostics such as foot slip, so
    those diagnostics must not block conversion when the exported composite and
    survival target are valid.
    """
    numeric_columns = [
        "composite_score",
        "mean_first_fall_survival_fraction",
    ]
    numeric = source[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing source score fields: {bad}")

    survival = numeric["mean_first_fall_survival_fraction"].clip(0.0, 1.0)
    if "resets" in source:
        resets = pd.to_numeric(source["resets"], errors="coerce").fillna(0.0)
    else:
        resets = pd.Series(0.0, index=source.index)
    if "ended_done" in source:
        ended_done = pd.to_numeric(source["ended_done"], errors="coerce").fillna(0.0)
    else:
        ended_done = pd.Series(0.0, index=source.index)
    return pd.DataFrame(
        {
            "overall_score": 100.0 * numeric["composite_score"].clip(0.0, 1.0),
            "stability_score": 100.0 * survival,
            "fall_label": (
                (resets > 0.0) | (ended_done > 0.0) | (survival < 1.0)
            ).astype(int),
        },
        index=source.index,
    )
