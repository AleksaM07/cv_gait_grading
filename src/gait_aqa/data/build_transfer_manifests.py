"""Build role-specific manifests for the frozen-video-backbone baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from gait_aqa.data.split_dataset import assert_no_group_overlap


def _portable_path(path: str | Path, workspace: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset video does not exist: {resolved}")
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return str(resolved)


def _cmu_subject_group(source_relative_path: str) -> str:
    path = PurePosixPath(source_relative_path.replace("\\", "/"))
    parent = path.parent.name
    if not parent:
        raise ValueError(f"Cannot infer CMU subject from: {source_relative_path}")
    return f"cmu:{parent}"


def _disabled_recording_group(source_id: str) -> str:
    if not source_id:
        raise ValueError("DisabledGait source ID is empty")
    return f"disabled_gait:series_{source_id[0].lower()}"


def build_transfer_manifests(
    workspace: str | Path = ".",
    output_dir: str | Path = "data/manifests",
) -> dict[str, Any]:
    """Create pretraining, supervised, and OOD manifests from local datasets."""
    root = Path(workspace).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    gahu = pd.read_csv(root / "datasets/gahu/manifest.csv")
    gahu_rows = pd.DataFrame(
        {
            "dataset": "gahu",
            "clip_id": gahu["clip_id"],
            "video_path": gahu["video_path"].map(
                lambda path: _portable_path(root / "datasets/gahu" / path, root)
            ),
            "split_group": "gahu:" + gahu["subject_id"].astype(str),
            "camera": "side",
            "variant": gahu["variant"],
            "role": "representation_pretrain",
            "split": "pretrain",
        }
    )

    cmu_metadata = pd.read_csv(root / "CMU_reference_videos/walking_manifest.csv")
    cmu_videos = pd.read_csv(
        root / "CMU_reference_videos/mujoco_render/training_manifest.csv"
    )
    cmu = cmu_videos.merge(
        cmu_metadata[["clip_id", "source_relative_path"]],
        on="clip_id",
        validate="one_to_one",
    )
    cmu = cmu[cmu["status"].astype(str) == "rendered"].copy()
    cmu_rows = pd.DataFrame(
        {
            "dataset": "cmu",
            "clip_id": "cmu__" + cmu["clip_id"].astype(str),
            "video_path": cmu["video_path"].map(
                lambda path: _portable_path(path, root)
            ),
            "split_group": cmu["source_relative_path"].map(_cmu_subject_group),
            "camera": "side",
            "variant": cmu["tier"],
            "role": "representation_pretrain",
            "split": "pretrain",
        }
    )
    representation = pd.concat([gahu_rows, cmu_rows], ignore_index=True)
    _validate_unique_videos(representation, "representation")
    representation_path = destination / "side_representation_pretrain.csv"
    representation.to_csv(representation_path, index=False)

    supervised = pd.read_csv(root / "data/manifests/better_videos_side_split.csv")
    if set(supervised["camera"].astype(str)) != {"side"}:
        raise ValueError("Supervised manifest must contain side-view clips only")
    supervised.insert(0, "dataset", "mujoco")
    supervised.insert(1, "role", "supervised_aqa")
    supervised["video_path"] = supervised["video_path"].map(
        lambda path: _portable_path(path, root)
    )
    _validate_unique_videos(supervised, "supervised")
    assert_no_group_overlap(supervised)
    supervised_path = destination / "side_aqa_supervised.csv"
    supervised.to_csv(supervised_path, index=False)

    disabled = pd.read_csv(root / "datasets/disabled_gait/manifest.csv")
    ood = pd.DataFrame(
        {
            "dataset": "disabled_gait",
            "clip_id": disabled["clip_id"],
            "video_path": disabled["video_path"].map(
                lambda path: _portable_path(
                    root / "datasets/disabled_gait" / path, root
                )
            ),
            "split_group": disabled["variant"].map(_disabled_recording_group),
            "camera": "front",
            "variant": disabled["category"],
            "role": "ood_only_no_quality_label",
            "split": "ood",
        }
    )
    _validate_unique_videos(ood, "OOD")
    ood_path = destination / "front_disabled_ood.csv"
    ood.to_csv(ood_path, index=False)

    summary = {
        "representation_clips": len(representation),
        "representation_datasets": representation["dataset"].value_counts().to_dict(),
        "supervised_clips": len(supervised),
        "supervised_splits": supervised["split"].value_counts().to_dict(),
        "supervised_policy_groups": int(supervised["split_group"].nunique()),
        "ood_clips": len(ood),
        "paths": {
            "representation": str(representation_path),
            "supervised": str(supervised_path),
            "ood": str(ood_path),
        },
    }
    (destination / "video_transfer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _validate_unique_videos(manifest: pd.DataFrame, name: str) -> None:
    if manifest.empty:
        raise ValueError(f"{name} manifest is empty")
    duplicate_ids = manifest["clip_id"].duplicated()
    duplicate_paths = manifest["video_path"].duplicated()
    if duplicate_ids.any() or duplicate_paths.any():
        raise ValueError(
            f"{name} manifest contains duplicate IDs or paths: "
            f"ids={int(duplicate_ids.sum())}, paths={int(duplicate_paths.sum())}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(build_transfer_manifests(args.workspace, args.output_dir), indent=2)
    )


if __name__ == "__main__":
    main()
