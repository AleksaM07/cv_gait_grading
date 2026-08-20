"""Normalize downloaded walking datasets without changing video pixels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required to validate videos")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise ValueError(f"No video stream found: {path}")
    stream = payload["streams"][0]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "frame_count": int(stream.get("nb_frames") or 0),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def _prepare_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Destination is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _deduplicate(paths: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    retained: list[Path] = []
    excluded: list[dict[str, str]] = []
    owner_by_hash: dict[str, Path] = {}
    for path in sorted(paths, key=lambda item: item.as_posix().lower()):
        digest = _sha256(path)
        owner = owner_by_hash.get(digest)
        if owner is None:
            owner_by_hash[digest] = path
            retained.append(path)
            continue
        excluded.append(
            {
                "source_path": str(path),
                "duplicate_of": str(owner),
                "source_sha256": digest,
                "reason": "byte_identical",
            }
        )
    return retained, excluded


def _disabled_clip_id(path: Path) -> tuple[str, str]:
    category = path.parent.name.lower().replace(" ", "_").replace("-", "_")
    source_id = path.stem.lower()
    return f"disabled_gait__{category}__{source_id}", category


GAHU_NAME_PATTERN = re.compile(
    r"^S(?P<subject>\d{3})(?:_T(?P<track>[123]))?(?:_(?P<view>[LR]))?$",
    re.IGNORECASE,
)


def _gahu_clip_id(path: Path) -> tuple[str, str, str]:
    match = GAHU_NAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected GaHu filename: {path.name}")
    subject = f"s{match.group('subject')}"
    track = match.group("track")
    if track is None:
        return f"gahu__{subject}__original", subject, "original"
    raw_view = match.group("view")
    view = {None: "center", "L": "left", "R": "right"}[
        raw_view.upper() if raw_view else None
    ]
    variant = f"track{track}_{view}"
    return f"gahu__{subject}__{variant}", subject, variant


def _remux_h264_to_mp4(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to remux GaHu videos")
    partial = destination.with_name(f"{destination.stem}.partial.mp4")
    partial.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def organize_disabled_gait(source: Path, destination: Path) -> dict[str, int]:
    """Remux unique DisabledGait videos into a flat, labeled folder."""
    source = source.resolve()
    destination = destination.resolve()
    video_dir = destination / "videos"
    _prepare_destination(destination)
    video_dir.mkdir()
    retained, excluded = _deduplicate(list(source.rglob("*.mp4")))
    rows: list[dict[str, Any]] = []
    for source_path in retained:
        clip_id, category = _disabled_clip_id(source_path)
        output_path = video_dir / f"{clip_id}.mp4"
        if output_path.exists():
            raise FileExistsError(f"Clip ID collision: {output_path}")
        _remux_h264_to_mp4(source_path, output_path)
        probe = _probe_video(output_path)
        rows.append(
            {
                "dataset": "disabled_gait",
                "clip_id": clip_id,
                "video_path": output_path.relative_to(destination).as_posix(),
                "category": category,
                "subject_id": "",
                "variant": source_path.stem.lower(),
                "source_path": str(source_path.relative_to(source)),
                "source_sha256": _sha256(source_path),
                "video_sha256": _sha256(output_path),
                **probe,
            }
        )
    fields = list(rows[0])
    _write_csv(destination / "manifest.csv", rows, fields)
    _write_csv(
        destination / "excluded_duplicates.csv",
        excluded,
        ["source_path", "duplicate_of", "source_sha256", "reason"],
    )
    return {"videos": len(rows), "excluded_duplicates": len(excluded)}


def organize_gahu(source: Path, destination: Path) -> dict[str, int]:
    """Remux unique GaHu walking AVIs into flat, silent MP4 files."""
    source = source.resolve()
    destination = destination.resolve()
    video_dir = destination / "videos"
    _prepare_destination(destination)
    video_dir.mkdir()
    all_videos = list(source.rglob("*.avi"))
    untrimmed_originals = [
        path for path in all_videos if path.parent.name.lower() == "originals"
    ]
    walking_clips = [path for path in all_videos if path not in untrimmed_originals]
    retained, excluded = _deduplicate(walking_clips)
    excluded.extend(
        {
            "source_path": str(path),
            "duplicate_of": "",
            "source_sha256": _sha256(path),
            "reason": "untrimmed_contains_nonwalking_intervals",
        }
        for path in sorted(untrimmed_originals)
    )
    rows: list[dict[str, Any]] = []
    for source_path in retained:
        clip_id, subject_id, variant = _gahu_clip_id(source_path)
        output_path = video_dir / f"{clip_id}.mp4"
        if output_path.exists():
            raise FileExistsError(f"Clip ID collision: {output_path}")
        _remux_h264_to_mp4(source_path, output_path)
        probe = _probe_video(output_path)
        rows.append(
            {
                "dataset": "gahu",
                "clip_id": clip_id,
                "video_path": output_path.relative_to(destination).as_posix(),
                "category": "walking",
                "subject_id": subject_id,
                "variant": variant,
                "source_path": str(source_path.relative_to(source)),
                "source_sha256": _sha256(source_path),
                "video_sha256": _sha256(output_path),
                **probe,
            }
        )
    fields = list(rows[0])
    _write_csv(destination / "manifest.csv", rows, fields)
    _write_csv(
        destination / "excluded_duplicates.csv",
        excluded,
        ["source_path", "duplicate_of", "source_sha256", "reason"],
    )
    return {
        "videos": len(rows),
        "excluded_duplicates": sum(
            row["reason"] == "byte_identical" for row in excluded
        ),
        "excluded_untrimmed_originals": len(untrimmed_originals),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disabled-source", type=Path, required=True)
    parser.add_argument("--gahu-source", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = {
        "disabled_gait": organize_disabled_gait(
            args.disabled_source, args.datasets_root / "disabled_gait"
        ),
        "gahu": organize_gahu(args.gahu_source, args.datasets_root / "gahu"),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
