# Visual Gait Quality Assessment

Standalone university seminar project for estimating simulated humanoid gait
quality from external RGB video. The primary question is whether walking quality
can be estimated from video without simulator state at inference time.

The project is designed to support, but not depend on, the MuJoCo walker:

```text
https://github.com/AleksaM07/mujoco-bipedal-joystick-walker
```

It can import the walker's exported CSV metrics for label construction, while
the inference path accepts only RGB video-like input plus metadata such as FPS.

## Outputs

For each clip, the scorer writes:

- JSON result with overall score, supported component scores, irregularity
  probabilities, and confidence when validation data is sufficient;
- CSV row for checkpoint comparison;
- atomic H.264 annotated video output when FFmpeg is available;
- optical-flow visualization;
- temporal motion-energy heatmap.

Targets absent from the source export are written as `null`; unknown labels are
never replaced with fabricated perfect scores.

## Architecture

```text
RGB video
-> preprocessing and resizing
-> dense optical flow
-> body-centered residual flow
-> SVD/PCA learned motion basis
-> temporal flow and PCA-coefficient features
-> classical score/irregularity predictors
-> JSON, CSV, plots, annotated video
```

The SVD/PCA motion-basis step is the main Szeliski-aligned baseline. See
`reports/assets/docs/theory.md` and `reports/assets/docs/book_mapping.md`.

## Repository Layout

The layout follows the useful separation used by OpenGait while retaining the
existing Python `src/` package:

```text
configs/       versioned experiment configuration
datasets/      versioned dataset catalog and provenance; no payloads
data/          ignored local manifests, labels, caches, and processed data
output/        ignored models, predictions, figures, logs, and videos
reports/       versioned report source, documentation, and final PDF
src/           importable gait_aqa package
tests/         automated tests
```

`MUJOCO_videos/`, `CMU_reference_videos/`, and `_references/` are local external
payload roots and are never committed. Their exact origin and regeneration
path are documented in `datasets/README.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

OpenCV-backed real video processing is the intended path. Deprecated synthetic
and placeholder deep-learning modules and unused wrappers have been removed.

## Dataset Preparation

Import existing walker metrics when needed:

```bash
PYTHONPATH=src python -m gait_aqa.cli import-walker \
  --walker-repo /path/to/mujoco-bipedal-joystick-walker \
  --output data/processed
```

Prepare the real rendered video manifest:

```bash
PYTHONPATH=src python -m gait_aqa.cli prepare-real-manifest \
  --input-manifest MUJOCO_videos/gait_dataset/manifest.csv \
  --dataset-root MUJOCO_videos/gait_dataset \
  --output data/manifests/real_videos_side_split.csv
```

The preparation command keeps `side` views by default because both the regional
features and CMU references use that camera. Pass `--camera all` only for an
explicit multi-view experiment.

## Training

```bash
gait-aqa train-classical \
  --manifest data/manifests/real_videos_side_split.csv \
  --model output/models/classical_side.pkl \
  --predictions output/predictions/classical_side.csv
```

The first run caches FPS-normalized dense flow under
`data/interim/flow/classical`; subsequent compatible runs reuse it. PCA is fit
incrementally on train clips only, avoiding a multi-gigabyte in-memory matrix.

## Evaluation

```bash
gait-aqa evaluate \
  --predictions output/predictions/classical_side.csv \
  --split test
```

Metrics include MAE, RMSE, R2, Spearman rank correlation, and pairwise ranking
accuracy.

## Single-Video Inference

```bash
gait-aqa score-video \
  --video MUJOCO_videos/gait_dataset/<policy_id>/<scenario>/<clip>.mp4 \
  --model output/models/classical_side.pkl \
  --annotated-output output/videos/example_scored.mp4
```

## Tests

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src
```

## Logging

CLI commands write logs to `output/logs/gait_aqa.log` by default.
Custom log file and level go before the subcommand:

```bash
PYTHONPATH=src python -m gait_aqa.cli \
  --log-file output/logs/train.log \
  --log-level INFO \
  train-classical --manifest data/manifests/real_videos_side_split.csv
```

Follow logs live:

```bash
tail -f output/logs/gait_aqa.log
```

## Reference Videos

The CMU reference area contains:

- `CMU_reference_videos/walking_manifest.csv`
- `CMU_reference_videos/walking_flat/`
- `CMU_reference_videos/models/`
- `CMU_reference_videos/mujoco_render/`

### Render CMU BVH motion with MuJoCo

The renderer in `src/gait_aqa/reference_videos/render_mujoco.py` retargets CMU
BVH poses onto the named MuJoCo joints and streams side-view frames directly to
FFmpeg. Playback is kinematic, so controller or contact instability cannot
corrupt a reference motion. Completed MP4s are validated and written
atomically, and reruns skip them.

Install the optional rendering dependencies and make sure FFmpeg is available:

```powershell
python -m pip install -e ".[render]"
ffmpeg -version
```

Preview the first file before a batch:

```powershell
$env:PYTHONPATH = "src"
python -m gait_aqa.reference_videos.render_mujoco `
  --limit 1 --max-duration 5 --workers 1 --overwrite
```

Render every BVH discovered in `CMU_reference_videos/walking_flat`:

```powershell
$env:PYTHONPATH = "src"
python -m gait_aqa.reference_videos.render_mujoco --workers 4
```

Outputs are written to `CMU_reference_videos/mujoco_render` by default. The
current folder has 580 BVHs, not 300. Use `--limit 300` for exactly the first
300 files, or omit the limit for the full discovered set. Progress and failures
are recorded in `render_manifest.csv`; rerunning the same command safely
resumes the batch. `training_manifest.csv` joins each rendered clip to its CMU
title and quality tier. For a conservative gait-training subset, filter it to
`tier1` and `tier2` (268 clips); keep the 52 `uneven` clips separate because the
rendering model has a flat plane. Use `--help` for resolution, quality,
duration, and camera options.

## Provenance and Licensing

See:

- `datasets/README.md` for dataset origin, local layout, transformations, and
  redistribution status;
- `reports/assets/docs/reference_repository_audit.md`
- `reports/assets/docs/provenance.csv`
- `THIRD_PARTY_NOTICES.md`

No source code was copied from the unlicensed PECoP checkout or the unlicensed
MuJoCo walker checkout.

## Limitations

The score is a simulation-derived gait quality score, not a medically or
biomechanically validated measure. RGB video may hide contact forces and
controller instability. Full Protocol B/C results require real rendered MuJoCo
rollout videos.
