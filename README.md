# Visual Gait Quality Assessment

Standalone university seminar project for grading previously unseen MuJoCo
humanoid walking policies from external side-view RGB video. The primary
question is whether a policy's simulator-derived walking quality can be
estimated without simulator state, reward, contacts, or controller data at
inference time.

The recommended workflow records six standardized scenarios for a policy,
scores each video, and reports their mean score and spread on a 0--100 research
scale. Human walking datasets are auxiliary representation/OOD data; human or
clinical scoring is not the claim of this project.

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

For a standardized policy directory, the transfer scorer additionally writes
one JSON document with the six scenario scores, policy mean and spread,
completeness, camera support, and OOD validity.

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

`MUJOCO_videos_better/`, `CMU_reference_videos/`, local dataset payloads, and
`_references/` are never committed. Their exact origin and regeneration path
are documented in `datasets/README.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the video-transfer dependencies before training or inference:

```bash
python -m pip install -e ".[dev,deep]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For the transfer pipeline on Windows:

```powershell
python -m pip install -e ".[dev,deep]"
```

OpenCV-backed real video processing is the intended path. Deprecated synthetic
and placeholder deep-learning modules and unused wrappers have been removed.

## Dataset Preparation

Render the successful policy cohort and build the role-specific manifests:

```powershell
render-walker-rollouts
prepare-transfer-manifests
```

The renderer writes 54 unique rollouts from nine policies across six scenarios,
with separate side and front-oblique videos. The transfer builder uses only the
side videos for supervised scoring.

To rebuild the classical comparison manifest directly from the same renderer
output:

```powershell
gait-aqa prepare-real-manifest `
  --input-manifest MUJOCO_videos_better/manifest.csv `
  --dataset-root MUJOCO_videos_better `
  --output data/manifests/better_videos_side_split.csv
```

The preparation command keeps `side` views by default because both the regional
features and CMU references use that camera. Pass `--camera all` only for an
explicit multi-view experiment.

## Training

The current v1 model is a frozen R3D-18 pretrained on Kinetics-400, followed by
an unlabeled-video PCA/whitening transform and a small Ridge regression head.
It is the recommended starting point for the current small labeled dataset:

```powershell
prepare-transfer-manifests
train-video-transfer
```

The representation manifest contains 975 unlabeled walking videos (580 CMU
renders and 395 GaHu clips). The supervised manifest contains 54 MuJoCo
side-view rollouts from nine policies. Every scenario from a policy stays in
one group, so neither the holdout test nor nested policy cross-validation can
see the same policy during training. PCA is fitted only on the unlabeled CMU
and GaHu videos; labeled test embeddings never influence it.

The completed pilot produced:

| Evaluation | MAE | RMSE | R2 | Spearman |
|---|---:|---:|---:|---:|
| Nested policy averages (9 held-out policies) | 5.84 | 6.99 | 0.830 | 0.983 |
| Nested leave-one-policy mean baseline | 15.59 | 19.07 | -0.266 | n/a |
| Nested leave-one-policy-out CV (54 clips, 9 policies) | 11.23 | 13.64 | 0.638 | 0.782 |
| Fixed held-out test (12 clips, 2 policies) | 12.35 | 15.49 | 0.660 | 0.877 |
| Train-mean baseline on the same test | 22.17 | 29.48 | -0.233 | n/a |
| Classical motion baseline on the same test | 35.84 | 43.64 | -1.701 | -0.252 |

These are simulator-proxy results from only nine policies, not a clinical
validation. See `reports/assets/docs/r3d18_transfer_v1_model_card.md` for the
complete intended use and limitations.

The first run extracts and caches embeddings under
`data/interim/embeddings/r3d18_kinetics400`; a completed-cache rerun takes
seconds instead of re-decoding every video. Outputs are written to:

- `output/models/r3d18_transfer_side.pt`;
- `output/predictions/r3d18_transfer_side.csv`;
- `output/predictions/r3d18_transfer_side_report.json`;
- `output/predictions/disabled_ood_distances.csv`.

### Classical comparison baseline

```powershell
gait-aqa train-classical `
  --manifest data/manifests/better_videos_side_split.csv `
  --model output/models/classical_side.pkl `
  --predictions output/predictions/classical_side.csv
```

The first run caches FPS-normalized dense flow under
`data/interim/flow/classical`; subsequent compatible runs reuse it. This branch
is retained for comparison and motion diagnostics, not as the recommended
policy scorer.

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
  --video MUJOCO_videos_better/<policy_id>/<scenario>/<side-clip>.mp4 \
  --model output/models/classical_side.pkl \
  --annotated-output output/videos/example_scored.mp4
```

For the recommended transfer model:

```powershell
score-video-transfer `
  --video MUJOCO_videos_better/<side-video>.mp4 `
  --camera side `
  --output output/predictions/example_transfer_score.json
```

The output calls the result `predicted_research_score` and always includes
camera and distribution checks. A frontal clip can still produce a diagnostic
raw value, but `camera_supported=false`, `distribution_warning=true` when
applicable, and `clinical_score_valid=false` make it explicitly unusable as a
valid score. The DisabledGait frontal collection is never assigned fabricated
quality labels.

## Policy-Level Inference

The main use case scores all six standardized side-view scenarios while loading
the 127 MiB backbone only once:

```powershell
score-policy-transfer `
  --policy-dir MUJOCO_videos_better/P01_auto_xml_standard_no_ref `
  --output output/predictions/P01_policy_score.json
```

The JSON contains the policy mean, scenario standard deviation, minimum,
maximum, and per-video results. `policy_score_valid` is true only when all six
expected videos pass the side-camera and embedding-domain checks. Numeric
quality bands are intentionally not invented; comparisons use the continuous
score and scenario breakdown.

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
  train-classical --manifest data/manifests/better_videos_side_split.csv
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

### Render successful policy rollouts

`render-walker-rollouts` selects one best available checkpoint from every run
under `_references/mujoco-bipedal-joystick-walker/runs/successful`, then records
six deterministic commands from separate side and front-oblique cameras:

```powershell
render-walker-rollouts
```

The default plan contains 54 unique policy/scenario/seed rollouts and 108 MP4
files under `MUJOCO_videos_better/`. Both camera clips retain the same
`rollout_id` and `split_group`. Camera distance and vertical framing scale from
the MuJoCo model extent, keeping both the 1.8 m human and smaller Berkeley robot
fully visible. A physical terminal state contributes to the manifest metrics
but is not rendered, preventing a fallen collision model from passing through
the floor in the video. The command is resumable; inspect `render_plan.csv`,
`manifest.csv`, and `summary.json` in the output directory. Ready-to-train
side-only and multi-view splits are written to
`data/manifests/better_videos_{side,all}_split.csv`.

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
controller instability. The current evidence covers nine independent
policy/checkpoint groups and standardized side-view rendering only.
