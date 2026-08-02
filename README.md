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

- JSON result with overall score, component scores, irregularity probabilities,
  and confidence;
- CSV row for checkpoint comparison;
- annotated video output, using MP4 when OpenCV is installed and `.npz` plus GIF
  fallback otherwise;
- score-over-time plot;
- optical-flow visualization;
- temporal irregularity heatmap.

## Architecture

```text
RGB video
-> preprocessing and resizing
-> dense optical flow
-> body-centered residual flow
-> SVD/PCA learned motion basis
-> temporal and regional flow features
-> classical score/irregularity predictors
-> JSON, CSV, plots, annotated video
```

The SVD/PCA motion-basis step is the main Szeliski-aligned baseline. See
`docs/theory.md` and `docs/book_mapping.md`.

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

The current workspace used during implementation did not have OpenCV,
scikit-learn, or pytest installed. The smoke workflow therefore uses `.npz`
synthetic clips and NumPy-only models.

## Reproduce Smoke Demo

```bash
bash scripts/reproduce_all.sh
```

Equivalent direct command:

```bash
PYTHONPATH=src python -m gait_aqa.cli reproduce-smoke
```

This generates synthetic clips, creates a leakage-safe grouped split, trains the
classical model, scores one clip, and writes artifacts under `outputs/`.

## Dataset Generation

Synthetic sanity-check data:

```bash
PYTHONPATH=src python -m gait_aqa.cli generate-synthetic \
  --output-dir data/raw/synthetic \
  --manifest data/manifests/synthetic.csv \
  --clip-count 24

PYTHONPATH=src python -m gait_aqa.cli split-dataset \
  --manifest data/manifests/synthetic.csv \
  --output data/manifests/synthetic_split.csv
```

Import existing walker metrics:

```bash
PYTHONPATH=src python -m gait_aqa.cli import-walker \
  --walker-repo /path/to/mujoco-bipedal-joystick-walker \
  --output data/processed
```

The imported walker table is label data only until RGB rollout clips are
rendered and linked in the manifest.

## Training

```bash
PYTHONPATH=src python -m gait_aqa.cli train-classical \
  --manifest data/manifests/synthetic_split.csv \
  --model outputs/models/classical.pkl
```

## Evaluation

```bash
PYTHONPATH=src python -m gait_aqa.cli evaluate \
  --predictions outputs/predictions/classical_predictions.csv \
  --split test
```

Metrics include MAE, RMSE, R2, Spearman rank correlation, and pairwise ranking
accuracy.

## Single-Video Inference

```bash
PYTHONPATH=src python -m gait_aqa.cli score-video \
  --video data/raw/synthetic/synthetic_000.npz \
  --model outputs/models/classical.pkl \
  --annotated-output outputs/videos/example_scored.mp4
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Provenance and Licensing

See:

- `docs/reference_repository_audit.md`
- `docs/provenance.csv`
- `THIRD_PARTY_NOTICES.md`

No source code was copied from the unlicensed PECoP checkout or the unlicensed
MuJoCo walker checkout.

## Limitations

The score is a simulation-derived gait quality score, not a medically or
biomechanically validated measure. RGB video may hide contact forces and
controller instability. Synthetic-to-real transfer is not demonstrated. Full
Protocol B/C results require real rendered MuJoCo rollout videos.
