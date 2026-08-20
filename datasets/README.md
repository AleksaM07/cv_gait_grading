# Dataset catalog and provenance

This directory is the local root for third-party datasets and their versioned
provenance documents. Dataset payloads remain excluded from Git. This mirrors
OpenGait's separation between dataset descriptions/configuration and runtime
outputs without redistributing third-party data.

Audit date: 2026-08-20.

## Local layout

```text
MUJOCO_videos_better/                    terminal-safe two-view policy rollouts
CMU_reference_videos/walking_flat/      filtered CMU walking BVH files
CMU_reference_videos/models/            local MuJoCo rendering model
CMU_reference_videos/mujoco_render/     derived side-view reference MP4 files
datasets/disabled_gait/videos/           125 unique real walking MP4 files
datasets/gahu/videos/                    395 trimmed side-view walking MP4 files
data/manifests/                          generated training manifests
data/processed/                          imported source metrics and labels
data/interim/flow/                       regenerable optical-flow cache
data/interim/embeddings/                 regenerable R3D-18 embedding cache
output/models/                           trained models
output/predictions/                      predictions, metrics, and figures
output/videos/                           scored videos
output/logs/                             runtime logs
```

Only this catalog, configuration, code, tests, and empty `.gitkeep` markers are
versioned. Local absolute paths in source manifests are normalized by
`prepare-real-manifest`; they are not portable provenance identifiers.

## Active data sources

### DisabledGait walking videos

- Local root: `datasets/disabled_gait/`.
- Source: https://data.mendeley.com/datasets/v6hy35ydch/2
- DOI/license: `10.17632/v6hy35ydch.2`, CC BY 4.0.
- Active selection: 125 byte-unique walking MP4s across `assistive`,
  `non_assistive`, and `normal`; five duplicate copies removed.
- Transformation: the H.264 stream was losslessly remuxed into a clean silent
  MP4; no video re-encoding was performed.
- Excluded payload: 6,500 JPG frames and 6,500 YOLO TXT annotations because
  this pipeline consumes video rather than detection frames.
- Details: `datasets/disabled_gait/SOURCE.md` and its local `manifest.csv`.

### GaHu-Video walking clips

- Local root: `datasets/gahu/`.
- Source: https://data.mendeley.com/datasets/gprg4s73v4/1
- DOI/license: `10.17632/gprg4s73v4.1`, CC BY 4.0.
- Active selection: 395 byte-unique, trimmed side-view walking clips from 44
  subjects and three tracks with left/center/right variants.
- Transformation: H.264 was losslessly remuxed from AVI to silent MP4. No
  video re-encoding was performed.
- Excluded payload: 44 untrimmed originals containing empty/non-walking
  intervals, one duplicate clip, and three precomputed `.dat` feature sets.
- Details: `datasets/gahu/SOURCE.md` and its local `manifest.csv`.

### CMU walking motion and derived MuJoCo reference videos

- Local BVH root: `CMU_reference_videos/walking_flat/`.
- Original database: Carnegie Mellon University Graphics Lab Motion Capture
  Database, http://mocap.cs.cmu.edu/.
- BVH conversion source recorded locally: CGSpeed's 3ds Max-friendly BVH
  release of the CMU database,
  https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-3dsmax-friendly-bvh-release-of-cmus-motion-capture-database.
- Selection metadata: `CMU_reference_videos/walking_manifest.csv` records the
  original relative BVH path, title, local clip ID, and curriculum tier.
- Current local selection: 580 walking BVH clips.
- Derived videos: 580 side-view H.264 renders under
  `CMU_reference_videos/mujoco_render/`, produced by
  `render-cmu-mujoco` using the local MuJoCo XML model.
- Renderer model: `CMU_reference_videos/models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml`.
  Its independent origin/license has not been established, so the XML and its
  derived media remain local.
- License/redistribution: consult the CMU database and conversion-distribution
  terms before sharing either BVH payloads or derived renders. This repository
  contains neither payload.

### Improved successful-policy rollout dataset

- Local root: `MUJOCO_videos_better/`.
- Checkpoint cohort: one best logged checkpoint from each run under
  `_references/mujoco-bipedal-joystick-walker/runs/successful`.
- Default design: 9 policies x 6 joystick scenarios x 2 cameras = 108 videos.
- Views: separate `side` and `front_oblique` MP4s generated from exactly the
  same simulator trajectory. Distance and vertical framing scale from each
  MuJoCo model's physical extent so differently sized robots fill the frame.
- Leakage control: every view and scenario from one policy/checkpoint shares a
  `split_group`; camera pairs also share a `rollout_id`.
- Fall handling: environment termination plus explicit root-height,
  torso-upright, and finite-state checks. Terminal metrics are retained, while
  the terminal pose is excluded from rendered frames.
- Raw labels: survival, reward, tracking RMSE, torso uprightness, action rate,
  commands, seed, checkpoint, and termination reason. Foot slip is included
  when the environment exposes compatible foot bodies; otherwise it is
  explicitly missing rather than fabricated.
- Composite label: scenario-wise min-max normalization followed by the walker
  definition `0.40 stability + 0.30 tracking + 0.20 upright + 0.10 smoothness`.
  Scores are computed once per rollout before being copied to its camera rows.
- Generator: `src/gait_aqa/reference_videos/render_walker_rollouts.py`.
- Runtime dependency: the local walker checkout owns environment and policy
  definitions; those files are not copied into this repository.

## Evaluated and not retained

### SSM synchronized scans and markers

- Contents: 4,528 PLY mesh frames for two subjects, synchronized marker PKLs,
  and MoSh++ NPZ results.
- Decision: excluded because its 17 motions contain no walking sequence. PLY
  files are 3D frames rather than ready-to-use videos.
- License: supplied terms restricted use to non-commercial work and prohibited
  redistribution.
- Local status: deleted on 2026-08-20 after audit; 22.517 GiB reclaimed.

## Derived artifacts

The following contain no new independent source data and can be regenerated:

- `data/manifests/better_videos_side_split.csv` and
  `data/manifests/better_videos_all_split.csv` from the improved walker render
  manifest;
- `data/manifests/side_representation_pretrain.csv`: 975 unlabeled side-view
  walking clips (580 CMU and 395 GaHu) used only to fit PCA/whitening;
- `data/manifests/side_aqa_supervised.csv`: 54 labeled MuJoCo side-view clips
  grouped by nine source policies for leakage-safe evaluation;
- `data/manifests/front_disabled_ood.csv`: 125 frontal DisabledGait clips used
  only for out-of-distribution distance checks, never as quality labels;
- `data/processed/walker_metric_labels.csv` from walker CSV exports;
- `data/interim/flow/classical/*.npz` from normalized RGB clips;
- `data/interim/embeddings/r3d18_kinetics400/*.npy` from time-normalized video
  windows;
- `CMU_reference_videos/mujoco_render/*` from the selected BVH files and XML;
- everything under `output/` from the manifest, configuration, and active code.

These paths are Git-ignored even when their individual file extensions are not
listed explicitly.

## Candidate not downloaded

- OU-ISIR Treadmill Dataset, candidate only; institutional approval is
  required: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitTM.html

## Structural reference

The metadata/config/output separation is inspired by OpenGait at inspected
commit `0efafd4779f127fbce34f22aff301bd82e923da5`. No OpenGait source code or
dataset payload was copied.

- Repository: https://github.com/ShiqiYu/OpenGait
- OpenGait keeps dataset-specific metadata under `datasets/`, experiment YAML
  under `configs/`, implementation under `opengait/`, and runtime artifacts
  under `output/`. This project keeps its standard installable implementation
  under `src/gait_aqa/` but adopts the same data/output boundary.

## Reproduction

```powershell
render-cmu-mujoco --workers 4

render-walker-rollouts

prepare-transfer-manifests

train-video-transfer

score-policy-transfer `
  --policy-dir MUJOCO_videos_better/P01_auto_xml_standard_no_ref

gait-aqa train-classical `
  --manifest data/manifests/better_videos_side_split.csv `
  --model output/models/classical_side.pkl `
  --predictions output/predictions/classical_side.csv
```

`train-video-transfer` uses torchvision's official R3D-18 Kinetics-400 V1
weights. The downloaded checkpoint stays under ignored `output/torch_cache/`
and is not redistributed. The self-contained trained model, predictions, and
logs remain under ignored `output/`.

Code-level reuse decisions and exact external repository inspection notes are
maintained separately in `reports/assets/docs/provenance.csv` and
`reports/assets/docs/reference_repository_audit.md`.
