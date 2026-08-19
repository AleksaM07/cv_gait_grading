# Dataset catalog and provenance

This directory is metadata-only. Dataset payloads, videos, BVH files, derived
arrays, and model outputs are intentionally excluded from Git. This mirrors
OpenGait's separation between dataset descriptions/configuration and runtime
outputs without redistributing third-party data.

Audit date: 2026-08-19.

## Local layout

```text
MUJOCO_videos/gait_dataset/             external rendered walker rollouts
CMU_reference_videos/walking_flat/      filtered CMU walking BVH files
CMU_reference_videos/models/            local MuJoCo rendering model
CMU_reference_videos/mujoco_render/     derived side-view reference MP4 files
data/manifests/                          generated training manifests
data/processed/                          imported source metrics and labels
data/interim/flow/                       regenerable optical-flow cache
output/models/                           trained models
output/predictions/                      predictions, metrics, and figures
output/videos/                           scored videos
output/logs/                             runtime logs
```

Only this catalog, configuration, code, tests, and empty `.gitkeep` markers are
versioned. Local absolute paths in source manifests are normalized by
`prepare-real-manifest`; they are not portable provenance identifiers.

## Active data sources

### MuJoCo walker rollout videos

- Local root: `MUJOCO_videos/gait_dataset/`.
- Producer: `AleksaM07/mujoco-bipedal-joystick-walker`.
- Inspected source commit: `33eaa1fb76f2ffb0fb8a821deb9cad27f3989426`.
- Source: https://github.com/AleksaM07/mujoco-bipedal-joystick-walker
- Contents: rendered policy rollouts plus simulator-derived metrics such as
  composite score, tracking error, torso uprightness, foot slip, survival,
  scenario, seed, camera, and checkpoint identity.
- Active selection: 56 side-view clips from seven policy/checkpoint groups in
  `data/manifests/real_videos_side_split.csv`.
- Transformation: `gait-aqa prepare-real-manifest` resolves local paths,
  retains side views by default, constructs only labels supported by the
  source export, and performs a group-safe split.
- License/redistribution: the inspected upstream checkout had no license file.
  The videos and CSV payloads remain local and must not be redistributed until
  the owner provides applicable terms.

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

## Derived artifacts

The following contain no new independent source data and can be regenerated:

- `data/manifests/real_videos_side_split.csv` from the walker render manifest;
- `data/processed/walker_metric_labels.csv` from walker CSV exports;
- `data/interim/flow/classical/*.npz` from normalized RGB clips;
- `CMU_reference_videos/mujoco_render/*` from the selected BVH files and XML;
- everything under `output/` from the manifest, configuration, and active code.

These paths are Git-ignored even when their individual file extensions are not
listed explicitly.

## Referenced but not downloaded

- DissabledGait v2, DOI 10.17632/v6hy35ydch.2, CC BY 4.0:
  https://data.mendeley.com/datasets/v6hy35ydch/2
- GaHu-Video v1, DOI 10.17632/gprg4s73v4.1, CC BY 4.0:
  https://data.mendeley.com/datasets/gprg4s73v4/1
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
gait-aqa prepare-real-manifest `
  --input-manifest MUJOCO_videos/gait_dataset/manifest.csv `
  --dataset-root MUJOCO_videos/gait_dataset `
  --output data/manifests/real_videos_side_split.csv

render-cmu-mujoco --workers 4

gait-aqa train-classical `
  --manifest data/manifests/real_videos_side_split.csv `
  --model output/models/classical_side.pkl `
  --predictions output/predictions/classical_side.csv
```

Code-level reuse decisions and exact external repository inspection notes are
maintained separately in `reports/assets/docs/provenance.csv` and
`reports/assets/docs/reference_repository_audit.md`.
