# Reference Repository Audit

Audit date: 2026-08-19.

The cloned repositories live under `_references/`, which is intentionally ignored
by Git. They are inspection inputs, not vendored dependencies.

| Repository | Commit | License files found | Classification | Reuse decision |
| --- | --- | --- | --- | --- |
| https://github.com/AleksaM07/mujoco-bipedal-joystick-walker | `33eaa1fb76f2ffb0fb8a821deb9cad27f3989426` | `README.md`, `pyproject.toml`; no license file found | unclear | Consume exported CSVs and command-line behavior only. Do not copy source. |
| https://github.com/AleksaM07/mujoco-bipedal-joystick-walker branch `project_v2_refrences_only` | `8b4d442` | `README.md`, `REFERENCES.md`; no license file found | unclear | Local worktree created for inspection and command-line use only. No source copied into `src/`; used as an external tooling/reference checkout for CMU/SMPL/BVH playback and render experiments. |
| https://github.com/Plrbear/PECoP | `af79e55c926457580989b72d27737c6e40f09e8f` | `README.md`; no license file found | unclear | Conceptual reference only for AQA structure and temporal sampling. No source copied. |
| https://github.com/TaatiTeam/CARE-PD | `9bc22b0258b15369cd8b62f2de20c70b3754349f` | `LICENSE` | permissive and reusable | MIT code may be reused with notice, but this project currently uses conceptual reimplementation only. |
| https://github.com/avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises | `838d3a46b04467610fa07f07827bccc8f2e6cec1` | `License - MIT.txt` | permissive and reusable | Conceptual influence for movement-quality scoring and PCA-distance ideas. No source copied. |
| https://github.com/avakanski/Rehabilitation-Assessment-through-Dimensionality-Reduction-and-Statistical-Modeling | `685b649ad1885eb70d78ddf608fb66c0febf3b21` | `License - MIT.txt` | permissive and reusable | Conceptual influence for dimensionality reduction and statistical movement scoring. No source copied. |

## Walker Inspection Summary

The MuJoCo walker already exports analysis tables under `analysis_outputs/`,
including `selected_checkpoints.csv`, `trial_metrics.csv`, `policy_metrics.csv`,
`episode_metrics.csv`, `training_history.csv`, and `actuator_metrics.csv`.

Important fields observed in exported CSVs include:

- checkpoint identity: `policy_id`, `run_name`, `policy_type`,
  `checkpoint_step`;
- scenarios and seeds: `scenario`, `seed`, `trial_id`;
- stability signals: `falls`, `first_fall_step`, `survival_fraction`,
  `mean_torso_up`, `min_torso_up`, `mean_root_height`, `min_root_height`;
- tracking signals: `tracking_rmse`, `command_failure_rate`;
- smoothness and effort: `mean_action_rate_norm`,
  `mean_mechanical_power_abs`;
- contact quality: `mean_foot_slip_speed`.

The original walker `evaluate.py` supports interactive viewer rollout and camera
control but did not expose a standalone RGB-video export CLI in the originally
inspected `main` version.

The separate local worktree on branch `project_v2_refrences_only` now adds:

- `CMU_SMPL+H-G/` sample AMASS/CMU SMPL+H `.npz` files;
- `filter_cmu_walking.py` for strict-walk filtering;
- `tools/render_raw_bvh_videos.py` for original-BVH skeleton previews;
- `tools/render_bvh_reference_videos.py` for MuJoCo-retargeted reference MP4s;
- richer `done` handling in `biomechanics_env.py`, including physical-fall,
  motion-over, and pose-termination signals.

That branch is therefore the preferred external tooling checkout for:

- generating CMU/AMASS-derived MuJoCo reference videos;
- testing whether a reference clip should terminate early;
- debugging why a retargeted clip falls, drifts, or penetrates the floor.

This repository now supplies the missing project-ready headless batch exporter
in `src/gait_aqa/reference_videos/render_walker_rollouts.py`. It discovers the
external successful checkpoints, renders six deterministic scenarios from side
and front-oblique cameras, excludes terminal fall poses, and writes resumable
manifests without copying the external environment or policy implementation.

## Additional Dataset And Software References

The following were added as conceptual references or candidate external datasets
for future experiments:

- OpenGait: https://github.com/ShiqiYu/OpenGait/tree/master
- AMASS: https://amass.is.tue.mpg.de/index.html
- CMU motion capture database / CGSpeed BVH conversion path
- CASIA / CBSR gait database hub: http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp
- SSM (Synchronized Scans and Markers) listed through AMASS

### Added gait-video dataset references

- **DissabledGait: Gait Dataset of Normal People and People with
  Disabilities**, version 2, DOI
  [10.17632/v6hy35ydch.2](https://doi.org/10.17632/v6hy35ydch.2). It contains
  130 walking videos and 6,500 labeled images across assistive,
  non-assistive, and normal gait classes. License: CC BY 4.0.
- **GaHu-Video: Parametrization System for Human Gait Recognition**, version
  1, DOI [10.17632/gprg4s73v4.1](https://doi.org/10.17632/gprg4s73v4.1). It
  provides recordings of 44 people, 396 edited walking videos, and derived
  geometric-feature files. License: CC BY 4.0.

### Maybe / candidate dataset

- **OU-ISIR Gait Database, Treadmill Dataset**:
  http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitTM.html. This is a useful
  side-view benchmark candidate with speed, clothing, view, and gait-
  fluctuation subsets, but access requires a signed institutional release
  agreement and case-by-case approval. It has not been downloaded or approved
  as an active project dataset.

## Licensing Notes

No PECoP source code was copied because the inspected checkout did not include a
license file. The implementation here is clean-room code written around the
seminar requirements and the public high-level ideas of Action Quality
Assessment.
