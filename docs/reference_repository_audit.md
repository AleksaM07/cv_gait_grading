# Reference Repository Audit

Audit date: 2026-08-02.

The cloned repositories live under `_references/`, which is intentionally ignored
by Git. They are inspection inputs, not vendored dependencies.

| Repository | Commit | License files found | Classification | Reuse decision |
| --- | --- | --- | --- | --- |
| https://github.com/AleksaM07/mujoco-bipedal-joystick-walker | `33eaa1fb76f2ffb0fb8a821deb9cad27f3989426` | `README.md`, `pyproject.toml`; no license file found | unclear | Consume exported CSVs and command-line behavior only. Do not copy source. |
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

The walker `evaluate.py` supports interactive viewer rollout and camera control
but does not expose a standalone RGB-video export CLI in the inspected version.
This repository therefore provides an independent synthetic-video smoke path and
an import path for existing walker CSV metrics. A future optional adapter can add
headless MuJoCo rendering without coupling the CV package to policy internals.

## Licensing Notes

No PECoP source code was copied because the inspected checkout did not include a
license file. The implementation here is clean-room code written around the
seminar requirements and the public high-level ideas of Action Quality
Assessment.
