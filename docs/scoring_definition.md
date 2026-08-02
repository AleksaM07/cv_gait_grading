# Simulation-Derived Gait Quality Score

All component scores are in `[0, 100]`. The overall score is a weighted average:

```text
overall = 0.25 stability
        + 0.20 foot_contact_quality
        + 0.15 left_right_symmetry
        + 0.15 periodicity
        + 0.10 smoothness
        + 0.15 command_tracking
```

The weights are configured in `configs/scoring.yaml`. They emphasize survival,
contact quality, and command tracking because those were already central in the
MuJoCo walker analysis.

## Components

| Component | Inputs used by synthetic smoke data | Walker CSV fields used when available |
| --- | --- | --- |
| Stability | fall severity, jitter, torso instability | `survival_fraction`, `falls`, `mean_torso_up`, `min_root_height` |
| Foot contact quality | injected foot sliding and toe dragging | `mean_foot_slip_speed` |
| Left-right symmetry | injected left/right amplitude asymmetry | contact/step asymmetry fields if exported in future |
| Periodicity | gait phase regularity and hopping | contact periodicity fields if exported in future |
| Smoothness | jitter and high-frequency motion | `mean_action_rate_norm`, `mean_jerk_norm` |
| Command tracking | command-ignoring severity | `tracking_rmse`, `command_failure_rate` |

## Irregularities

Each irregularity is represented as a severity in `[0, 1]`, a binary label after
thresholding, and optional temporal intervals.

| Label | Definition | Known failure case |
| --- | --- | --- |
| `foot_sliding` | High tangential foot motion during stance or injected sliding severity. | RGB-only flow can confuse body translation with foot slip. |
| `hopping` | Excess vertical periodic energy or injected hopping severity. | Camera pitch can make normal steps look vertical. |
| `micro_stepping` | High step count with low displacement or injected short-step severity. | Static videos with jitter can mimic small steps. |
| `left_right_asymmetry` | Difference between left/right motion amplitude or injected asymmetry. | Side-view left/right assignment can swap under occlusion. |
| `torso_instability` | Torso tilt/jitter severity. | Crop jitter can look like torso wobble. |
| `toe_dragging` | Low foot clearance or injected toe-dragging severity. | Not always visible from an RGB side camera. |
| `fall_or_near_fall` | Fall indicator or severe loss of height/uprightness. | Short clips may stop before the fall completes. |
| `command_ignoring` | High tracking RMSE or injected command-ignoring severity. | Command is not visible at inference; only labels can use it. |

Thresholds are configured in `configs/scoring.yaml`.
