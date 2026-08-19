# Training Targets

The rendered walker manifest contains a source `composite_score` in `[0, 1]`.
The training target is therefore:

```text
overall_score = 100 * composite_score
```

The upstream walker defines that composite as:

```text
composite = 0.40 * stability
          + 0.30 * tracking
          + 0.20 * upright
          + 0.10 * smoothness
```

Those source components are scenario-relative min-max scores. The compact
renderer export contains only a subset of the raw inputs and was created from a
larger policy population than the 56 side-view clips copied into this project.
Recomputing min-max values on the smaller local population would silently
change the target definition. Smoothness also cannot be recovered because the
required `mean_action_rate_norm` field is absent.

Consequently, the video manifest trains only targets supported directly by the
export:

| Target | Definition | Status |
| --- | --- | --- |
| `overall_score` | `100 * composite_score` | available |
| `stability_score` | `100 * mean_first_fall_survival_fraction` | available |
| `fall_label` | reset, terminal event, or survival fraction below one | available |
| Contact, symmetry, periodicity, smoothness, tracking | source components not recoverable from this export | missing (`NaN`) |
| Other irregularities | no ground-truth event labels in this export | missing (`NaN`) |

The independent Ridge heads skip a target when every training label is missing.
Inference serializes such outputs as JSON `null`, rather than a plausible-looking
but unsupported number. `confidence` is also `null` until at least ten finite
validation clips are available for error calibration; the current validation
split has eight clips.

This is a simulation-derived target, not a clinical or biomechanically validated
gait score.
