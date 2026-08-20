# Training Targets

The improved walker render manifest computes a `composite_score` in `[0, 1]`
for each unique rollout. The video-model target is:

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

The four components are computed from recorded raw rollout metrics and
min-max-normalized separately within each of the six scenarios. The score is
computed once per rollout before being copied to its two camera rows, avoiding
view-dependent labels. The active supervised manifest retains only the side
camera.

The video manifest exposes the following targets:

| Target | Definition | Status |
| --- | --- | --- |
| `overall_score` | `100 * composite_score` | available |
| `stability_score` | `100 * mean_first_fall_survival_fraction` | available |
| `fall_label` | reset, terminal event, or survival fraction below one | available |
| Smoothness, tracking | available in the render manifest but not trained as separate v1 heads | missing (`NaN`) |
| Contact, symmetry, periodicity | no compatible v1 ground truth | missing (`NaN`) |
| Other irregularities | no ground-truth event labels in this export | missing (`NaN`) |

The v1 R3D-18 transfer model predicts only `overall_score`. It does not invent
component values or categorical quality bands. A policy-level score is the
mean of six valid standardized scenario predictions; scenario spread is
reported separately and is not presented as calibrated statistical uncertainty.

This is a simulation-derived target, not a clinical or biomechanically validated
gait score.
