# Experiments

Verified workflows:

- terminal-safe MuJoCo policy rendering across six standardized scenarios and
  two separate cameras;
- grouped policy/checkpoint splitting over 54 side-view clips from nine
  independent policies;
- frozen Kinetics-400 R3D-18 embedding extraction with resumable caching;
- PCA/whitening fitted only on 975 unlabeled CMU and GaHu walking clips;
- fixed policy-held-out and nested leave-one-policy-out evaluation;
- single-video and six-scenario policy-level scoring with OOD rejection;
- classical optical-flow/PCA baseline and visual diagnostics.

Current overall-score results:

| Evaluation | MAE | RMSE | R2 | Spearman | Ranking |
|---|---:|---:|---:|---:|---:|
| R3D-18 nested policy averages | 5.84 | 6.99 | 0.830 | 0.983 | 0.972 |
| Nested leave-one-policy mean | 15.59 | 19.07 | -0.266 | n/a | n/a |
| R3D-18 nested policy CV | 11.23 | 13.64 | 0.638 | 0.782 | 0.790 |
| R3D-18 fixed held-out test | 12.35 | 15.49 | 0.660 | 0.877 | 0.773 |
| Train-mean fixed test | 22.17 | 29.48 | -0.233 | n/a | n/a |
| Classical motion fixed test | 35.84 | 43.64 | -1.701 | -0.252 | 0.439 |

The primary supported claim is policy-level assessment of simulated MuJoCo
walking from external side-view RGB video. Human datasets are auxiliary
representation/OOD data and are not evidence of human gait scoring.
