# R3D-18 side-view transfer baseline v1

Audit date: 2026-08-20.

## Intended use

This model ranks and estimates the simulator-derived gait quality of a
side-view MuJoCo walking rollout. Its primary use is to aggregate six
standardized scenario videos into one policy score with a scenario breakdown.
It is a research baseline for policy comparison and pipeline development. It
is not a medical device, clinical gait score, or measure of disability.

## Model and data

- Frozen torchvision R3D-18 backbone with official Kinetics-400 V1 weights.
- Two deterministic, time-normalized 16-frame windows per video.
- A 64-component PCA/whitening transform fitted only on 580 CMU rendered and
  395 GaHu real side-view walking clips without quality labels.
- A Ridge head trained on 54 MuJoCo side-view rollouts from nine policies.
- Deployment Ridge alpha 10, selected by grouped cross-validation; the final
  deployment head is fitted on all 54 labeled clips.

All clips from a source policy share one group. Camera pairs and scenarios are
therefore prevented from leaking across evaluation folds.

## Pilot performance

| Evaluation | MAE | RMSE | R2 | Spearman | Pairwise ranking |
|---|---:|---:|---:|---:|---:|
| Nested policy averages (9 policies) | 5.84 | 6.99 | 0.830 | 0.983 | 0.972 |
| Nested leave-one-policy mean baseline | 15.59 | 19.07 | -0.266 | n/a | n/a |
| Nested leave-one-policy-out CV | 11.23 | 13.64 | 0.638 | 0.782 | 0.790 |
| Fixed two-policy holdout | 12.35 | 15.49 | 0.660 | 0.877 | 0.773 |
| Train-mean holdout baseline | 22.17 | 29.48 | -0.233 | n/a | n/a |
| Classical holdout baseline | 35.84 | 43.64 | -1.701 | -0.252 | 0.439 |

The fixed holdout Ridge alpha was selected using its separate validation
policies. Nested cross-validation is the more complete estimate because it
rotates every policy through the outer test fold.

## Domain checks

The training-domain distance threshold is the labeled training distribution's
95th percentile, 1.661. DisabledGait frontal videos have median distance 2.307
and 95th percentile 2.559, confirming a substantial view/domain shift. They
have no compatible quality labels and are not reported as scored evaluation
data.

Single-video inference reports `camera_supported`, `domain_distance`, and
`distribution_warning`. It always reports `clinical_score_valid=false`.
Policy inference is valid only when all six expected side-view scenarios pass
those checks; the reported scenario spread is not calibrated uncertainty.

## Limitations

- Quality targets come from a simulated controller and composite proxy, not
  expert human ratings or force-plate measurements.
- Nine source policies are too few for a high-confidence generalization claim.
- The supported camera is side view. Frontal or strongly oblique predictions
  are diagnostic raw outputs only and must not be treated as valid scores.
- The unlabeled CMU videos are retargeted, kinematic renders; GaHu adds real
  appearance variation but no quality supervision.
- The model may encode background, morphology, clothing, camera, or simulator
  cues. It has not been audited for demographic fairness.

## Required next validation

Before any human-facing use, collect independent side-view walking clips with
multiple expert raters, freeze this checkpoint and scoring protocol, and test
on subjects and acquisition sites that were absent from all model-development
data.
