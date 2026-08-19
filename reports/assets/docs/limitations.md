# Limitations

- The target score is constructed from simulator metrics.
- The model may learn simulator-specific visual cues.
- Background, camera, robot appearance, and rendering affect predictions.
- A visually natural gait is not necessarily dynamically optimal.
- A high score is not proof of human-like biomechanics.
- Simulator labels are not independent human judgements.
- Generalization from simulated renderings to real camera video is not demonstrated.
- Performance on known checkpoints does not prove generalization to new
  policies.
- RGB video may not reveal contact forces or internal control instability.
- The active side-view set has only seven independent policy/checkpoint groups;
  two camera/seed variants do not create independent policies.
- Current held-out performance is weak (overall test MAE 4.16, RMSE 5.84,
  Spearman -0.39), so the model is an executable baseline, not training-ready
  evidence of generalization.
- A constant train-mean predictor is better on that test policy (MAE 3.51,
  RMSE 5.17). Ridge alpha selection improves validation MAE but does not remove
  the policy-domain shift.
- Unsupported component targets are intentionally returned as missing.
- Every clip in the current test policy has `fall_label=1`; its apparent F1 of
  1.0 is therefore a one-class sanity check, not evidence of fall discrimination.

This project does not claim clinical validity, biological validation,
real-world robot deployment, general human gait assessment, or complete
objectivity.
