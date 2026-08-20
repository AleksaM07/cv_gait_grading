# Limitations

- The target score is constructed from simulator metrics.
- The model may learn simulator-specific visual cues.
- Background, camera, robot appearance, and rendering affect predictions.
- A visually natural gait is not necessarily dynamically optimal.
- A high score is not proof of human-like biomechanics.
- Simulator labels are not independent human judgements.
- Generalization from simulated renderings to real camera video is not demonstrated.
- Nested group evaluation supports limited generalization to held-out policies,
  but nine policy/checkpoint groups are still a small independent sample.
- RGB video may not reveal contact forces or internal control instability.
- The active supervised set has only nine independent policy/checkpoint groups;
  six scenarios from one policy do not create six independent policies.
- R3D-18 nested policy CV is promising (MAE 11.23, $R^2=0.638$, Spearman
  0.782), but uncertainty across unseen policy families remains substantial.
- The supported input is a standardized side view. Front and front-oblique
  predictions are diagnostic only and are marked invalid.
- A policy aggregate is valid only when all six expected scenario videos pass
  camera and embedding-domain checks.
- Unsupported component targets are intentionally returned as missing.

This project does not claim clinical validity, biological validation,
real-world robot deployment, general human gait assessment, or complete
objectivity.
