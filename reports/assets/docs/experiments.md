# Experiments

Verified workflows:

- CMU BVH retargeting sanity checks across walking styles and subject scales;
- grouped policy/checkpoint split over side-view clips;
- classical flow/PCA feature extraction;
- Ridge-style prediction for supported targets only;
- headless, side-view MuJoCo rendering with atomic MP4 output.

The current overall-score result is a weak baseline (test MAE 4.16, RMSE 5.84,
$R^2=-1.23$, Spearman -0.39). More independent policies and grouped
cross-validation are required before model selection or a deep baseline. The
train-mean constant baseline is currently stronger on test (MAE 3.51).
