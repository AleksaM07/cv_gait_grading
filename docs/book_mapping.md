# Szeliski Book Mapping

The requested PDF, `SzeliskiBook_20100903_draft(1).pdf`, was not present in the
workspace or attachment directory during implementation. Section numbers below
come from the public 2010 draft structure named in the project brief. Printed
and PDF page numbers are therefore deliberately marked `TODO: verify page`
rather than invented.

| Project concept | Szeliski chapter/section | Printed book pages | PDF pages | Implementation file | Experiment |
| --------------- | ------------------------ | -----------------: | --------: | ------------------- | ---------- |
| Mild denoising, grayscale normalization, image pyramids | Ch. 3 image processing, filtering, pyramids | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/preprocessing.py` | image degradation smoke |
| Periodicity and spectral features | Ch. 3 Fourier transforms | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/temporal_features.py` | synthetic sanity check |
| Feature tracking comparison point | Ch. 4 feature detection and tracking | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/optical_flow.py` | optical-flow ablation |
| Least-squares Ridge regression and robust scaling | Ch. 6 alignment, least squares, robust estimation; App. B | TODO: verify page | TODO: verify page | `src/gait_aqa/models/classical_regressor.py` | grouped score regression |
| Translational alignment / body-centered residual motion | Sec. 8.1 translational alignment | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/alignment.py` | absolute vs residual flow |
| Coarse-to-fine motion estimation | Sec. 8.1.1 hierarchical estimation | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/optical_flow.py` | image degradation smoke |
| Parametric motion and learned motion fields | Sec. 8.2 and Sec. 8.2.2 learned motion models | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/motion_basis.py` | PCA component ablation |
| Dense optical flow | Sec. 8.4 optical flow | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/optical_flow.py` | clean synthetic and walker clips |
| Multi-frame motion estimates and temporal coefficients | Sec. 8.4.1 multi-frame motion estimation | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/temporal_features.py` | score-over-time explanation |
| Whole-body and kinematic tracking context | Ch. 12 tracking | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/body_regions.py` | region-flow explanations |
| Recognition, SVM-style baselines, datasets, evaluation | Ch. 14 recognition and datasets | TODO: verify page | TODO: verify page | `src/gait_aqa/evaluation/metrics.py` | checkpoint-ranking evaluation |
| SVD/PCA linear algebra | App. A SVD, PCA, least squares | TODO: verify page | TODO: verify page | `src/gait_aqa/vision/motion_basis.py` | PCA component ablation |

## Citation Policy

All detailed report citations must be updated after visually inspecting the PDF
and confirming printed page numbers. Until then, documentation uses section-only
citations or `TODO: verify page` notes.
