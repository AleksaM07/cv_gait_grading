# Szeliski Book Mapping

Inspected source PDF: `C:\tmp\cv_gait_grading\SzeliskiBook_20100903_draft.pdf`.
The page mapping was checked with `pdftotext -layout`. For the main chapters,
printed page `p` maps to PDF page `p + 22`. Appendix title/contents pages were
checked separately.

| Project concept | Szeliski chapter/section | Printed book pages | PDF pages | Implementation file | Experiment |
| --------------- | ------------------------ | -----------------: | --------: | ------------------- | ---------- |
| Mild denoising, grayscale normalization, filtering, pyramids | Ch. 3 image processing; Sec. 3.2 linear filtering; Sec. 3.5 pyramids and wavelets | 99-203; 111-122; 170-184 | 121-225; 133-144; 192-206 | `src/gait_aqa/vision/preprocessing.py` | image degradation smoke |
| Periodicity and spectral features | Ch. 3; Sec. 3.4 Fourier transforms | 138-169 | 160-191 | `src/gait_aqa/vision/temporal_features.py` | rendered-walk sanity check |
| Feature tracking comparison point | Ch. 4 feature detection and matching; Sec. 4.1 points and patches | 205-259; 207-237 | 227-281; 229-259 | `src/gait_aqa/vision/optical_flow.py` | optical-flow ablation |
| Least-squares Ridge regression and robust scaling | Ch. 6 feature-based alignment; Sec. 6.1.1 least squares; Appendix B.2 maximum likelihood and least squares | 309-336; 312-315; 759-766 | 331-358; 334-337; 781-788 | `src/gait_aqa/models/classical_regressor.py` | grouped score regression |
| Translational alignment / body-centered residual motion | Sec. 8.1 translational alignment | 384-397 | 406-419 | `src/gait_aqa/vision/alignment.py` | absolute vs residual flow |
| Coarse-to-fine motion estimation | Sec. 8.1.1 hierarchical motion estimation | 387-388 | 409-410 | `src/gait_aqa/vision/optical_flow.py` | image degradation smoke |
| Parametric motion fields | Sec. 8.2 parametric motion | 398-404 | 420-426 | `src/gait_aqa/vision/alignment.py` | absolute vs residual flow |
| Learned motion basis from stacked walking flow fields | Sec. 8.2.2 learned motion models | 403-404 | 425-426 | `src/gait_aqa/vision/motion_basis.py` | PCA component ablation |
| Dense optical flow | Sec. 8.4 optical flow | 409-414 | 431-436 | `src/gait_aqa/vision/optical_flow.py` | CMU reference and walker clips |
| Multi-frame motion estimates and temporal coefficients | Sec. 8.4.1 multi-frame motion estimation | 413-414 | 435-436 | `src/gait_aqa/vision/temporal_features.py` | score-over-time explanation |
| Recognition, PCA/subspace models, datasets, evaluation | Ch. 14 recognition; Sec. 14.2.1 eigenfaces; Sec. 14.6 recognition databases and test sets | 655-725; 671-678; 719-721 | 677-747; 693-700; 741-743 | `src/gait_aqa/evaluation/metrics.py` | checkpoint-ranking evaluation |
| SVD/PCA linear algebra | Appendix A; Sec. A.1.1 singular value decomposition; Sec. A.1.2 eigenvalue decomposition | 736-741 | 757-763 | `src/gait_aqa/vision/motion_basis.py` | PCA component ablation |
| Estimation and uncertainty | Appendix B; Sec. B.1 estimation theory | 757-758 | 779-780 | `src/gait_aqa/training/train_classical.py` | validation-error calibration gate |

## Citation Policy

Use both printed and PDF pages in report text, e.g. `[Szeliski, 2010 draft,
Sec. 8.2.2, printed pp. 403-404, PDF pp. 425-426]`. Do not quote long passages
from the book; paraphrase and cite.
