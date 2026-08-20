# Progress

## Phase 1

- Workspace inspected: fresh repository.
- Szeliski PDF was later provided at
  `C:\tmp\cv_gait_grading\SzeliskiBook_20100903_draft.pdf`; relevant section
  page ranges were extracted with `pdftotext` and added to
  `reports/assets/docs/book_mapping.md` /
  `reports/assets/docs/theory.md`.
- Reference repositories cloned into ignored `_references/`.
- Commit hashes and license decisions recorded.
- Walker CSV exports were inspected and imported successfully from the cloned
  reference checkout.

## Phase 2-5

- CMU BVH retargeting, honest source labels, FPS-normalized dense motion,
  incremental SVD/PCA, stable Ridge least squares, grouped splitting,
  evaluation, H.264 output, and visualization are implemented.
- The side-view renderer validates and atomically writes H.264 reference clips.
- A frozen R3D-18 transfer baseline is trained and evaluated without policy
  leakage. It outperforms both train-mean and classical motion baselines.
- Single-video and standardized six-scenario policy inference are implemented.
- Pytest, Ruff, and mypy cover the active classical, rendering, transfer, and
  report-figure pipelines.

## Remaining Work

- Add more independent policies before treating metrics as generalization.
- Collect independent human side-view clips and expert labels only if the scope
  is deliberately expanded beyond simulated policy evaluation.
- Add calibrated uncertainty after substantially more policy groups exist.
- Fill in any faculty-required mentor/course metadata before submission.
