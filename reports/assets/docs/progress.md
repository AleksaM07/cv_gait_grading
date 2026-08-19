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

## Phase 2-4

- CMU BVH retargeting, honest source labels, FPS-normalized dense motion,
  incremental SVD/PCA, stable Ridge least squares, grouped splitting,
  evaluation, H.264 output, and visualization are implemented.
- The side-view renderer validates and atomically writes H.264 reference clips.
- Pytest, Ruff, and mypy cover the active classical pipeline.

## Remaining Work

- Generate a real `uv.lock` once `uv` is installed.
- Add more independent policies before treating metrics as generalization.
- Export the upstream component scores if component-level supervision is needed.
- Evaluate a deep-video baseline after the classical dataset is complete.
