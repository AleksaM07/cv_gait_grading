# Progress

## Phase 1

- Workspace inspected: fresh repository.
- Szeliski PDF was not found locally; page-number entries are marked for
  verification.
- Reference repositories cloned into ignored `_references/`.
- Commit hashes and license decisions recorded.
- Walker CSV exports were inspected and imported successfully from the cloned
  reference checkout.

## Phase 2-4

- Synthetic dataset, score labels, dense motion features, SVD/PCA basis,
  classical regressor, grouped split, evaluation, and visualization are
  implemented for the smoke path.
- `python -m gait_aqa.cli reproduce-smoke` completed and wrote score JSON, CSV,
  plots, flow visualization, and annotated fallback video artifacts.
- `python -m unittest discover -s tests` passed 12 tests.

## Remaining Work

- Render real MuJoCo RGB clips from selected walker checkpoints.
- Replace `TODO: verify page` citations after inspecting the required PDF.
- Generate a real `uv.lock` once `uv` is installed.
- Run full Protocol B/C experiments on real rendered rollout clips.
- Add the optional deep-video baseline after the classical dataset is complete.
