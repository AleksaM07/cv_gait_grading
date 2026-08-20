# Seminar submission checklist

## Required before upload

- Open `reports/main.pdf` and visually inspect the title page, table of
  contents, result tables, plots, page breaks, and final literature page.
- Confirm the exact course name, student name/index, submission date, required
  filename, and whether the title page must include professor or mentor name.
- Confirm whether the faculty expects only the PDF or also a repository/archive.
- If code is submitted, include tracked source/config/tests/docs and exclude all
  local datasets, videos, caches, checkpoints, reference checkouts, and secrets.
- Create one final Git commit or immutable archive after all checks pass.

## Defensible central claim

The system receives external side-view RGB recordings from a previously unseen
MuJoCo walking policy and estimates its simulator-derived gait quality without
using simulator state during inference. Six standardized scenarios are scored
and aggregated into a policy-level 0--100 research score with scenario spread
and OOD checks.

## Claims not to make

- Do not call the score a clinical, disability, rehabilitation, or validated
  human gait score.
- Do not describe six scenarios from one policy as six independent policies.
- Do not report deployment-head training predictions as test performance.
- Do not claim calibrated uncertainty; scenario standard deviation is only
  variation across commands.

## Optional improvements after submission

- Add more independently trained policy/checkpoint groups.
- Pre-register a final untouched policy test cohort.
- Calibrate prediction intervals after substantially more groups are available.
- Collect expert-labeled human side-view data only for a separate human-gait
  research question.
