#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

python -m gait_aqa.cli generate-synthetic \
  --output-dir data/raw/synthetic \
  --manifest data/manifests/synthetic.csv \
  --clip-count 24

python -m gait_aqa.cli split-dataset \
  --manifest data/manifests/synthetic.csv \
  --output data/manifests/synthetic_split.csv
