#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

python -m gait_aqa.cli train-classical \
  --manifest data/manifests/synthetic_split.csv \
  --model outputs/models/classical.pkl
