#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

bash scripts/generate_dataset.sh
bash scripts/train_classical.sh
python -m gait_aqa.cli evaluate \
  --predictions outputs/predictions/classical_predictions.csv \
  --split test
