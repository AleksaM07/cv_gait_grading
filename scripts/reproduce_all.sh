#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

python -m gait_aqa.cli reproduce-smoke
