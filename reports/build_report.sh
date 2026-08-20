#!/usr/bin/env bash
set -euo pipefail

report_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$report_dir"

training_report_path="$report_dir/../output/predictions/r3d18_transfer_side_report.json"
result_figure_path="$report_dir/assets/transfer_results_summary.png"

if [[ -f "$training_report_path" ]]; then
  python -m gait_aqa.visualization.transfer_report \
    --report "$training_report_path" \
    --output "$result_figure_path"
elif [[ ! -f "$result_figure_path" ]]; then
  echo "Result artifacts and the tracked fallback figure are missing" >&2
  exit 1
else
  echo "Runtime result files are absent; using the tracked audited figure." >&2
fi

pdflatex -interaction=nonstopmode -halt-on-error -output-directory . main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory . main.tex
rm -f main.aux main.log main.out main.toc
printf 'Generated %s\n' "$report_dir/main.pdf"
