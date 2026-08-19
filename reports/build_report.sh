#!/usr/bin/env bash
set -euo pipefail

report_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$report_dir"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory . main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory . main.tex
rm -f main.aux main.log main.out main.toc
printf 'Generated %s\n' "$report_dir/main.pdf"
