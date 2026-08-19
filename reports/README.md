# LaTeX report

`main.tex` is the report source and `main.pdf` is its built form. The build
scripts run two LaTeX passes and remove temporary `.aux`, `.log`, `.out`, and
`.toc` files.

Runtime models, predictions, figures, logs, and scored videos are stored under
`../output/`, not in this report-source directory. Historical pre-audit outputs
remain locally under the Git-ignored `archive/` directory.

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\reports\build_report.ps1
```

Bash:

```bash
bash ./reports/build_report.sh
```
