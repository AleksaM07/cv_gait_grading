$ErrorActionPreference = "Stop"

$reportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainingReportPath = Join-Path $reportDir '..\output\predictions\r3d18_transfer_side_report.json'
$resultFigurePath = Join-Path $reportDir 'assets\transfer_results_summary.png'
Push-Location $reportDir

try {
    $resultInputs = @($trainingReportPath)
    $missingResultInputs = @($resultInputs | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missingResultInputs.Count -eq 0) {
        python -m gait_aqa.visualization.transfer_report `
            --report $trainingReportPath `
            --output $resultFigurePath
        if ($LASTEXITCODE -ne 0) {
            throw "Result figure generation failed with exit code $LASTEXITCODE"
        }
    } elseif (-not (Test-Path -LiteralPath $resultFigurePath)) {
        throw "Result artifacts and the tracked fallback figure are missing"
    } else {
        Write-Warning "Runtime result files are absent; using the tracked audited figure."
    }
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory . main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "First pdflatex pass failed with exit code $LASTEXITCODE"
    }
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory . main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Second pdflatex pass failed with exit code $LASTEXITCODE"
    }
    Remove-Item -Force -ErrorAction SilentlyContinue main.aux, main.log, main.out, main.toc
    Write-Host "Generated $(Join-Path $reportDir 'main.pdf')"
} finally {
    Pop-Location
}
