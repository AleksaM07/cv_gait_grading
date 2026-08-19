$ErrorActionPreference = "Stop"

$reportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $reportDir

try {
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
