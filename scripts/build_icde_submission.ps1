$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$tectonic = Join-Path $repoRoot 'tools\tectonic\tectonic.exe'
$env:TECTONIC_CACHE_DIR = Join-Path $repoRoot 'tools\tectonic\cache'

& $python scripts\make_icde_paper_assets.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $tectonic -X compile paper\icde2027\main.tex --keep-logs --keep-intermediates --outdir output\pdf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $tectonic -X compile paper\icde2027\supplement.tex --keep-logs --keep-intermediates --outdir output\pdf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Copy-Item -LiteralPath output\pdf\main.pdf -Destination output\pdf\quarm-icde2027-submission.pdf -Force
Copy-Item -LiteralPath output\pdf\supplement.pdf -Destination output\pdf\quarm-icde2027-supplement.pdf -Force

& $python scripts\qa_icde_pdf.py output\pdf\quarm-icde2027-submission.pdf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts\qa_icde_pdf.py output\pdf\quarm-icde2027-supplement.pdf
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python scripts\audit_icde_submission.py --run-tests
exit $LASTEXITCODE
