# Set this project up on a machine that has never seen it.
#
#   Right-click -> Run with PowerShell,  or:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# What it does: installs uv if missing, builds the two Python environments on
# LOCAL DISK, and runs the tests.
#
# WHY LOCAL DISK. This repo normally lives inside a Google Drive folder. A venv
# is tens of thousands of small files, and writing them onto a cloud-synced
# virtual filesystem fails part-way with
#     os error 1450: Insufficient system resources exist to complete the service
# leaving a venv where `import numpy` succeeds and `numpy.array` does not exist.
# Measured on this project. The CODE can live on the Drive; the ENVIRONMENT
# cannot. A venv also bakes absolute paths into pyvenv.cfg and its scripts, so
# copying one between machines would not work even if it copied cleanly.

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$envRoot = Join-Path $env:USERPROFILE 'venvs'
$main = Join-Path $envRoot 'deadbug'
$bench = Join-Path $envRoot 'deadbug-a'

Write-Host ''
Write-Host '  Dead Bug AQA - setup' -ForegroundColor Cyan
Write-Host "  repo:         $repo"
Write-Host "  environments: $envRoot  (local disk, on purpose)"
Write-Host ''

# --- uv ---------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host '  installing uv...' -ForegroundColor Yellow
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host "  uv $(uv --version)" -ForegroundColor Green

# --- main environment: app, signals, tests ----------------------------
Write-Host ''
Write-Host '  [1/3] main environment (mediapipe, opencv, fastapi)...' -ForegroundColor Yellow
uv venv $main --clear --python 3.13
uv pip install --python $main -r (Join-Path $repo 'requirements.txt')
uv pip install --python $main yt-dlp        # optional: the app's YouTube tab

# --- benchmark environment: Track A only ------------------------------
# Separate by necessity. aeon pins numpy<2.5, pandas<2.4 and scipy<1.18, all
# below what MediaPipe and OpenCV need; one shared env downgrades a working
# stack. Pin aeon==1.5.0 EXPLICITLY -- a looser constraint lets the resolver
# settle on aeon==0.0.0, an empty placeholder on PyPI, and the install then
# SUCCEEDS with exit code 0 while `import aeon` fails.
Write-Host ''
Write-Host '  [2/3] benchmark environment (aeon, tensorflow)...' -ForegroundColor Yellow
$reqA = Join-Path $repo 'requirements-a.txt'
if (Test-Path $reqA) {
    try {
        uv venv $bench --clear --python 3.13
        uv pip install --python $bench -r $reqA
    } catch {
        Write-Host '  benchmark env failed - not fatal, only Track A needs it' -ForegroundColor DarkYellow
    }
}

# --- verify -----------------------------------------------------------
Write-Host ''
Write-Host '  [3/3] running the tests...' -ForegroundColor Yellow
$py = Join-Path $main 'Scripts\python.exe'
Push-Location $repo
& $py -m pytest tests/ -q
$code = $LASTEXITCODE
Pop-Location

Write-Host ''
if ($code -eq 0) {
    Write-Host '  All tests passed. Ready.' -ForegroundColor Green
    Write-Host ''
    Write-Host '  Run the app:' -ForegroundColor Cyan
    Write-Host "      & '$py' scripts\run_app.py"
    Write-Host '      then open http://localhost:8000'
    Write-Host ''
    Write-Host '  Run the CLI coach on a clip:' -ForegroundColor Cyan
    Write-Host "      & '$py' scripts\run_live.py --source `"data\clips\videoplayback (1).mp4`""
} else {
    Write-Host '  TESTS FAILED. Do not demo until this is green.' -ForegroundColor Red
    Write-Host '  Gate 0 (test_normalize + test_skeleton) blocks everything:' -ForegroundColor Red
    Write-Host '  if those two fail, normalization is broken and every number is meaningless.'
}
Write-Host ''
exit $code
