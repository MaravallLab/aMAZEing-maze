<#
.SYNOPSIS
  Build the standalone Windows application (dist\amazeing-app\amazeing-app.exe).

.DESCRIPTION
  Installs the exact, tested library versions from requirements-lock.txt into
  the current Python environment, then runs PyInstaller with the spec in this
  folder. Run from the repository root:

      .\packaging\build_windows.ps1

  Use a fresh virtual environment for a clean build:

      python -m venv .venv-build
      .\.venv-build\Scripts\Activate.ps1
      .\packaging\build_windows.ps1

  The result is a folder you can zip and hand to another lab; no Python
  installation is needed on the target machine.
#>
[CmdletBinding()]
param(
    [switch]$SkipInstall   # reuse the already-installed environment
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

if (-not $SkipInstall) {
    Write-Host "Installing pinned dependencies..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    python -m pip install -e ".[build]" -c packaging\requirements-lock.txt
}

Write-Host "Running the test suite..." -ForegroundColor Cyan
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed; not building." -ForegroundColor Red; exit 1 }
Remove-Item Env:QT_QPA_PLATFORM

Write-Host "Building with PyInstaller..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean packaging\amazeing-app.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exe = Join-Path $repo "dist\amazeing-app\amazeing-app.exe"
Write-Host "Smoke test: writing a config template with the built executable..." -ForegroundColor Cyan
& $exe --entry auditory --write-config (Join-Path $env:TEMP "amazeing_smoke.yaml")
if ($LASTEXITCODE -ne 0) { Write-Host "Smoke test failed." -ForegroundColor Red; exit 1 }

Write-Host "`nBuilt: $exe" -ForegroundColor Green
Write-Host "Zip the dist\amazeing-app folder to distribute it."
