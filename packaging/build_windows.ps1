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
    # dev as well as build: the script runs the test suite below, and pytest
    # lives in the dev extra. The lock file pins its version but a constraints
    # file never installs anything, so in a clean environment (which is what
    # the notes above recommend) the test step would fail with "No module
    # named pytest".
    python -m pip install -e ".[build,dev]" -c packaging\requirements-lock.txt
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

# The executable is windowed, so calling it with & returns at once and leaves it
# running: $LASTEXITCODE would then mean nothing, and the zip below would read
# files the smoke test still had open. Wait for it, and check what it produced.
$smoke = Join-Path $env:TEMP "amazeing_smoke.yaml"
if (Test-Path $smoke) { Remove-Item $smoke -Force }
$proc = Start-Process -FilePath $exe `
    -ArgumentList "--entry", "auditory", "--write-config", $smoke `
    -PassThru -Wait -WindowStyle Hidden
if ($proc.ExitCode -ne 0) {
    Write-Host "Smoke test exited with $($proc.ExitCode)." -ForegroundColor Red; exit 1
}
if (-not (Test-Path $smoke)) {
    Write-Host "Smoke test wrote no config; the executable is not working." -ForegroundColor Red
    exit 1
}

Write-Host "Copying the launchers into dist..." -ForegroundColor Cyan
Copy-Item (Join-Path $repo "packaging\dist_launchers\*.cmd") (Join-Path $repo "dist") -Force

# The download is named in the README and the installation page, so build it
# here rather than leaving the name to whoever makes the release.
# ZipFile rather than Compress-Archive: the folder is around 500 MB, where
# Compress-Archive is slow and reports its failures without failing.
$zip = Join-Path $repo "dist\amazeing-app.zip"
Write-Host "Zipping for release (this takes a minute)..." -ForegroundColor Cyan

# OneDrive and antivirus both take a handle on a file this size, and the build
# has just written half a gigabyte next to it, so the old zip is often still
# locked at this point. Wait a few seconds rather than throw away the build,
# and if it is still held, move it aside instead of failing.
if (Test-Path $zip) {
    $removed = $false
    foreach ($attempt in 1..5) {
        try {
            Remove-Item $zip -Force -ErrorAction Stop
            $removed = $true
            break
        } catch {
            Write-Host "  the old zip is in use, waiting ($attempt of 5)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 3
        }
    }
    if (-not $removed) {
        $aside = Join-Path $repo ("dist\amazeing-app.previous-build-{0}.zip" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
        try {
            Move-Item $zip $aside -Force -ErrorAction Stop
            Write-Host "  still in use; moved it to $(Split-Path -Leaf $aside)" -ForegroundColor Yellow
        } catch {
            Write-Host "Could not replace $zip : $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Close anything using it, then run again with -SkipInstall." -ForegroundColor Red
            exit 1
        }
    }
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
try {
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        (Join-Path $repo "dist\amazeing-app"), $zip,
        [System.IO.Compression.CompressionLevel]::Optimal, $true)
} catch {
    Write-Host "Could not write $zip : $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Close anything using dist\amazeing-app and run the script again." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $zip)) { Write-Host "Zip missing after build." -ForegroundColor Red; exit 1 }

Write-Host "`nBuilt: $exe" -ForegroundColor Green
Write-Host "Upload $zip to the release page; the README tells people to download that name."
Write-Host "To run it yourself, dist now holds both launchers:"
Write-Host "  Run the packaged app.cmd   the build above"
Write-Host "  Run from source.cmd        the live repository code"
