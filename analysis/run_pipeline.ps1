<#
.SYNOPSIS
  Full aMAZEing-maze pose pipeline: sleap-nn inference (+ per-frame CSV) then
  in-maze filtering, in one command.

.DESCRIPTION
  Stage 2  (sleap_batch_processing.py): predict 10 keypoints/frame for every
           *_cropped.mp4, writing <name>.predictions.slp + <name>.keypoints.csv.
           This is the long part (~4-5 days for a few hundred ~100k-frame
           videos). It is RESUMABLE: already-predicted videos are skipped.
  Stage 4  (filter_in_maze.py): drop out-of-maze / mouse-absent hallucinations,
           writing <name>.predictions.filtered.slp + <name>.keypoints.filtered.csv.
           Fast (minutes); safe to re-run.

  Re-run this script anytime to resume — inference skips done videos and the
  filter simply recomputes. Stage 1 (crop & align) is assumed already done; see
  analysis/README.md.

.EXAMPLE
  # Use the defaults below (this dataset):
  .\run_pipeline.ps1

.EXAMPLE
  # Only re-run the filter (inference already finished):
  .\run_pipeline.ps1 -SkipInference

.EXAMPLE
  # Different dataset / models:
  .\run_pipeline.ps1 -InputDir E:\study2_out -ModelsDir E:\models -CalibrationDir E:\study2_out
#>
[CmdletBinding()]
param(
    [string]$InputDir       = "D:\simplermaze_output",
    [string]$ModelsDir      = "C:\Users\shahd\OneDrive\Desktop\CROPPED_VIDEOS_FOR_SLEAP\models",
    [string]$CalibrationDir = "D:\simplermaze_output",
    [string]$Device         = "cuda",
    [string]$Python         = "$env:APPDATA\uv\tools\sleap-nn\Scripts\python.exe",
    [switch]$SkipInference,   # run only the filter stage
    [switch]$SkipFilter       # run only the inference stage
)

$repo  = Split-Path -Parent $MyInvocation.MyCommand.Path
$infer = Join-Path $repo "analysis\sleap_batch_processing.py"
$filt  = Join-Path $repo "analysis\filter_in_maze.py"

function Stamp { (Get-Date).ToString("yyyy-MM-dd HH:mm:ss") }
function Section($msg) { Write-Host "`n========== $(Stamp)  $msg ==========" -ForegroundColor Cyan }

# --- preflight checks -----------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: sleap-nn Python not found at:`n  $Python" -ForegroundColor Red
    Write-Host "Install it with:  uv tool install --python 3.13 `"sleap-nn[torch]`" --torch-backend auto"
    exit 1
}
foreach ($p in @($infer, $filt)) {
    if (-not (Test-Path $p)) { Write-Host "ERROR: missing script $p" -ForegroundColor Red; exit 1 }
}
foreach ($d in @($InputDir, $ModelsDir, $CalibrationDir)) {
    if (-not (Test-Path $d)) { Write-Host "ERROR: path does not exist: $d" -ForegroundColor Red; exit 1 }
}

Write-Host "sleap-nn python : $Python"
Write-Host "input_dir       : $InputDir"
Write-Host "models_dir      : $ModelsDir"
Write-Host "calibration_dir : $CalibrationDir"
Write-Host "device          : $Device"

# --- Stage 2: inference (+ CSV) -------------------------------------------
if (-not $SkipInference) {
    Section "Stage 2/4  inference (+ keypoints CSV)  -- this is the multi-day step"
    & $Python $infer --input_dir $InputDir --models_dir $ModelsDir --device $Device
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nInference exited with code $LASTEXITCODE -- stopping before the filter." -ForegroundColor Red
        Write-Host "Re-run this script to resume (finished videos are skipped)."
        exit $LASTEXITCODE
    }
} else {
    Section "Stage 2/4  inference  -- SKIPPED (-SkipInference)"
}

# --- Stage 4: in-maze filter ----------------------------------------------
if (-not $SkipFilter) {
    Section "Stage 4/4  in-maze / hallucination filter"
    & $Python $filt --input_dir $InputDir --calibration_dir $CalibrationDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nFilter exited with code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "Re-run with -SkipInference to retry just the filter."
        exit $LASTEXITCODE
    }
} else {
    Section "Stage 4/4  filter  -- SKIPPED (-SkipFilter)"
}

Section "Pipeline complete"
Write-Host "Per video you now have:"
Write-Host "  <name>.predictions.slp / .keypoints.csv            (raw)"
Write-Host "  <name>.predictions.filtered.slp / .keypoints.filtered.csv  (in-maze only)"
Write-Host "Run summary: $(Join-Path $InputDir 'sleap_inference_summary.csv')"
