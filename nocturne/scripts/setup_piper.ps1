<#
    setup_piper.ps1 — download the Piper neural-TTS binary + a voice model into
    nocturne/voices/, so the `piper` backend has a warm, natural, offline voice.

    Piper binaries and voices are large (~85 MB) and machine-specific, so they're
    gitignored and fetched here instead of committed. Re-run this on a fresh
    checkout or to add another voice.

    Usage (from the nocturne/ directory):
        pwsh -File scripts/setup_piper.ps1                  # default voice: en_US-amy-medium
        pwsh -File scripts/setup_piper.ps1 -Voice en_US-ryan-medium
        pwsh -File scripts/setup_piper.ps1 -Voice en_GB-alba-medium

    Then set these in nocturne.toml (the script prints the exact lines):
        [tts]
        backend    = "piper"
        piper_exe   = "<...>/voices/piper/piper.exe"
        piper_model = "<...>/voices/models/<voice>.onnx"
#>
param(
    [string]$Voice = "en_US-amy-medium",
    [string]$PiperRelease = "2023.11.14-2"
)

$ErrorActionPreference = "Stop"

# Resolve nocturne/ root (this script lives in nocturne/scripts/).
$root      = Split-Path -Parent $PSScriptRoot
$voicesDir = Join-Path $root "voices"
$modelsDir = Join-Path $voicesDir "models"
$piperDir  = Join-Path $voicesDir "piper"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

# --- Piper binary --------------------------------------------------------- #
$exe = Join-Path $piperDir "piper.exe"
if (Test-Path $exe) {
    Write-Host "[setup_piper] piper.exe already present, skipping binary download."
} else {
    $zipUrl = "https://github.com/rhasspy/piper/releases/download/$PiperRelease/piper_windows_amd64.zip"
    $zip    = Join-Path $voicesDir "piper_windows_amd64.zip"
    Write-Host "[setup_piper] downloading Piper binary ($PiperRelease)…"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zip
    Write-Host "[setup_piper] extracting…"
    Expand-Archive -Path $zip -DestinationPath $voicesDir -Force
    Remove-Item $zip
}

# --- Voice model ---------------------------------------------------------- #
# Voice id is <lang>_<REGION>-<name>-<quality>, e.g. en_US-amy-medium.
$parts = $Voice -split "-"
if ($parts.Count -lt 3) { throw "voice must look like en_US-amy-medium; got '$Voice'" }
$quality = $parts[-1]
$name    = $parts[-2]
$locale  = $parts[0]                 # en_US
$lang    = ($locale -split "_")[0]   # en
$base    = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/$locale/$name/$quality/$Voice"

$onnx     = Join-Path $modelsDir "$Voice.onnx"
$onnxJson = Join-Path $modelsDir "$Voice.onnx.json"
if (Test-Path $onnx) {
    Write-Host "[setup_piper] voice '$Voice' already present, skipping model download."
} else {
    Write-Host "[setup_piper] downloading voice '$Voice' (~60 MB)…"
    Invoke-WebRequest -Uri "$base.onnx"      -OutFile $onnx
    Invoke-WebRequest -Uri "$base.onnx.json" -OutFile $onnxJson
}

# --- Report --------------------------------------------------------------- #
Write-Host ""
Write-Host "[setup_piper] done. Put these in nocturne.toml under [tts]:" -ForegroundColor Green
Write-Host ""
Write-Host "  backend     = `"piper`""
Write-Host "  piper_exe   = `"$($exe -replace '\\','/')`""
Write-Host "  piper_model = `"$($onnx -replace '\\','/')`""
