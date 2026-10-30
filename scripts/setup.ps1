#requires -Version 5.1
<#
.SYNOPSIS
    Installs Node and Python dependencies for WakeForge Studio.

.DESCRIPTION
    Creates a project-local .venv with Python 3.10-3.12, installs the ML
    requirements, and (unless -SkipNode is passed) runs npm install.

.PARAMETER PythonExe
    Python interpreter to use. Accepts a path or a command on PATH.
    Defaults to searching py -3.12, py -3.11, py -3.10, python and python3.

.PARAMETER Recreate
    Delete and rebuild .venv before installing. Use this if the environment
    was created with an unsupported Python or is broken.

.PARAMETER SkipNode
    Do not run npm install.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Recreate -PythonExe C:\Python312\python.exe
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$Recreate,
    [switch]$SkipNode
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Requirements = Join-Path $Root "python\requirements.txt"
$CheckEnv = Join-Path $Root "python\check_env.py"

Set-Location $Root

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "OK  $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "!!  $Message" -ForegroundColor Yellow
}

function Test-SupportedPython {
    param(
        [string]$Exe,
        [string[]]$Prefix
    )
    $callArgs = @($Prefix) + @(
        "-c",
        "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"
    )
    try {
        & $Exe @callArgs 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

Write-Step "WakeForge Studio setup"
Write-Host "    Project: $Root"

if (-not $SkipNode) {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw "Node.js 20+ was not found. Install it from https://nodejs.org and re-run this script."
    }
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm was not found. Reinstall Node.js with npm included."
    }
    $nodeVersion = (& node --version).Trim()
    Write-Step "Installing frontend dependencies (Node $nodeVersion)"
    & npm install
    if ($LASTEXITCODE -ne 0) {
        throw "npm install failed."
    }
    Write-Ok "Node dependencies installed."
}

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Write-Warn "Removing the existing .venv"
    Remove-Item -LiteralPath $Venv -Recurse -Force
}

$chosen = $null
if (Test-Path -LiteralPath $VenvPython) {
    Write-Step "Reusing .venv"
} else {
    if ($PythonExe) {
        $candidates = @(@{ Exe = $PythonExe; Args = @() })
    } else {
        $candidates = @(
            @{ Exe = "py"; Args = @("-3.12") },
            @{ Exe = "py"; Args = @("-3.11") },
            @{ Exe = "py"; Args = @("-3.10") },
            @{ Exe = "python"; Args = @() },
            @{ Exe = "python3"; Args = @() }
        )
    }

    foreach ($candidate in $candidates) {
        if (Test-SupportedPython -Exe $candidate.Exe -Prefix $candidate.Args) {
            $chosen = $candidate
            break
        }
    }
    if (-not $chosen) {
        throw "Python 3.10-3.12 was not found. Install it from https://www.python.org/downloads/ or pass -PythonExe <path>."
    }

    Write-Step "Creating .venv (TensorFlow requires Python 3.10-3.12)"
    $createArgs = @($chosen.Args) + @("-m", "venv", $Venv)
    & $chosen.Exe @createArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        throw "Could not create .venv with $($chosen.Exe)."
    }
    Write-Ok "Virtual environment created."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw ".venv is incomplete. Re-run this script with -Recreate."
}

$pythonVersion = (& $VenvPython -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
Write-Step "Installing Python ML dependencies (Python $pythonVersion)"
& $VenvPython -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) {
    throw "Could not update pip."
}
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Could not install python/requirements.txt."
}

$probe = (& $VenvPython $CheckEnv | Select-Object -Last 1) | ConvertFrom-Json
if (-not $probe.supported_python) {
    throw "The .venv Python $($probe.version) is not supported. Re-run with -Recreate using Python 3.10-3.12."
}
$missing = @($probe.packages.PSObject.Properties | Where-Object { -not $_.Value } | ForEach-Object { $_.Name })
if ($missing.Count -gt 0) {
    throw "Missing Python packages: $($missing -join ', ')."
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Warn "Rust/cargo was not found. Install Rust from https://rustup.rs to run the desktop app."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Warn "npm was not found. Install Node.js before running the app."
}

Write-Host ""
Write-Ok "Setup complete."
Write-Host "    Python : $VenvPython"
Write-Host "    Start  : npm run tauri dev"
Write-Host ""
Write-Host "If the app uses a different interpreter, set WAKE_WORD_PYTHON to its path."
