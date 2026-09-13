#Requires -Version 5.1
<#
.SYNOPSIS
    Installs ClauDali: a virtual environment, PyTorch matched to your GPU, the
    runtime dependencies, and SDXL model weights.

.DESCRIPTION
    A thin wrapper over `python -m installer install`, which does the real work
    and is identical on every platform. This script only locates a suitable
    Python interpreter and forwards the options.

    Everything is installed inside this project folder -- the environment in
    .venv\ and the weights in models\ -- so uninstall.ps1 can remove all of it
    with nothing left behind in your user profile.

.PARAMETER ModelProfile
    minimal   ~7.5 GB   SDXL base + the fp16-fix VAE. Everything works.
    standard  ~8.1 GB   Adds the depth and canny ControlNets. The default.
    full      ~22.3 GB  Adds the photoreal and painterly fine-tunes.

.PARAMETER RecreateVenv
    Delete and rebuild .venv\ instead of reusing an existing one.

.PARAMETER Cpu
    Install the CPU build of PyTorch even if an NVIDIA GPU is present.

.PARAMETER SkipTorch
    Leave PyTorch untouched. Useful when only adding models.

.PARAMETER SkipModels
    Set up the environment without downloading any weights.

.PARAMETER QualityModels
    Also download the optional quality models on top of the profile: the SDXL
    refiner (~6.25 GB) and Real-ESRGAN x4 (~0.07 GB). They are used by
    refiner.enabled, hires.upscaler "realesrgan-x4" and render.quality "max".

.EXAMPLE
    .\install.ps1
    Installs the standard profile.

.EXAMPLE
    .\install.ps1 -ModelProfile full
    Installs everything, including both fine-tunes.

.EXAMPLE
    .\install.ps1 -ModelProfile full -QualityModels
    Installs everything, plus the refiner and the Real-ESRGAN upscaler.
#>
[CmdletBinding()]
param(
    [ValidateSet('minimal', 'standard', 'full')]
    [string]$ModelProfile = 'standard',
    [switch]$RecreateVenv,
    [switch]$Cpu,
    [switch]$SkipTorch,
    [switch]$SkipModels,
    [switch]$QualityModels
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Find-Python {
    # `py` is the Windows launcher and is the most reliable way to get a real
    # CPython rather than the Microsoft Store alias stub, which exits silently.
    $candidates = @(
        @{ Exe = 'py';     Args = @('-3') },
        @{ Exe = 'python'; Args = @() },
        @{ Exe = 'python3'; Args = @() }
    )
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        try {
            $version = & $candidate.Exe @($candidate.Args + '--version') 2>&1
            if ($LASTEXITCODE -eq 0 -and "$version" -match 'Python 3\.(\d+)') {
                if ([int]$Matches[1] -ge 10) {
                    return $candidate
                }
                Write-Host "  skipping $($candidate.Exe): $version (need 3.10+)" -ForegroundColor DarkYellow
            }
        } catch { continue }
    }
    return $null
}

Write-Host ''
Write-Host '  ClauDali installer' -ForegroundColor Yellow
Write-Host "  $PSScriptRoot"
Write-Host ''

$python = Find-Python
if (-not $python) {
    Write-Host 'No suitable Python found.' -ForegroundColor Red
    Write-Host 'ClauDali needs Python 3.10 or newer: https://www.python.org/downloads/'
    Write-Host 'During setup, tick "Add python.exe to PATH".'
    exit 1
}

$forwarded = @('-m', 'installer', 'install', '--profile', $ModelProfile)
if ($RecreateVenv) { $forwarded += '--recreate-venv' }
if ($Cpu)          { $forwarded += '--cpu' }
if ($SkipTorch)    { $forwarded += '--skip-torch' }
if ($SkipModels)   { $forwarded += '--skip-models' }
if ($QualityModels) { $forwarded += '--with-quality-models' }

& $python.Exe @($python.Args + $forwarded)
$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host ''
    Write-Host "Installation stopped with exit code $code." -ForegroundColor Red
    Write-Host 'Re-running this script resumes partial downloads.'
}
exit $code
