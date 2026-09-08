#Requires -Version 5.1
<#
.SYNOPSIS
    Removes parts of a ClauDali installation and reports how much space each
    part reclaims.

.DESCRIPTION
    A wrapper over `python -m installer uninstall`. Run with no arguments to see
    a breakdown of what exists and what it costs, deleting nothing.

    Because everything ClauDali downloads lives inside this project folder --
    including its HuggingFace cache -- removal is complete. Nothing is left
    behind in your user profile, and nothing outside this folder is touched.

    Generated images are never deleted unless you pass -Outputs or -All, and
    those require typing DELETE to confirm.

.PARAMETER Models
    Delete downloaded weights (~22 GB with the full profile). Re-downloadable.

.PARAMETER Env
    Delete the .venv\ virtual environment. Rebuilt by install.ps1.

.PARAMETER Outputs
    Delete generated images and uploads. NOT recoverable.

.PARAMETER All
    Everything above.

.PARAMETER Yes
    Skip the confirmation prompt. Intended for scripts.

.PARAMETER DryRun
    Print the plan and exit without deleting anything.

.EXAMPLE
    .\uninstall.ps1
    Shows disk usage without removing anything.

.EXAMPLE
    .\uninstall.ps1 -Models
    Frees the model weights, keeping the environment and your images.
#>
[CmdletBinding()]
param(
    [switch]$Models,
    [switch]$Env,
    [switch]$Outputs,
    [switch]$All,
    [switch]$Yes,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Find-Python {
    foreach ($candidate in @(
            @{ Exe = 'py';      Args = @('-3') },
            @{ Exe = 'python';  Args = @() },
            @{ Exe = 'python3'; Args = @() })) {
        $command = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        try {
            $version = & $candidate.Exe @($candidate.Args + '--version') 2>&1
            if ($LASTEXITCODE -eq 0 -and "$version" -match 'Python 3\.\d+') { return $candidate }
        } catch { continue }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host 'No Python found, so the uninstaller cannot run.' -ForegroundColor Red
    Write-Host 'You can remove ClauDali by hand: delete the .venv, models, outputs and runs folders.'
    exit 1
}

$forwarded = @('-m', 'installer', 'uninstall')
if ($Models)  { $forwarded += '--models' }
if ($Env)     { $forwarded += '--env' }
if ($Outputs) { $forwarded += '--outputs' }
if ($All)     { $forwarded += '--all' }
if ($Yes)     { $forwarded += '--yes' }
if ($DryRun)  { $forwarded += '--dry-run' }

& $python.Exe @($python.Args + $forwarded)
exit $LASTEXITCODE
