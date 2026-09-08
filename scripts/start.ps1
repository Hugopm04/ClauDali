#Requires -Version 5.1
<#
.SYNOPSIS
    Starts the ClauDali server: the web UI and the HTTP API.

.DESCRIPTION
    Uses the project's own virtual environment, so nothing needs to be
    activated first. The server binds to localhost only unless you pass -Host.

.PARAMETER BindHost
    Address to bind. Defaults to 127.0.0.1 (this machine only). Use 0.0.0.0 to
    expose ClauDali to your local network -- there is no authentication, so do
    that only on a network you trust.

.PARAMETER Port
    Port to listen on. Defaults to 8188.

.PARAMETER NoBrowser
    Do not open a browser window.
#>
[CmdletBinding()]
param(
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 8188,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host 'ClauDali is not installed yet.' -ForegroundColor Red
    Write-Host "Run:  .\install.ps1"
    exit 1
}

if (-not $NoBrowser) {
    # Build the URL here and pass it in, rather than interpolating $using:
    # variables inside the job's script block, where the colon before the port
    # needs escaping and is easy to get subtly wrong.
    $url = 'http://{0}:{1}' -f $BindHost, $Port
    Start-Job -ArgumentList $url -ScriptBlock {
        param($Address)
        # Give uvicorn a moment to bind before the browser asks for the page.
        Start-Sleep -Seconds 3
        Start-Process $Address
    } | Out-Null
}

& $python -m claudali serve --host $BindHost --port $Port
exit $LASTEXITCODE
