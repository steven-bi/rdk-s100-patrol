[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-SafeValue {
    if ($User -notmatch '^[A-Za-z_][A-Za-z0-9_-]*$') {
        throw "Unsafe SSH user value."
    }
    if ($BoardHost -notmatch '^[A-Za-z0-9.:-]+$') {
        throw "Unsafe board host value."
    }
}

function Assert-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command is missing: $Name. Install Windows OpenSSH Client."
    }
}

Assert-SafeValue
Assert-Command "scp.exe"
Assert-Command "ssh.exe"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$localTool = Join-Path $projectRoot "scripts\tools\collect_gs130w_pairs_web.py"
if (-not (Test-Path -LiteralPath $localTool -PathType Leaf)) {
    throw "Local collector tool is missing: $localTool"
}

$target = "$User@$BoardHost"
$remoteTemporary = "/tmp/collect_gs130w_pairs_web.py"
$remoteTool = "/opt/rdk-patrol/current/scripts/tools/collect_gs130w_pairs_web.py"
$logPath = Join-Path $env:TEMP (
    "GS130W_manual_collector_install_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)
$transcriptStarted = $false

try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Local tool: $localTool"
    Write-Host "Board: $target"
    Write-Host "Log: $logPath"

    if ($DryRun) {
        Write-Host "DRY RUN: network commands were not executed."
        Write-Host "WEB_COLLECTOR_DRY_RUN_OK"
        exit 0
    }

    Write-Host ""
    Write-Host "Step 1/2: upload the collector to the board temporary directory."
    Write-Host "The SSH password prompt does not display typed characters."
    & scp.exe -P "$SshPort" -o "ConnectTimeout=10" $localTool "${target}:$remoteTemporary"
    if ($LASTEXITCODE -ne 0) {
        throw "SCP upload failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "Step 2/2: install and verify the collector."
    $remoteCommand = (
        "sudo install -o root -g root -m 0755 " +
        "$remoteTemporary $remoteTool && " +
        "rm -f $remoteTemporary && " +
        "/opt/rdk-patrol/current/.venv/bin/python $remoteTool --help >/dev/null && " +
        "echo WEB_COLLECTOR_READY"
    )
    & ssh.exe -t -p "$SshPort" -o "ConnectTimeout=10" $target $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote installation failed with exit code $LASTEXITCODE."
    }
    Write-Host ""
    Write-Host "WEB_COLLECTOR_READY"
    Write-Host "Installation completed. Keep this window for verification."
    exit 0
}
catch {
    Write-Host ""
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Installation was not completed. Do not start calibration capture."
    Write-Host "Send a screenshot of this window and the log path shown above."
    exit 1
}
finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # The main result has already been reported.
        }
    }
}
