[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",

    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,

    [ValidateRange(10, 600)]
    [int]$BoardWaitSeconds = 180,

    [string]$LocalToolPath = "",

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

function Assert-SafeInputs {
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

function Wait-TcpEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $timer = [Diagnostics.Stopwatch]::StartNew()
    $attempt = 0
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $attempt++
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $null
        try {
            $async = $client.BeginConnect($HostName, $Port, $null, $null)
            if ($async.AsyncWaitHandle.WaitOne(1000, $false)) {
                $client.EndConnect($async)
                if ($client.Connected) {
                    Write-Host "BOARD_SSH_READY"
                    return
                }
            }
        }
        catch {
            # The board may still be booting.
        }
        finally {
            if ($null -ne $async) {
                $async.AsyncWaitHandle.Close()
            }
            $client.Close()
        }

        if (($attempt % 5) -eq 0) {
            Write-Host ("Waiting for {0}:{1} ... {2}s" -f $HostName, $Port, [int]$timer.Elapsed.TotalSeconds)
        }
        Start-Sleep -Seconds 1
    }
    throw "Timed out waiting for board SSH at ${HostName}:${Port}."
}

Assert-SafeInputs
Assert-Command "scp.exe"
Assert-Command "ssh.exe"

if ([string]::IsNullOrWhiteSpace($LocalToolPath)) {
    # Installed layout: scripts\windows (this file) and scripts\tools (Python).
    # The second path keeps this staging bundle directly testable before copy.
    $installedTool = Join-Path $PSScriptRoot "..\tools\validate_gs130w_distance_web.py"
    $stagingTool = Join-Path (Split-Path -Parent $PSScriptRoot) "validate_gs130w_distance_web.py"
    if (Test-Path -LiteralPath $installedTool -PathType Leaf) {
        $LocalToolPath = $installedTool
    }
    else {
        $LocalToolPath = $stagingTool
    }
}
if (-not (Test-Path -LiteralPath $LocalToolPath -PathType Leaf)) {
    throw "Local distance validator is missing: $LocalToolPath"
}
$LocalToolPath = (Resolve-Path -LiteralPath $LocalToolPath).Path

$target = "$User@$BoardHost"
$remoteTemporary = "/tmp/validate_gs130w_distance_web.py"
$remoteTool = "/opt/rdk-patrol/current/scripts/tools/validate_gs130w_distance_web.py"
$candidateYaml = "/var/lib/rdk-patrol/calibration/stereo_gs130w_20260805_v2_candidate_01.yaml"
$candidateSha256 = "10FC69CFCC28B606DB40955EA2727D4B5D0DE2F378EB5A969D9D5EC0A62300C6"
$logPath = Join-Path $env:TEMP (
    "GS130W_distance_validator_install_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

if ($DryRun) {
    Write-Host "DRY RUN: no network connection was opened and no file was written."
    Write-Host "Local tool: $LocalToolPath"
    Write-Host "Board: $target"
    Write-Host "Remote tool: $remoteTool"
    Write-Host "Candidate YAML: $candidateYaml"
    Write-Host "Required candidate SHA256: $candidateSha256"
    Write-Host "DISTANCE_VALIDATOR_INSTALL_DRY_RUN_OK"
    exit 0
}

$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true

    Write-Host "Local tool: $LocalToolPath"
    Write-Host "Board: $target"
    Write-Host "Log: $logPath"
    Write-Host "Waiting for the board to finish booting."
    Wait-TcpEndpoint -HostName $BoardHost -Port $SshPort -TimeoutSeconds $BoardWaitSeconds

    Write-Host ""
    Write-Host "Step 1/2: uploading the validator."
    Write-Host "The SSH password prompt does not display typed characters."
    $scpArgs = @(
        "-P", "$SshPort",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        $LocalToolPath,
        "${target}:$remoteTemporary"
    )
    & scp.exe @scpArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SCP upload failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "Step 2/2: installing and checking the validator."
    $remoteCommand = (
        "sudo install -o root -g root -m 0755 $remoteTemporary $remoteTool && " +
        "rm -f $remoteTemporary && " +
        "cd /opt/rdk-patrol/current && " +
        "export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 && " +
        "./.venv/bin/python $remoteTool --help >/dev/null && " +
        "test -r $candidateYaml && " +
        "test `$(sha256sum $candidateYaml | awk '{print toupper(`$1)}') = $candidateSha256 && " +
        "echo DISTANCE_VALIDATOR_READY"
    )
    $sshArgs = @(
        "-t",
        "-p", "$SshPort",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        $target,
        $remoteCommand
    )
    & ssh.exe @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Remote installation failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "DISTANCE_VALIDATOR_READY" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ""
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Installation was not completed. Keep this window and send a screenshot."
    exit 1
}
finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # The result remains visible in the console.
        }
    }
}
