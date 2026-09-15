[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [ValidateRange(1, 65535)]
    [int]$LocalWebPort = 8765,
    [ValidateRange(1, 5)]
    [int]$RestoreAttempts = 3,
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

if (-not (Get-Command "ssh.exe" -ErrorAction SilentlyContinue)) {
    throw "Required command is missing: ssh.exe. Install Windows OpenSSH Client."
}
Assert-SafeValue

$target = "$User@$BoardHost"
$webUrl = "http://127.0.0.1:$LocalWebPort/"
$remoteOutput = "/var/lib/rdk-patrol/calibration/gs130w_pairs_20260729_v2"
$logPath = Join-Path $env:TEMP (
    "GS130W_manual_capture_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)
$remoteCommand = (
    "cd /opt/rdk-patrol/current && " +
    "source /opt/tros/humble/setup.bash && " +
    "if ! ros2 topic list | grep -Fxq '/image_combine_jpeg'; then " +
    "echo 'ERROR: ROS 2 camera topic /image_combine_jpeg is unavailable; rdk-patrol.service was not stopped.' >&2; " +
    "exit 69; fi; " +
    "sudo systemctl stop rdk-patrol.service && " +
    "trap 'sudo -n systemctl start rdk-patrol.service >/dev/null 2>&1 || true' EXIT && " +
    "./.venv/bin/python scripts/tools/collect_gs130w_pairs_web.py " +
    "--topic /image_combine_jpeg --layout vertical " +
    "--physical-left bottom --rotation ccw90 " +
    "--board-cols 9 --board-rows 6 --count 30 " +
    "--output $remoteOutput --host 127.0.0.1 --port $LocalWebPort " +
    "--stable-seconds 1.2 --stable-frames 3 --max-motion-px 3.5 " +
    "--max-frame-age-seconds 2.0 --capture-timeout-seconds 2.0 " +
    "--min-area-fraction 0.006 --min-outer-margin-px 16 " +
    "--wait-seconds 20 --resume"
)

$transcriptStarted = $false
$captureExit = 1
$serviceExit = 1

try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Board: $target"
    Write-Host "Browser URL: $webUrl"
    Write-Host "Output: $remoteOutput"
    Write-Host "Log: $logPath"
    Write-Host ""
    Write-Host "Keep this window open during capture."
    Write-Host "After the web collector starts, open the Browser URL shown above."
    Write-Host "Only the green Capture button can save a pair."

    if ($DryRun) {
        Write-Host ""
        Write-Host "DRY RUN: network commands were not executed."
        Write-Host "MANUAL_CAPTURE_DRY_RUN_OK"
        $captureExit = 0
        $serviceExit = 0
    }
    else {
        Write-Host ""
        Write-Host "Starting the SSH tunnel and manual collector."
        Write-Host "The SSH password prompt does not display typed characters."
        $sshArgs = @(
            "-t",
            "-p", "$SshPort",
            "-o", "ConnectTimeout=10",
            "-o", "ExitOnForwardFailure=yes",
            "-L", "${LocalWebPort}:127.0.0.1:${LocalWebPort}",
            $target,
            $remoteCommand
        )
        & ssh.exe @sshArgs
        $captureExit = $LASTEXITCODE
    }
}
catch {
    Write-Host ""
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    $captureExit = 1
}
finally {
    if (-not $DryRun) {
        Write-Host ""
        Write-Host "Restoring and verifying rdk-patrol.service."
        $restoreCommand = (
            "sudo systemctl reset-failed rdk-patrol.service && " +
            "sudo systemctl start rdk-patrol.service && " +
            "sleep 3 && " +
            "systemctl is-active --quiet rdk-patrol.service && " +
            "echo RDK_PATROL_ACTIVE"
        )
        Start-Sleep -Seconds 3
        for ($attempt = 1; $attempt -le $RestoreAttempts; $attempt++) {
            Write-Host "Service restore verification attempt $attempt/$RestoreAttempts."
            try {
                & ssh.exe -t -p "$SshPort" -o "ConnectTimeout=10" $target $restoreCommand
                $serviceExit = $LASTEXITCODE
            }
            catch {
                Write-Host ("SERVICE RESTORE ERROR: " + $_.Exception.Message) -ForegroundColor Red
                $serviceExit = 1
            }
            if ($serviceExit -eq 0) {
                break
            }
            if ($attempt -lt $RestoreAttempts) {
                Write-Host "The board closed this verification connection; retrying in 3 seconds." -ForegroundColor Yellow
                Start-Sleep -Seconds 3
            }
        }
    }
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # The result remains visible in the console.
        }
    }
}

if ($serviceExit -ne 0) {
    Write-Host ""
    Write-Host "RDK service was not confirmed active. Keep this window and send a screenshot." -ForegroundColor Red
    exit 1
}
if ($captureExit -eq 0) {
    Write-Host ""
    Write-Host "Capture completed and RDK_PATROL_ACTIVE was confirmed."
    exit 0
}
if ($captureExit -eq 130) {
    Write-Host ""
    Write-Host "Capture paused safely and RDK_PATROL_ACTIVE was confirmed."
    exit 0
}

Write-Host ""
Write-Host "Capture exited with code $captureExit. RDK service restoration was attempted."
Write-Host "Do not delete or reuse the board capture directory."
exit 1
