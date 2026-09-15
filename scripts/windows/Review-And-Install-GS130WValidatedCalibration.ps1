[CmdletBinding()]
param(
    [string]$ValidatedYaml = "",
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$PythonPath = "D:\miniconda\python.exe",
    [switch]$DryRun,
    [switch]$Yes,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command is missing: $Name"
    }
}

function Assert-SafeConnectionValues {
    if ($User -notmatch '^[A-Za-z_][A-Za-z0-9_-]*$') {
        throw "Unsafe SSH user value."
    }
    if ($BoardHost -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]*$') {
        throw "Unsafe board host value. Use an IPv4 address or a DNS hostname."
    }
    if ($User -cne "sunrise") {
        throw "This production installer is locked to the audited RDK service user: sunrise."
    }
}

function Test-RequiredInstallMarkers {
    param([Parameter(Mandatory = $true)][object[]]$Lines)
    $normalized = @($Lines | ForEach-Object { ([string]$_).Trim() })
    return (
        ($normalized -contains "RDK_PATROL_ACTIVE") -and
        ($normalized -contains "GS130W_VALIDATED_CALIBRATION_INSTALL_OK")
    )
}

if ($SelfTest) {
    if (Test-RequiredInstallMarkers -Lines @("RDK_PATROL_ACTIVE")) {
        throw "Marker self-test accepted a missing install-success marker."
    }
    if (Test-RequiredInstallMarkers -Lines @("GS130W_VALIDATED_CALIBRATION_INSTALL_OK")) {
        throw "Marker self-test accepted a missing service-active marker."
    }
    if (-not (Test-RequiredInstallMarkers -Lines @(
        "noise",
        "RDK_PATROL_ACTIVE",
        "GS130W_VALIDATED_CALIBRATION_INSTALL_OK"
    ))) {
        throw "Marker self-test rejected both required success markers."
    }
    Write-Host "INSTALLER_SELF_TEST_OK"
    exit 0
}

if ($PSVersionTable.PSVersion.Major -lt 5) {
    throw "Windows PowerShell 5.1 or newer is required."
}

Write-Host "GS130W validated-calibration review and production installer"
Write-Host "This tool is NOT part of capture/validation and is never run automatically."
Write-Host "Run it only after the 1m/3m/5m session has completed and a human has reviewed the result."
Write-Host ""

if ([string]::IsNullOrWhiteSpace($ValidatedYaml)) {
    $ValidatedYaml = Read-Host "Paste the full path of the exported validated YAML"
}
$ValidatedYaml = $ValidatedYaml.Trim().Trim('"')
if ([string]::IsNullOrWhiteSpace($ValidatedYaml)) {
    throw "No validated YAML path was supplied."
}

$validatedItem = Get-Item -LiteralPath $ValidatedYaml -Force -ErrorAction Stop
if ($validatedItem.PSIsContainer) {
    throw "ValidatedYaml must name a file, not a directory."
}
$ValidatedYaml = $validatedItem.FullName

$verifier = Join-Path $PSScriptRoot "verify_validated_calibration.py"
if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) {
    throw "The read-only verification helper is missing: $verifier"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python was not found at $PythonPath. Keep the approved D:\miniconda\python.exe installation available."
}

Write-Host "Step 1/4: strict local verification (read-only)"
$verifyLines = @(& $PythonPath -B $verifier --validated $ValidatedYaml --json)
$verifyExit = $LASTEXITCODE
$verifyText = $verifyLines -join ""
if ($verifyExit -ne 0) {
    $detail = $verifyText
    try {
        $failedReport = $verifyText | ConvertFrom-Json
        if ($failedReport.error) {
            $detail = [string]$failedReport.error
        }
    }
    catch {
        # Preserve the verifier's original output when it is not JSON.
    }
    throw "Strict verification failed. Production was not changed. Detail: $detail"
}

try {
    $report = $verifyText | ConvertFrom-Json
}
catch {
    throw "The verifier returned unreadable JSON. Production was not changed."
}
if ($report.ok -ne $true) {
    throw "The verifier did not return ok=true. Production was not changed."
}

$validatedHash = [string]$report.validated_yaml_sha256
$candidateHash = [string]$report.source_candidate_sha256
if ($validatedHash -notmatch '^[0-9a-f]{64}$' -or $candidateHash -notmatch '^[0-9a-f]{64}$') {
    throw "The verifier returned an invalid SHA-256 value."
}

Write-Host "STRICT_LOCAL_VERIFICATION_OK" -ForegroundColor Green
Write-Host ("Session:             " + [string]$report.session_id)
Write-Host ("Validated SHA-256:   " + $validatedHash)
Write-Host ("Candidate SHA-256:   " + $candidateHash)
Write-Host ("Baseline:            {0:N9} m" -f [double]$report.baseline_m)
Write-Host ("Calibration pairs:   " + [string]$report.used_pair_count)
foreach ($point in $report.points) {
    Write-Host (
        "{0}/{1}: known={2:N3} m, estimated={3:N3} m, error={4:P2}, passing frames={5}/3" -f
        [string]$point.point,
        [string]$point.slot,
        [double]$point.known_distance_m,
        [double]$point.estimated_distance_m,
        [double]$point.relative_error,
        [int]$point.passing_frames
    )
}

Write-Host ""
Write-Host "Step 2/4: installation plan"
Write-Host "Board:               $User@$BoardHost`:$SshPort"
Write-Host "Production site file: /var/lib/rdk-patrol/site-config/stereo_gs130w.yaml"
Write-Host "Runtime config file:  /opt/rdk-patrol/current/configs/stereo_gs130w.yaml"
Write-Host "A versioned remote backup will be retained before replacement."
Write-Host "If apply-site-config or preflight fails, both files are restored automatically."

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN: no network connection was made and no file was written." -ForegroundColor Yellow
    Write-Host "GS130W_VALIDATED_INSTALL_DRY_RUN_OK" -ForegroundColor Green
    exit 0
}

Assert-SafeConnectionValues
Assert-Command "scp.exe"
Assert-Command "ssh.exe"

if (-not $Yes) {
    Write-Host ""
    Write-Warning "The next step changes the board's production stereo calibration."
    $confirmation = Read-Host "Type INSTALL to continue; anything else cancels"
    if ($confirmation -cne "INSTALL") {
        Write-Host "Cancelled. Production was not changed."
        exit 3
    }
}

$target = "$User@$BoardHost"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$hashPrefix = $validatedHash.Substring(0, 12)
$remoteUpload = "/tmp/rdk-patrol-gs130w-validated-$stamp-$hashPrefix.yaml"

Write-Host ""
Write-Host "Step 3/4: upload the hash-bound validated YAML"
$scpArgs = @(
    "-P", "$SshPort",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=3",
    $ValidatedYaml,
    "${target}:$remoteUpload"
)
& scp.exe @scpArgs
if ($LASTEXITCODE -ne 0) {
    throw "Upload failed. Production was not changed."
}

$remoteInstaller = @'
set -Eeuo pipefail

upload="$1"
expected_hash="$2"
stamp="$3"
service_user="$4"
site_dir="/var/lib/rdk-patrol/site-config"
site_file="$site_dir/stereo_gs130w.yaml"
runtime_dir="/opt/rdk-patrol/current/configs"
runtime_file="$runtime_dir/stereo_gs130w.yaml"
backup_dir="$site_dir/backups"
apply_script="/opt/rdk-patrol/current/scripts/rdk/apply-site-config.sh"
preflight_script="/opt/rdk-patrol/current/scripts/rdk/preflight.sh"
hash_short="${expected_hash:0:12}"
site_backup="$backup_dir/stereo_gs130w_before_${stamp}_${hash_short}.yaml"
runtime_backup="$backup_dir/runtime_stereo_gs130w_before_${stamp}_${hash_short}.yaml"
site_stage="$site_dir/.stereo_gs130w.install-${stamp}-${hash_short}.tmp"
runtime_rollback_stage="$runtime_dir/.stereo_gs130w.rollback-${stamp}-${hash_short}.tmp"
site_rollback_stage="$site_dir/.stereo_gs130w.rollback-${stamp}-${hash_short}.tmp"
lock_dir="$site_dir/.stereo_gs130w.install.lock"

sudo_ready=0
lock_acquired=0
production_changed=0
committed=0
old_site_hash=""
old_runtime_hash=""

finish() {
    rc=$?
    trap - EXIT INT TERM HUP
    rollback_rc=0
    if [[ "$production_changed" == "1" && "$committed" != "1" ]]; then
        echo "INSTALL_FAILED_ROLLBACK_START" >&2
        set +e
        sudo install -o root -g root -m 0644 -- "$site_backup" "$site_rollback_stage"
        first_rc=$?
        if [[ $first_rc -eq 0 ]]; then
            sudo mv -f -- "$site_rollback_stage" "$site_file"
            first_rc=$?
        fi
        sudo install -o root -g root -m 0644 -- "$runtime_backup" "$runtime_rollback_stage"
        second_rc=$?
        if [[ $second_rc -eq 0 ]]; then
            sudo mv -f -- "$runtime_rollback_stage" "$runtime_file"
            second_rc=$?
        fi
        sudo systemctl reset-failed rdk-patrol.service >/dev/null 2>&1
        reset_rc=$?
        sudo systemctl start rdk-patrol.service >/dev/null 2>&1
        start_rc=$?
        sleep 3
        sudo systemctl is-active --quiet rdk-patrol.service >/dev/null 2>&1
        active_rc=$?
        restored_site_hash="$(sudo sha256sum -- "$site_file" 2>/dev/null | awk '{print $1}')"
        restored_runtime_hash="$(sudo sha256sum -- "$runtime_file" 2>/dev/null | awk '{print $1}')"
        if [[ $first_rc -ne 0 || $second_rc -ne 0 || $reset_rc -ne 0 || $start_rc -ne 0 || $active_rc -ne 0 || "$restored_site_hash" != "$old_site_hash" || "$restored_runtime_hash" != "$old_runtime_hash" ]]; then
            rollback_rc=1
            echo "ROLLBACK_FAILED_MANUAL_RECOVERY_REQUIRED" >&2
            echo "SITE_BACKUP=$site_backup" >&2
            echo "RUNTIME_BACKUP=$runtime_backup" >&2
        else
            echo "ROLLBACK_COMPLETED_AND_RDK_PATROL_ACTIVE" >&2
        fi
        set -e
    fi
    if [[ "$sudo_ready" == "1" ]]; then
        sudo rm -f -- "$site_stage" "$site_rollback_stage" "$runtime_rollback_stage" >/dev/null 2>&1 || true
    fi
    if [[ "$lock_acquired" == "1" ]]; then
        sudo rmdir -- "$lock_dir" >/dev/null 2>&1 || true
    fi
    rm -f -- "$upload" >/dev/null 2>&1 || true
    if [[ $rollback_rc -ne 0 ]]; then
        exit 90
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

[[ "$upload" =~ ^/tmp/rdk-patrol-gs130w-validated-[0-9]{8}_[0-9]{6}-[0-9a-f]{12}\.yaml$ ]]
[[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]]
[[ "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]
[[ "$service_user" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]]
[[ -s "$upload" ]]
for command_name in base64 sha256sum awk install mv mkdir rmdir systemctl sudo; do
    command -v "$command_name" >/dev/null 2>&1
done
[[ "$(sha256sum -- "$upload" | awk '{print $1}')" == "$expected_hash" ]]
[[ -s "$site_file" ]]
[[ -s "$runtime_file" ]]
[[ -s "$site_dir/system.yaml" ]]
[[ -s "$site_dir/points.yaml" ]]
[[ -x "$apply_script" ]]
[[ -x "$preflight_script" ]]
[[ ! -e "$site_backup" ]]
[[ ! -e "$runtime_backup" ]]

sudo -v
sudo_ready=1
if ! sudo mkdir -- "$lock_dir"; then
    echo "INSTALL_LOCK_EXISTS_NO_PRODUCTION_CHANGE: $lock_dir" >&2
    exit 75
fi
lock_acquired=1
sudo test ! -e "$site_backup"
sudo test ! -e "$runtime_backup"
sudo test ! -e "$site_stage"
sudo test ! -e "$site_rollback_stage"
sudo test ! -e "$runtime_rollback_stage"
sudo install -d -o root -g root -m 0755 -- "$backup_dir"
old_site_hash="$(sudo sha256sum -- "$site_file" | awk '{print $1}')"
old_runtime_hash="$(sudo sha256sum -- "$runtime_file" | awk '{print $1}')"
sudo install -o root -g root -m 0644 -- "$site_file" "$site_backup"
sudo install -o root -g root -m 0644 -- "$runtime_file" "$runtime_backup"
[[ "$(sudo sha256sum -- "$site_backup" | awk '{print $1}')" == "$old_site_hash" ]]
[[ "$(sudo sha256sum -- "$runtime_backup" | awk '{print $1}')" == "$old_runtime_hash" ]]

sudo install -o root -g root -m 0644 -- "$upload" "$site_stage"
[[ "$(sudo sha256sum -- "$site_stage" | awk '{print $1}')" == "$expected_hash" ]]
production_changed=1
sudo mv -f -- "$site_stage" "$site_file"
[[ "$(sudo sha256sum -- "$site_file" | awk '{print $1}')" == "$expected_hash" ]]

sudo "$apply_script"
sleep 3
sudo -u "$service_user" "$preflight_script"
[[ "$(sudo sha256sum -- "$site_file" | awk '{print $1}')" == "$expected_hash" ]]
[[ "$(sudo sha256sum -- "$runtime_file" | awk '{print $1}')" == "$expected_hash" ]]
sudo systemctl is-active --quiet rdk-patrol.service

committed=1
echo "SITE_BACKUP=$site_backup"
echo "RUNTIME_BACKUP=$runtime_backup"
echo "RDK_PATROL_ACTIVE"
echo "GS130W_VALIDATED_CALIBRATION_INSTALL_OK"
'@

$remoteBytes = [System.Text.Encoding]::UTF8.GetBytes($remoteInstaller)
$remoteBase64 = [Convert]::ToBase64String($remoteBytes)
$remotePipeline = (
    "printf %s $remoteBase64 | base64 -d | bash -s -- " +
    "$remoteUpload $validatedHash $stamp $User"
)
$remoteCommand = "bash -o pipefail -c '$remotePipeline'"

Write-Host ""
Write-Host "Step 4/4: backup, atomically install, apply, and run preflight"
$sshArgs = @(
    "-tt",
    "-p", "$SshPort",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=3",
    $target,
    $remoteCommand
)
$sshOutput = @()
& ssh.exe @sshArgs 2>&1 | ForEach-Object {
    $line = [string]$_
    $sshOutput += $line
    Write-Host $line
}
$installExit = $LASTEXITCODE
if ($installExit -ne 0) {
    throw (
        "Board installation failed with exit code $installExit. " +
        "If the output says ROLLBACK_COMPLETED, the old production files were restored. " +
        "If it says ROLLBACK_FAILED, keep the window open and recover from the displayed backup paths."
    )
}
if (-not (Test-RequiredInstallMarkers -Lines $sshOutput)) {
    throw (
        "SSH returned exit code 0 but one or both mandatory success markers were absent. " +
        "Treat the installation as unconfirmed and inspect the board before retrying."
    )
}

Write-Host ""
Write-Host "Installation completed: strict validation, backup, atomic replacement, apply-site-config, preflight, and service check all passed." -ForegroundColor Green
Write-Host "GS130W_VALIDATED_CALIBRATION_INSTALL_OK" -ForegroundColor Green
exit 0
