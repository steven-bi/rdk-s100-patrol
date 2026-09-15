[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",

    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,

    [ValidateRange(1, 65535)]
    [int]$LocalWebPort = 8766,

    [ValidateRange(10, 600)]
    [int]$BoardWaitSeconds = 180,

    [ValidateRange(30, 7200)]
    [int]$BrowserWaitSeconds = 3600,

    [ValidateRange(1, 3)]
    [int]$RestoreAttempts = 3,

    [string]$SessionId = "",

    [switch]$Resume,

    [string]$LocalToolPath = "",

    [string]$CandidateYaml = "/var/lib/rdk-patrol/calibration/stereo_gs130w_20260805_v2_candidate_01.yaml",

    [string]$CandidateSha256 = "10FC69CFCC28B606DB40955EA2727D4B5D0DE2F378EB5A969D9D5EC0A62300C6",

    [string]$RuntimeDepthSha256 = "82AA6716CBF43078F0FA563770497BADD9C514F0A3D066F6C603D58637D86ED7",

    [string]$RuntimeCalibrationSha256 = "1377BF7982B96F1EC07664C0F4014B2B26624C76779A2DECD7C0D59B0481E65F",

    [string]$RuntimeGs130wSha256 = "2DB1C0350FC275E1A3D784DFE85B0ECEBFA355DC61E558D0FDDE96FFB9E6EC9B",

    [string]$Topic = "/image_combine_jpeg",

    [string]$LocalSessionsRoot = "",

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
    if ($CandidateYaml -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw "Unsafe candidate YAML path."
    }
    if ($Topic -notmatch '^/[A-Za-z0-9_/]+$') {
        throw "Unsafe ROS topic value."
    }
    if ($CandidateSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Candidate SHA256 must contain exactly 64 hexadecimal characters."
    }
    foreach ($runtimeHash in @($RuntimeDepthSha256, $RuntimeCalibrationSha256, $RuntimeGs130wSha256)) {
        if ($runtimeHash -notmatch '^[0-9a-fA-F]{64}$') {
            throw "Every approved runtime-source SHA256 must contain exactly 64 hexadecimal characters."
        }
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

function Assert-LocalPortFree {
    param([Parameter(Mandatory = $true)][int]$Port)

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $Port
        )
        $listener.Start()
    }
    catch {
        throw "Local TCP port $Port is already in use. Close the previous validator window first."
    }
    finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Assert-DownloadedSession {
    param([Parameter(Mandatory = $true)][string]$SessionPath)

    $manifestPath = Join-Path $SessionPath "session.json"
    $auditPath = Join-Path $SessionPath "audit.jsonl"
    $checksumsPath = Join-Path $SessionPath "SESSION_SHA256SUMS.txt"
    $validatedPath = Join-Path $SessionPath "stereo_gs130w_validated.yaml"
    foreach ($required in @($manifestPath, $auditPath, $checksumsPath, $validatedPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Downloaded session is missing required file: $required"
        }
    }

    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "Downloaded session.json is not valid JSON: $($_.Exception.Message)"
    }
    if ([string]$manifest.status -ne "completed") {
        throw "Downloaded session is not completed. Status: $($manifest.status)"
    }
    if ([string]$manifest.session_id -ne $SessionId) {
        throw "Downloaded session ID does not match the requested session."
    }
    if ([string]$manifest.candidate_sha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Downloaded manifest has no valid candidate SHA256."
    }
    if (([string]$manifest.candidate_sha256).ToUpperInvariant() -ne $CandidateSha256.ToUpperInvariant()) {
        throw "Downloaded session is bound to a different candidate SHA256."
    }
    if ([string]$manifest.validated_yaml_sha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Downloaded manifest has no valid validated-YAML SHA256."
    }

    $slotNames = @($manifest.slots.PSObject.Properties.Name)
    if ($slotNames.Count -ne 3) {
        throw "Downloaded manifest does not contain exactly three distance slots."
    }
    foreach ($slotName in @("1m", "3m", "5m")) {
        $slotProperty = $manifest.slots.PSObject.Properties[$slotName]
        if ($null -eq $slotProperty -or
            -not [bool]$slotProperty.Value.locked -or
            $null -eq $slotProperty.Value.aggregate -or
            -not [bool]$slotProperty.Value.aggregate.passed) {
            throw "Downloaded manifest slot $slotName is not locked and passed."
        }
    }

    $actualValidatedHash = (Get-FileHash -LiteralPath $validatedPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedValidatedHash = ([string]$manifest.validated_yaml_sha256).ToLowerInvariant()
    if ($actualValidatedHash -ne $expectedValidatedHash) {
        throw "Validated YAML SHA256 does not match session.json."
    }
    $validatedText = Get-Content -LiteralPath $validatedPath -Raw -Encoding UTF8
    if ($validatedText -notmatch '(?m)^\s*valid:\s*true\s*(?:#.*)?$') {
        throw "Validated YAML does not contain valid: true."
    }

    $sessionFull = [IO.Path]::GetFullPath($SessionPath)
    $sessionPrefix = $sessionFull.TrimEnd([char[]]@('\', '/')) + [IO.Path]::DirectorySeparatorChar
    $validatedCovered = $false
    $checksumCount = 0
    $knownFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($known in @(
        "session.json",
        "audit.jsonl",
        "SESSION_SHA256SUMS.txt",
        "audit.zip",
        "audit.zip.sha256",
        "GS130W_distance_validation_audit.zip"
    )) {
        [void]$knownFiles.Add($known)
    }
    foreach ($line in (Get-Content -LiteralPath $checksumsPath -Encoding UTF8)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            throw "Invalid line in SESSION_SHA256SUMS.txt: $line"
        }
        $expectedHash = $Matches[1].ToLowerInvariant()
        $relative = $Matches[2]
        if ([IO.Path]::IsPathRooted($relative) -or $relative -match '(^|/)\.\.(/|$)' -or $relative.Contains(':')) {
            throw "Unsafe relative path in SESSION_SHA256SUMS.txt: $relative"
        }
        $localRelative = $relative.Replace('/', [IO.Path]::DirectorySeparatorChar)
        $filePath = [IO.Path]::GetFullPath((Join-Path $sessionFull $localRelative))
        if (-not $filePath.StartsWith($sessionPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Checksum path escaped the downloaded session: $relative"
        }
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
            throw "Checksum references a missing file: $relative"
        }
        $actualHash = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "SHA256 mismatch for downloaded file: $relative"
        }
        if ($relative -eq "stereo_gs130w_validated.yaml") {
            $validatedCovered = $true
        }
        [void]$knownFiles.Add($relative.Replace('\', '/'))
        $checksumCount++
    }
    if ($checksumCount -lt 1 -or -not $validatedCovered) {
        throw "Session checksums do not cover the validated YAML."
    }
    foreach ($downloadedFile in (Get-ChildItem -LiteralPath $sessionFull -Recurse -Force -File)) {
        $downloadedFull = [IO.Path]::GetFullPath($downloadedFile.FullName)
        if (-not $downloadedFull.StartsWith($sessionPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Downloaded file escaped the audit session: $downloadedFull"
        }
        $downloadedRelative = $downloadedFull.Substring($sessionPrefix.Length).Replace('\', '/')
        if (-not $knownFiles.Contains($downloadedRelative)) {
            throw "Downloaded audit contains an unlisted file: $downloadedRelative"
        }
    }
}

function Expand-AuditZipSafely {
    param(
        [Parameter(Mandatory = $true)][string]$ZipPath,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $destinationFull = [IO.Path]::GetFullPath($Destination)
    $destinationPrefix = $destinationFull.TrimEnd([char[]]@('\', '/')) + [IO.Path]::DirectorySeparatorChar
    $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $archive = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        if ($archive.Entries.Count -lt 1) {
            throw "Downloaded audit ZIP is empty."
        }
        foreach ($entry in $archive.Entries) {
            $relative = [string]$entry.FullName
            $normalized = $relative.Replace('\', '/')
            if ([string]::IsNullOrWhiteSpace($normalized) -or
                $normalized.StartsWith('/') -or
                $normalized -match '(^|/)\.\.(/|$)' -or
                $normalized.Contains(':')) {
                throw "Unsafe path in downloaded audit ZIP: $relative"
            }
            if (-not $names.Add($normalized)) {
                throw "Duplicate path in downloaded audit ZIP: $relative"
            }
            $localRelative = $normalized.Replace('/', [IO.Path]::DirectorySeparatorChar)
            $expandedPath = [IO.Path]::GetFullPath((Join-Path $destinationFull $localRelative))
            if (-not $expandedPath.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Audit ZIP path escaped the destination: $relative"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
    [IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $Destination)
}

function New-VerifiedLocalAuditZip {
    param(
        [Parameter(Mandatory = $true)][string]$SessionPath,
        [Parameter(Mandatory = $true)][string]$DestinationZip
    )

    if (Test-Path -LiteralPath $DestinationZip) {
        throw "Refusing to overwrite audit ZIP destination: $DestinationZip"
    }
    $checksumsPath = Join-Path $SessionPath "SESSION_SHA256SUMS.txt"
    if (-not (Test-Path -LiteralPath $checksumsPath -PathType Leaf)) {
        throw "Cannot rebuild audit ZIP without SESSION_SHA256SUMS.txt."
    }

    $relativePaths = New-Object 'System.Collections.Generic.List[string]'
    foreach ($required in @("session.json", "audit.jsonl", "SESSION_SHA256SUMS.txt")) {
        $relativePaths.Add($required)
    }
    foreach ($line in (Get-Content -LiteralPath $checksumsPath -Encoding UTF8)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') {
            throw "Invalid checksum line while rebuilding local audit ZIP: $line"
        }
        $relativePaths.Add($Matches[2])
    }

    $sessionFull = [IO.Path]::GetFullPath($SessionPath)
    $sessionPrefix = $sessionFull.TrimEnd([char[]]@('\', '/')) + [IO.Path]::DirectorySeparatorChar
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $temporaryZip = "$DestinationZip.creating_$PID"
    if (Test-Path -LiteralPath $temporaryZip) {
        throw "Refusing to overwrite partial audit ZIP evidence: $temporaryZip"
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = $null
    try {
        $archive = [IO.Compression.ZipFile]::Open(
            $temporaryZip,
            [IO.Compression.ZipArchiveMode]::Create
        )
        foreach ($relative in $relativePaths) {
            $normalized = ([string]$relative).Replace('\', '/')
            if ([string]::IsNullOrWhiteSpace($normalized) -or
                $normalized.StartsWith('/') -or
                $normalized -match '(^|/)\.\.(/|$)' -or
                $normalized.Contains(':')) {
                throw "Unsafe path while rebuilding local audit ZIP: $relative"
            }
            if (-not $seen.Add($normalized)) {
                continue
            }
            $localRelative = $normalized.Replace('/', [IO.Path]::DirectorySeparatorChar)
            $sourcePath = [IO.Path]::GetFullPath((Join-Path $sessionFull $localRelative))
            if (-not $sourcePath.StartsWith($sessionPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Audit ZIP source escaped the verified session: $relative"
            }
            if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
                throw "Verified audit ZIP source is missing: $relative"
            }
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $sourcePath,
                $normalized,
                [IO.Compression.CompressionLevel]::Optimal
            )
        }
    }
    finally {
        if ($null -ne $archive) {
            $archive.Dispose()
        }
    }
    Move-Item -LiteralPath $temporaryZip -Destination $DestinationZip
}

function Stop-TranscriptSafely {
    if ($script:transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        }
        catch {
            # The main result remains visible in the console.
        }
        $script:transcriptStarted = $false
    }
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

if ([string]::IsNullOrWhiteSpace($SessionId)) {
    if ($Resume) {
        throw "-Resume requires the original -SessionId."
    }
    $SessionId = "gs130w_3point_{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $PID
}
if ($SessionId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$') {
    throw "Unsafe session ID. Use 3-128 characters: first alphanumeric, then letters, digits, dot, underscore, or hyphen."
}

$target = "$User@$BoardHost"
$webUrl = "http://127.0.0.1:$LocalWebPort/"
$remoteProjectRoot = "/opt/rdk-patrol/current"
$remoteTool = "$remoteProjectRoot/scripts/tools/validate_gs130w_distance_web.py"
$remoteTemporary = "/tmp/validate_gs130w_distance_web_$SessionId.py"
$remoteSessionParent = "/var/lib/rdk-patrol/calibration/known_distance_validation_sessions"
$remoteSession = "$remoteSessionParent/$SessionId"
$remoteValidated = "$remoteSession/stereo_gs130w_validated.yaml"
$browserHelper = Join-Path $PSScriptRoot "Open-GS130WDistanceValidationBrowser.ps1"
$logPath = Join-Path $env:TEMP (
    "GS130W_distance_validation_{0}.log" -f $SessionId
)

if (-not (Test-Path -LiteralPath $browserHelper -PathType Leaf)) {
    throw "Browser helper is missing: $browserHelper"
}

if ([string]::IsNullOrWhiteSpace($LocalSessionsRoot)) {
    $LocalSessionsRoot = "D:\RDK_S100_Patrol_Final\calibration_data\GS130W_20260805\known_distance_validation\sessions"
}
$localSession = Join-Path $LocalSessionsRoot $SessionId
$incomingName = ".{0}.incoming_{1}_{2}" -f $SessionId, (Get-Date -Format "yyyyMMdd_HHmmss"), $PID
$incomingPath = Join-Path $LocalSessionsRoot $incomingName
$auditZipPath = Join-Path $incomingPath "audit.zip"

if ($Resume) {
    $sessionGuard = (
        "if [ ! -d $remoteSession ]; then " +
        "echo 'ERROR: resume session does not exist: $remoteSession' >&2; exit 73; fi"
    )
    $resumeArgument = "--resume"
}
else {
    $sessionGuard = (
        "if [ -e $remoteSession ]; then " +
        "echo 'ERROR: refusing to overwrite existing session: $remoteSession' >&2; exit 73; fi"
    )
    $resumeArgument = ""
}

if ($DryRun) {
    Write-Host "DRY RUN: no network connection was opened and no file was written."
    Write-Host "Local tool: $LocalToolPath"
    Write-Host "Board: $target"
    Write-Host "Browser URL: $webUrl"
    Write-Host "Candidate YAML: $CandidateYaml"
    Write-Host "Required candidate SHA256: $CandidateSha256"
    Write-Host "Approved runtime depth.py SHA256: $RuntimeDepthSha256"
    Write-Host "Approved runtime calibration.py SHA256: $RuntimeCalibrationSha256"
    Write-Host "Approved runtime gs130w.py SHA256: $RuntimeGs130wSha256"
    Write-Host "Remote session: $remoteSession"
    Write-Host "Validated output: $remoteValidated"
    Write-Host "Local result: $localSession"
    Write-Host "Audit ZIP staging path: $auditZipPath"
    if ($Resume) {
        Write-Host "Mode: resume"
        Write-Host "Resume plan: read-only completion probe first; completed sessions skip camera/service/web and pull audit directly."
    }
    else {
        Write-Host "Mode: new session"
    }
    Write-Host "DISTANCE_VALIDATION_DRY_RUN_OK"
    exit 0
}

if (Test-Path -LiteralPath $localSession) {
    throw "Refusing to overwrite existing local session: $localSession"
}
if (Test-Path -LiteralPath $incomingPath) {
    throw "Refusing to overwrite existing local staging path: $incomingPath"
}

$transcriptStarted = $false
$browserProcess = $null
$validationExit = 1
$serviceExit = 1
$localAuditExit = 1
$completedResumeFastPath = $false
$resumeProbeServiceActive = $false
$fullValidationLaunched = $false

try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true

    Write-Host "Board: $target"
    Write-Host "Browser URL: $webUrl"
    Write-Host "Session: $remoteSession"
    Write-Host "Local result after completion: $localSession"
    Write-Host "Candidate YAML: $CandidateYaml"
    Write-Host "Required candidate SHA256: $CandidateSha256"
    Write-Host "Log: $logPath"
    Write-Host ""
    Write-Host "Keep this window open until all three distances pass."
    Write-Host "The browser will open automatically after the web validator is healthy."
    Write-Host "Waiting for the board to finish booting."
    Wait-TcpEndpoint -HostName $BoardHost -Port $SshPort -TimeoutSeconds $BoardWaitSeconds

    if ($Resume) {
        Write-Host ""
        Write-Host "Read-only probe: checking whether this session is already completed."
        Write-Host "This probe does not stop services and does not access the camera."
        $remoteManifest = "$remoteSession/session.json"
        $remoteChecksums = "$remoteSession/SESSION_SHA256SUMS.txt"
        $probePython = @"
import json
import pathlib
import sys
m = json.loads(pathlib.Path("$remoteManifest").read_text(encoding="utf-8"))
identity = (
    m.get("session_id") == "$SessionId"
    and str(m.get("candidate_sha256", "")).upper() == "$($CandidateSha256.ToUpperInvariant())"
)
status = str(m.get("status", ""))
sys.exit(45 if not identity else (0 if status == "completed" else (10 if status in ("paused", "in_progress") else 47)))
"@
        $probeBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($probePython))
        $probeCommand = (
            "if [ ! -r $remoteManifest ]; then exit 44; fi; " +
            "printf %s $probeBase64 | base64 -d | python3 -; probe_rc=`$?; " +
            "if [ `$probe_rc -eq 0 ]; then " +
            "if [ ! -r $remoteValidated ] || [ ! -r $remoteChecksums ]; then exit 46; fi; " +
            "echo COMPLETED_RESUME_READ_ONLY_OK; " +
            "if systemctl is-active --quiet rdk-patrol.service; then echo RDK_PATROL_ACTIVE; else echo RDK_PATROL_NOT_ACTIVE; fi; " +
            "exit 0; fi; exit `$probe_rc"
        )
        $probeArgs = @(
            "-p", "$SshPort",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3",
            $target,
            $probeCommand
        )
        $probeLines = @(& ssh.exe @probeArgs)
        $probeExit = $LASTEXITCODE
        foreach ($probeLine in $probeLines) {
            Write-Host $probeLine
        }
        if ($probeExit -eq 0) {
            if (-not ($probeLines -match 'COMPLETED_RESUME_READ_ONLY_OK')) {
                throw "Completed-resume probe returned success without its safety marker."
            }
            $completedResumeFastPath = $true
            $resumeProbeServiceActive = [bool]($probeLines -match '^RDK_PATROL_ACTIVE\s*$')
            $validationExit = 0
            Write-Host "COMPLETED_RESUME_FAST_PATH"
            Write-Host "Skipping camera checks, service stop, upload, and web startup."
        }
        elseif ($probeExit -eq 10) {
            Write-Host "Resume session is not completed; continuing with the normal camera workflow."
        }
        else {
            throw "Read-only resume probe failed with exit code $probeExit. No service was stopped."
        }
    }

    if (-not $completedResumeFastPath) {
    Assert-LocalPortFree -Port $LocalWebPort
    Write-Host ""
    Write-Host "Step 1/2: uploading the current validator."
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
    Write-Host "Step 2/2: checking live frames and starting the web validator."
    Write-Host "A controlled sudo keepalive will run only while validation is open."
    Write-Host "The SSH/sudo prompt may request the board password; typed characters are not displayed."
    Write-Host "If an abnormal recovery connection is needed later, enter the board password again when prompted."
    $helperArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"' + $browserHelper + '"'),
        "-Url", $webUrl,
        "-TimeoutSeconds", "$BrowserWaitSeconds",
        "-AuditZipPath", ('"' + $auditZipPath + '"')
    )
    $browserProcess = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $helperArgs `
        -WindowStyle Hidden `
        -PassThru

    $remoteTemplate = @'
set -e;
cd __PROJECT_ROOT__;
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8;
source /opt/tros/humble/setup.bash;
check_fresh_frames() { label="$1"; if ! ros2 topic list | grep -Fxq '__TOPIC__'; then echo "ERROR: ROS topic __TOPIC__ is unavailable ($label)." >&2; return 69; fi; if ! timeout 15 ros2 topic echo '__TOPIC__' --once --qos-reliability best_effort >/dev/null 2>&1; then echo "ERROR: no fresh frame arrived on __TOPIC__ ($label, sample 1)." >&2; return 70; fi; sleep 1; if ! timeout 15 ros2 topic echo '__TOPIC__' --once --qos-reliability best_effort >/dev/null 2>&1; then echo "ERROR: no second fresh frame arrived on __TOPIC__ ($label)." >&2; return 70; fi; echo "$label"; };
sudo_keepalive_pid='';
cleanup_service() { rc=$?; trap - EXIT INT TERM HUP; if [ -n "$sudo_keepalive_pid" ]; then kill "$sudo_keepalive_pid" >/dev/null 2>&1 || true; wait "$sudo_keepalive_pid" >/dev/null 2>&1 || true; sudo_keepalive_pid=''; fi; restore_rc=0; sudo systemctl reset-failed rdk-patrol.service >/dev/null 2>&1 || restore_rc=1; sudo systemctl start rdk-patrol.service >/dev/null 2>&1 || restore_rc=1; sleep 3; systemctl is-active --quiet rdk-patrol.service || restore_rc=1; if [ "$restore_rc" -ne 0 ]; then echo 'ERROR: rdk-patrol.service restoration was not confirmed.' >&2; exit 95; fi; echo RDK_PATROL_ACTIVE; exit "$rc"; };
sudo install -o root -g root -m 0755 __REMOTE_TEMP__ __REMOTE_TOOL__;
rm -f __REMOTE_TEMP__;
./.venv/bin/python __REMOTE_TOOL__ --help >/dev/null;
if [ ! -r __CANDIDATE__ ]; then echo 'ERROR: candidate YAML is missing or unreadable: __CANDIDATE__' >&2; exit 66; fi;
candidate_sha256=$(sha256sum __CANDIDATE__ | awk '{print toupper($1)}');
if [ "$candidate_sha256" != '__CANDIDATE_SHA256__' ]; then echo "ERROR: candidate SHA256 mismatch. actual=$candidate_sha256" >&2; exit 65; fi;
echo CANDIDATE_SHA256_OK;
depth_sha256=$(sha256sum __PROJECT_ROOT__/src/rdk_patrol/stereo/depth.py | awk '{print toupper($1)}');
calibration_sha256=$(sha256sum __PROJECT_ROOT__/src/rdk_patrol/stereo/calibration.py | awk '{print toupper($1)}');
gs130w_sha256=$(sha256sum __PROJECT_ROOT__/src/rdk_patrol/stereo/gs130w.py | awk '{print toupper($1)}');
if [ "$depth_sha256" != '__RUNTIME_DEPTH_SHA256__' ]; then echo "ERROR: deployed depth.py is not the approved version. actual=$depth_sha256" >&2; exit 65; fi;
if [ "$calibration_sha256" != '__RUNTIME_CALIBRATION_SHA256__' ]; then echo "ERROR: deployed calibration.py is not the approved version. actual=$calibration_sha256" >&2; exit 65; fi;
if [ "$gs130w_sha256" != '__RUNTIME_GS130W_SHA256__' ]; then echo "ERROR: deployed gs130w.py is not the approved version. actual=$gs130w_sha256" >&2; exit 65; fi;
echo RUNTIME_SOURCE_SHA256_OK;
sudo install -d -o __USER__ -g __USER__ -m 0750 __SESSION_PARENT__;
__SESSION_GUARD__;
check_fresh_frames CAMERA_FRAMES_BEFORE_STOP_OK;
trap cleanup_service EXIT;
trap 'exit 130' INT TERM HUP;
sudo -v;
sudo systemctl stop rdk-patrol.service;
(while true; do sleep 30; sudo -n true >/dev/null 2>&1 || exit 1; done) &
sudo_keepalive_pid=$!;
echo SUDO_KEEPALIVE_STARTED;
check_fresh_frames CAMERA_FRAMES_AFTER_STOP_OK;
./.venv/bin/python __REMOTE_TOOL__ --candidate-yaml __CANDIDATE__ --session-dir __SESSION__ --validated-output __VALIDATED__ --topic __TOPIC__ --host 127.0.0.1 --port __WEB_PORT__ --project-root __PROJECT_ROOT__ __RESUME__;
'@
    $remoteCommand = $remoteTemplate
    $remoteCommand = $remoteCommand.Replace("__PROJECT_ROOT__", $remoteProjectRoot)
    $remoteCommand = $remoteCommand.Replace("__TOPIC__", $Topic)
    $remoteCommand = $remoteCommand.Replace("__REMOTE_TEMP__", $remoteTemporary)
    $remoteCommand = $remoteCommand.Replace("__REMOTE_TOOL__", $remoteTool)
    $remoteCommand = $remoteCommand.Replace("__CANDIDATE__", $CandidateYaml)
    $remoteCommand = $remoteCommand.Replace("__CANDIDATE_SHA256__", $CandidateSha256.ToUpperInvariant())
    $remoteCommand = $remoteCommand.Replace("__RUNTIME_DEPTH_SHA256__", $RuntimeDepthSha256.ToUpperInvariant())
    $remoteCommand = $remoteCommand.Replace("__RUNTIME_CALIBRATION_SHA256__", $RuntimeCalibrationSha256.ToUpperInvariant())
    $remoteCommand = $remoteCommand.Replace("__RUNTIME_GS130W_SHA256__", $RuntimeGs130wSha256.ToUpperInvariant())
    $remoteCommand = $remoteCommand.Replace("__USER__", $User)
    $remoteCommand = $remoteCommand.Replace("__SESSION_PARENT__", $remoteSessionParent)
    $remoteCommand = $remoteCommand.Replace("__SESSION_GUARD__", $sessionGuard)
    $remoteCommand = $remoteCommand.Replace("__SESSION__", $remoteSession)
    $remoteCommand = $remoteCommand.Replace("__VALIDATED__", $remoteValidated)
    $remoteCommand = $remoteCommand.Replace("__WEB_PORT__", "$LocalWebPort")
    $remoteCommand = $remoteCommand.Replace("__RESUME__", $resumeArgument)
    $remoteCommand = (($remoteCommand -replace '(\r?\n)+', ' ').Trim())

    $sshArgs = @(
        "-t",
        "-p", "$SshPort",
        "-o", "ConnectTimeout=10",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-L", "${LocalWebPort}:127.0.0.1:${LocalWebPort}",
        $target,
        $remoteCommand
    )
    $fullValidationLaunched = $true
    & ssh.exe @sshArgs
    $validationExit = $LASTEXITCODE
    }
}
catch {
    Write-Host ""
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    $validationExit = 1
}
finally {
    if ($null -ne $browserProcess) {
        try {
            if (-not $browserProcess.HasExited) {
                Stop-Process -Id $browserProcess.Id -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            # The browser helper may have already exited normally.
        }
    }

    if ($completedResumeFastPath) {
        Write-Host ""
        Write-Host "Completed-resume fast path: rdk-patrol.service was not stopped or changed."
        $serviceExit = 0
        if ($resumeProbeServiceActive) {
            Write-Host "RDK_PATROL_ACTIVE" -ForegroundColor Green
        }
        else {
            Write-Host "RDK_PATROL_STATUS_UNCHANGED_NOT_ACTIVE" -ForegroundColor Yellow
        }
    }
    elseif (-not $fullValidationLaunched) {
        Write-Host ""
        Write-Host "Validation SSH was not launched; rdk-patrol.service was not stopped or changed."
        $serviceExit = 0
        Write-Host "RDK_PATROL_SERVICE_NOT_TOUCHED"
    }
    else {
        Write-Host ""
        Write-Host "Restoring and verifying rdk-patrol.service."
        $restoreCommand = (
            "sudo systemctl reset-failed rdk-patrol.service && " +
            "sudo systemctl start rdk-patrol.service && " +
            "sleep 3 && " +
            "systemctl is-active --quiet rdk-patrol.service"
        )
    if ($validationExit -eq 0 -or $validationExit -eq 130) {
        # The board-side EXIT trap already ran reset-failed/start, waited three
        # seconds, and returned these codes only after is-active succeeded.
        $serviceExit = 0
        Write-Host "RDK_PATROL_ACTIVE" -ForegroundColor Green
    }
    else {
        Write-Host "A new recovery SSH connection is required; enter the board password again if prompted." -ForegroundColor Yellow
        for ($attempt = 1; $attempt -le $RestoreAttempts; $attempt++) {
            Write-Host "Service restore verification attempt $attempt/$RestoreAttempts."
            try {
                $restoreArgs = @(
                    "-t",
                    "-p", "$SshPort",
                    "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=5",
                    "-o", "ServerAliveCountMax=3",
                    $target,
                    $restoreCommand
                )
                & ssh.exe @restoreArgs
                $serviceExit = $LASTEXITCODE
            }
            catch {
                Write-Host ("SERVICE RESTORE ERROR: " + $_.Exception.Message) -ForegroundColor Red
                $serviceExit = 1
            }
            if ($serviceExit -eq 0) {
                Write-Host "RDK_PATROL_ACTIVE" -ForegroundColor Green
                break
            }
            if ($attempt -lt $RestoreAttempts) {
                Write-Host "Restore was not confirmed; retrying in 3 seconds." -ForegroundColor Yellow
                Start-Sleep -Seconds 3
            }
        }
    }
    }

}

if ($serviceExit -ne 0) {
    Write-Host ""
    Write-Host "RDK service was not confirmed active. Keep this window and send a screenshot." -ForegroundColor Red
    Stop-TranscriptSafely
    exit 1
}

if ($validationExit -eq 0) {
    try {
        Write-Host ""
        Write-Host "Checking the downloaded audit package."
        if (Test-Path -LiteralPath $localSession) {
            throw "Refusing to overwrite existing local session: $localSession"
        }
        $downloadedRoot = $null
        $auditTransport = ""
        $httpAuditRejected = $false
        if (Test-Path -LiteralPath $auditZipPath -PathType Leaf) {
            Write-Host "Audit ZIP was downloaded through the SSH web tunnel; no extra SCP password is needed."
            try {
                Expand-AuditZipSafely -ZipPath $auditZipPath -Destination $incomingPath
                Assert-DownloadedSession -SessionPath $incomingPath
                $downloadedRoot = $incomingPath
                $auditTransport = "http"
            }
            catch {
                $httpAuditRejected = $true
                Write-Host "HTTP_AUDIT_REJECTED: $($_.Exception.Message)" -ForegroundColor Yellow
                Write-Host "Bad HTTP audit evidence was preserved at: $incomingPath" -ForegroundColor Yellow
                Write-Host "A clean SCP staging directory will be used; the rejected ZIP will not be reused." -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "Web audit download was not available." -ForegroundColor Yellow
        }

        if ($null -eq $downloadedRoot) {
            if ($httpAuditRejected) {
                Write-Host "Falling back to SCP after rejecting the HTTP audit package." -ForegroundColor Yellow
            }
            else {
                Write-Host "Falling back to SCP." -ForegroundColor Yellow
            }
            Write-Host "SCP uses a new SSH connection; enter the board password again if prompted." -ForegroundColor Yellow
            $fallbackName = ".{0}.scp_{1}_{2}" -f $SessionId, (Get-Date -Format "yyyyMMdd_HHmmss"), $PID
            $fallbackPath = Join-Path $LocalSessionsRoot $fallbackName
            if (-not (Test-Path -LiteralPath $LocalSessionsRoot -PathType Container)) {
                New-Item -ItemType Directory -Path $LocalSessionsRoot -Force | Out-Null
            }
            if (Test-Path -LiteralPath $fallbackPath) {
                throw "Refusing to overwrite existing SCP staging directory: $fallbackPath"
            }
            New-Item -ItemType Directory -Path $fallbackPath | Out-Null
            $pullArgs = @(
                "-r",
                "-P", "$SshPort",
                "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=3",
                "${target}:${remoteSession}/.",
                $fallbackPath
            )
            & scp.exe @pullArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Result download failed with exit code $LASTEXITCODE. Remote session was retained."
            }
            $downloadedRoot = $fallbackPath
            if (-not (Test-Path -LiteralPath (Join-Path $downloadedRoot "session.json") -PathType Leaf)) {
                $nested = Join-Path $fallbackPath $SessionId
                if (Test-Path -LiteralPath (Join-Path $nested "session.json") -PathType Leaf) {
                    $downloadedRoot = $nested
                }
            }
            Assert-DownloadedSession -SessionPath $downloadedRoot
            $auditTransport = "scp"
            Write-Host "SCP_AUDIT_FALLBACK_OK"
        }

        # Recheck immediately before publication, regardless of transport.
        Assert-DownloadedSession -SessionPath $downloadedRoot
        $bundlePath = Join-Path $downloadedRoot "audit.zip"
        if ($auditTransport -eq "scp") {
            # Never promote a server-side bundle after an HTTP bundle failed.
            # Build a fresh ZIP from only the files covered by the verified
            # checksum manifest plus session.json/audit.jsonl.
            if (Test-Path -LiteralPath $bundlePath) {
                $untrustedName = "untrusted_remote_audit_{0}.zip" -f (Get-Date -Format "yyyyMMdd_HHmmss")
                $untrustedPath = Join-Path $downloadedRoot $untrustedName
                if (Test-Path -LiteralPath $untrustedPath) {
                    throw "Refusing to overwrite untrusted remote audit evidence: $untrustedPath"
                }
                Move-Item -LiteralPath $bundlePath -Destination $untrustedPath
            }
            $serverBundle = Join-Path $downloadedRoot "GS130W_distance_validation_audit.zip"
            if (Test-Path -LiteralPath $serverBundle -PathType Leaf) {
                $untrustedServerName = "untrusted_server_bundle_{0}.zip" -f (Get-Date -Format "yyyyMMdd_HHmmss")
                $untrustedServerPath = Join-Path $downloadedRoot $untrustedServerName
                if (Test-Path -LiteralPath $untrustedServerPath) {
                    throw "Refusing to overwrite server bundle evidence: $untrustedServerPath"
                }
                Move-Item -LiteralPath $serverBundle -Destination $untrustedServerPath
            }
            New-VerifiedLocalAuditZip -SessionPath $downloadedRoot -DestinationZip $bundlePath
        }
        elseif (-not (Test-Path -LiteralPath $bundlePath -PathType Leaf)) {
            throw "Verified HTTP audit ZIP disappeared before publication."
        }
        $bundleHash = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToUpperInvariant()
        $hashRecord = "$bundleHash  audit.zip`r`n"
        $hashRecordPath = Join-Path $downloadedRoot "audit.zip.sha256"
        if (Test-Path -LiteralPath $hashRecordPath) {
            throw "Refusing to overwrite existing audit ZIP hash record: $hashRecordPath"
        }
        [IO.File]::WriteAllText(
            $hashRecordPath,
            $hashRecord,
            (New-Object Text.UTF8Encoding($false))
        )
        if (Test-Path -LiteralPath $localSession) {
            throw "Refusing to overwrite local session created during audit: $localSession"
        }
        Move-Item -LiteralPath $downloadedRoot -Destination $localSession
        $localAuditExit = 0
        Write-Host "VALIDATED_YAML_READY: $(Join-Path $localSession 'stereo_gs130w_validated.yaml')" -ForegroundColor Green
        Write-Host "AUDIT_ZIP_SHA256: $bundleHash"
        Write-Host "LOCAL_AUDIT_OK" -ForegroundColor Green
    }
    catch {
        Write-Host ""
        Write-Host ("LOCAL AUDIT ERROR: " + $_.Exception.Message) -ForegroundColor Red
        Write-Host "The board session was retained at: $remoteSession"
        $localAuditExit = 1
    }

    if ($localAuditExit -eq 0) {
        Write-Host ""
        Write-Host "DISTANCE_VALIDATION_COMPLETE" -ForegroundColor Green
        Write-Host "Local result: $localSession"
        Stop-TranscriptSafely
        exit 0
    }
    Stop-TranscriptSafely
    exit 1
}

if ($validationExit -eq 130) {
    Write-Host ""
    Write-Host "DISTANCE_VALIDATION_PAUSED" -ForegroundColor Yellow
    Write-Host "Resume with: -SessionId $SessionId -Resume"
    Write-Host "Remote session retained: $remoteSession"
    Stop-TranscriptSafely
    exit 0
}

Write-Host ""
Write-Host "Validation exited with code $validationExit. The production calibration was not changed." -ForegroundColor Red
Write-Host "Do not delete the board session directory: $remoteSession"
Stop-TranscriptSafely
exit 1
