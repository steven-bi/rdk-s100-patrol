[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^http://127\.0\.0\.1:[0-9]+/$')]
    [string]$Url,

    [ValidateRange(30, 7200)]
    [int]$TimeoutSeconds = 3600,

    [Parameter(Mandatory = $true)]
    [string]$AuditZipPath,

    [switch]$NoOpenBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$healthUrl = $Url.TrimEnd('/') + "/healthz"
$statusUrl = $Url.TrimEnd('/') + "/api/status"
$finishUrl = $Url.TrimEnd('/') + "/api/finish"
$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
$browserOpened = $false

function Save-AuditZip {
    if (Test-Path -LiteralPath $AuditZipPath -PathType Leaf) {
        if ((Get-Item -LiteralPath $AuditZipPath).Length -gt 0) {
            return
        }
        throw "Existing audit ZIP is empty."
    }

    $parent = Split-Path -Parent $AuditZipPath
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $partial = "$AuditZipPath.part.$PID"
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    try {
        Invoke-WebRequest `
            -Uri ($Url.TrimEnd('/') + "/download/audit.zip") `
            -UseBasicParsing `
            -TimeoutSec 300 `
            -OutFile $partial `
            -ErrorAction Stop
        if (-not (Test-Path -LiteralPath $partial -PathType Leaf) -or
            (Get-Item -LiteralPath $partial).Length -le 0) {
            throw "Downloaded audit ZIP is empty."
        }
        Move-Item -LiteralPath $partial -Destination $AuditZipPath
    }
    catch {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        }
        throw
    }
}

while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $response = Invoke-WebRequest `
            -Uri $healthUrl `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -ErrorAction Stop
        if ([int]$response.StatusCode -eq 200) {
            if (-not $browserOpened) {
                if (-not $NoOpenBrowser) {
                    Start-Process -FilePath $Url | Out-Null
                }
                $browserOpened = $true
            }

            $statusResponse = Invoke-WebRequest `
                -Uri $statusUrl `
                -UseBasicParsing `
                -TimeoutSec 2 `
                -ErrorAction Stop
            $status = $statusResponse.Content | ConvertFrom-Json
            if ([string]$status.status -eq "completed") {
                # Download the complete immutable audit package through the
                # existing SSH tunnel before asking the board process to exit.
                Save-AuditZip
                Start-Sleep -Seconds 5
                $finishResponse = Invoke-WebRequest `
                    -Uri $finishUrl `
                    -Method Post `
                    -ContentType "application/json; charset=utf-8" `
                    -Body "{}" `
                    -UseBasicParsing `
                    -TimeoutSec 3 `
                    -ErrorAction Stop
                if ([int]$finishResponse.StatusCode -in @(200, 202)) {
                    exit 0
                }
                throw "The safe-finish request was not accepted."
            }
            if ([string]$status.status -eq "paused") {
                exit 0
            }
        }
    }
    catch {
        # The SSH tunnel or the board-side web process is not ready yet.
    }
    Start-Sleep -Milliseconds 750
}

exit 1
