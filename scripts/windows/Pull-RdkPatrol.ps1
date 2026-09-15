[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$RemotePrefix = "/opt/rdk-patrol",
    [string]$RemoteDataDir = "/var/lib/rdk-patrol",
    [string]$DestinationRoot = "",
    [switch]$NoOpenLedger
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($User -notmatch '^[A-Za-z_][A-Za-z0-9_-]*$' -or $BoardHost -notmatch '^[A-Za-z0-9.:-]+$') {
    throw "SSH 用户名或板端地址包含不安全字符。"
}
foreach ($path in @($RemotePrefix, $RemoteDataDir)) {
    if ($path -notmatch '^/[A-Za-z0-9._/-]+$' -or $path -in @("/", "/opt", "/var")) {
        throw "远程程序/数据目录必须是安全绝对路径：$path"
    }
}
foreach ($command in @("ssh.exe", "scp.exe", "tar.exe")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "找不到 $command。请安装 Windows OpenSSH Client。"
    }
}

if ([string]::IsNullOrWhiteSpace($DestinationRoot)) {
    $DestinationRoot = Join-Path ([Environment]::GetFolderPath("Desktop")) "RDK巡检拉回"
}
$DestinationRoot = [System.IO.Path]::GetFullPath($DestinationRoot)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$destination = Join-Path $DestinationRoot "拉回_$stamp"
New-Item -ItemType Directory -Path $destination -Force | Out-Null

$target = "$User@$BoardHost"
$sshArgs = @("-p", "$SshPort", "-o", "ConnectTimeout=10")
$scpArgs = @("-P", "$SshPort", "-o", "ConnectTimeout=10")
$remoteArchive = "$RemoteDataDir/exports/rdk-patrol-handover-$stamp.tar.gz"
$localArchive = Join-Path $destination "rdk-patrol-handover-$stamp.tar.gz"

Write-Host "1/3 在板端生成只读拉回包"
$remoteExport = "'$RemotePrefix/current/scripts/rdk/export_handover.sh' --data-dir '$RemoteDataDir' --output '$remoteArchive'"
& ssh.exe @sshArgs $target $remoteExport
if ($LASTEXITCODE -ne 0) {
    throw "板端导出失败。"
}

Write-Host "2/3 下载拉回包"
& scp.exe @scpArgs "${target}:$remoteArchive" $localArchive
if ($LASTEXITCODE -ne 0) {
    throw "下载拉回包失败。板端导出文件仍保留：$remoteArchive"
}

Write-Host "3/3 解压并校验"
& tar.exe -xzf $localArchive -C $destination
if ($LASTEXITCODE -ne 0) {
    throw "拉回包已下载，但解压失败：$localArchive"
}

$ledger = Join-Path $destination "报警台账.html"
if (-not (Test-Path -LiteralPath $ledger -PathType Leaf)) {
    throw "拉回包中缺少 报警台账.html。"
}
$sumFile = Join-Path $destination "SHA256SUMS"
if (-not (Test-Path -LiteralPath $sumFile -PathType Leaf)) {
    throw "拉回包中缺少 SHA256SUMS。"
}
$destinationPrefix = $destination.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
    [System.IO.Path]::DirectorySeparatorChar
foreach ($line in Get-Content -LiteralPath $sumFile -Encoding UTF8) {
    if ($line -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
        throw "SHA256SUMS 行格式无效：$line"
    }
    $expectedHash = $Matches[1].ToUpperInvariant()
    $relativePath = $Matches[2] -replace '^[.][/\\]', ''
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $destination $relativePath))
    if (-not $candidate.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "校验清单包含目录外路径：$relativePath"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "校验文件缺失：$relativePath"
    }
    $actualHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash
    if ($actualHash -ne $expectedHash) {
        throw "SHA256 不匹配：$relativePath"
    }
}

Write-Host ""
Write-Host "拉回完成：$destination"
Write-Host "离线台账：$ledger"
Write-Host "板端报警和板端导出包均未删除。人工核对并确认后，才能按现场制度单独清理。"

if (-not $NoOpenLedger) {
    try {
        Start-Process $ledger
    }
    catch {
        Write-Warning "台账已成功拉回，但无法自动打开：$ledger"
    }
}
