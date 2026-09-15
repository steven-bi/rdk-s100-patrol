[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$ProjectRoot = "",
    [string]$RemotePrefix = "/opt/rdk-patrol",
    [string]$RemoteDataDir = "/var/lib/rdk-patrol",
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "找不到 $Name。请在 Windows 可选功能中安装 OpenSSH Client，并确认系统 tar.exe 可用。"
    }
}

function Test-SafeRemoteApplicationPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path -notmatch '^/[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+$') {
        return $false
    }
    if ($Path -notmatch '(rdk|patrol)') {
        return $false
    }
    $parts = $Path.TrimStart('/').Split('/')
    switch -CaseSensitive ($parts[0]) {
        "var" {
            if ($parts.Count -lt 3 -or $parts[1] -cne "lib") {
                return $false
            }
        }
        "etc" { return $false }
        "usr" { return $false }
        "bin" { return $false }
        "sbin" { return $false }
        "lib" { return $false }
        "lib32" { return $false }
        "lib64" { return $false }
        "boot" { return $false }
        "dev" { return $false }
        "proc" { return $false }
        "sys" { return $false }
        "run" { return $false }
        "root" { return $false }
        "home" { return $false }
        "tmp" { return $false }
        "snap" { return $false }
        "lost+found" { return $false }
    }
    return $true
}

function Assert-SafeRemoteValue {
    if ($User -notmatch '^[A-Za-z_][A-Za-z0-9_-]*$') {
        throw "SSH 用户名包含不安全字符：$User"
    }
    if ($BoardHost -notmatch '^[A-Za-z0-9.:-]+$') {
        throw "板端地址包含不安全字符：$BoardHost"
    }
    foreach ($path in @($RemotePrefix, $RemoteDataDir)) {
        if (-not (Test-SafeRemoteApplicationPath -Path $path)) {
            throw "远程路径必须是至少两级、名称含 rdk/patrol 的应用专用绝对目录，且不能位于系统关键目录：$path"
        }
    }
    if (
        [string]::Equals($RemotePrefix, $RemoteDataDir, [System.StringComparison]::Ordinal) -or
        $RemotePrefix.StartsWith(
            $RemoteDataDir + "/",
            [System.StringComparison]::Ordinal
        ) -or
        $RemoteDataDir.StartsWith(
            $RemotePrefix + "/",
            [System.StringComparison]::Ordinal
        )
    ) {
        throw "远程程序目录和数据目录不能相同，也不能互为父子目录。"
    }
}

Assert-Command "ssh.exe"
Assert-Command "scp.exe"
Assert-Command "tar.exe"
Assert-SafeRemoteValue

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -PathType Leaf)) {
    throw "项目根目录无效：$ProjectRoot"
}
$modelPath = Join-Path $ProjectRoot "models\hbm\patrol_v2_yolo11s_static_640x320_rect_rgb_int8.hbm"
if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw "缺少 HBM 模型：$modelPath"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveName = "rdk-patrol-upload-$stamp.tar.gz"
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
$target = "$User@$BoardHost"
$sshArgs = @("-p", "$SshPort", "-o", "ConnectTimeout=10")
$sshTtyArgs = @("-t", "-p", "$SshPort", "-o", "ConnectTimeout=10")
$scpArgs = @("-P", "$SshPort", "-o", "ConnectTimeout=10")

try {
    Write-Host "1/4 检查 SSH 连接：$target"
    & ssh.exe @sshArgs $target "printf 'SSH_OK\n'"
    if ($LASTEXITCODE -ne 0) {
        throw "SSH 连接失败。请检查 IP、网线、用户名和密钥/密码。"
    }

    Write-Host "2/4 打包交付目录（包含 HBM 模型，不包含运行数据和缓存）"
    & tar.exe -czf $archivePath `
        --exclude=".git" `
        --exclude=".venv" `
        --exclude="runtime" `
        --exclude="__pycache__" `
        --exclude=".pytest_cache" `
        -C $ProjectRoot .
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $archivePath)) {
        throw "本地打包失败。"
    }

    Write-Host "3/4 上传并安装"
    & scp.exe @scpArgs $archivePath "${target}:/tmp/$archiveName"
    if ($LASTEXITCODE -ne 0) {
        throw "上传失败。"
    }

    $stage = "/tmp/rdk-patrol-stage-$stamp"
    $remoteInstall = @(
        "set -e",
        "mkdir -p '$stage'",
        "tar -xzf '/tmp/$archiveName' -C '$stage'",
        "sudo bash '$stage/scripts/rdk/install.sh' --source '$stage' --prefix '$RemotePrefix' --data-dir '$RemoteDataDir' --service-user '$User'"
    ) -join " && "
    & ssh.exe @sshTtyArgs $target $remoteInstall
    if ($LASTEXITCODE -ne 0) {
        throw "板端安装失败；上传包和暂存目录已保留，便于排查。"
    }

    Write-Host "4/4 板端预检"
    if (-not $SkipPreflight) {
        & ssh.exe @sshArgs $target "'$RemotePrefix/current/scripts/rdk/preflight.sh'"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "部署已完成，但预检存在失败项。修复后重新运行状态脚本。"
        }
    } else {
        Write-Warning "已按参数跳过预检。"
    }

    Write-Host ""
    Write-Host "部署完成。"
    Write-Host "实时报警网页：http://${BoardHost}:8081/"
    Write-Host "MiniMax Key、GS130W 标定和 AprilTag 登记仍需按 docs 目录现场完成。"
    Write-Host "板端上传包保留在 /tmp/$archiveName；脚本没有删除报警或旧版本。"
}
finally {
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
}
