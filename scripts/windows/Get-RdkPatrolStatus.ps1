[CmdletBinding()]
param(
    [string]$BoardHost = "192.168.66.65",
    [string]$User = "sunrise",
    [ValidateRange(1, 65535)]
    [int]$SshPort = 22,
    [string]$RemoteDataDir = "/var/lib/rdk-patrol",
    [ValidateRange(1, 65535)]
    [int]$WebPort = 8081,
    [switch]$OpenWeb
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($User -notmatch '^[A-Za-z_][A-Za-z0-9_-]*$' -or $BoardHost -notmatch '^[A-Za-z0-9.:-]+$') {
    throw "SSH 用户名或板端地址包含不安全字符。"
}
if ($RemoteDataDir -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteDataDir -in @("/", "/var")) {
    throw "远程数据目录必须是安全绝对路径。"
}
if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
    throw "找不到 Windows OpenSSH Client（ssh.exe）。"
}

$target = "$User@$BoardHost"
$sshArgs = @("-p", "$SshPort", "-o", "ConnectTimeout=10")
$remoteCommand = @"
echo "===== 服务状态 ====="
systemctl is-active rdk-patrol.service || true
systemctl status rdk-patrol.service --no-pager -n 25 || true
echo "===== 网页健康 ====="
curl -fsS --max-time 3 http://127.0.0.1:$WebPort/health || true
echo
echo "===== 完整链路健康快照 ====="
cat '$RemoteDataDir/state/health.json' 2>/dev/null || true
echo
echo "===== 存储空间 ====="
df -h '$RemoteDataDir' 2>/dev/null || true
echo "===== 最新日志 ====="
journalctl -u rdk-patrol.service -n 30 --no-pager || true
"@

& ssh.exe @sshArgs $target $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "无法取得板端状态。"
}

$webUrl = "http://${BoardHost}:$WebPort/"
Write-Host ""
Write-Host "只读报警网页：$webUrl"
if ($OpenWeb) {
    Start-Process $webUrl
}
