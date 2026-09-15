#!/usr/bin/env bash
set -Eeuo pipefail

prefix="${RDK_PATROL_PREFIX:-/opt/rdk-patrol}"
data_dir="${RDK_PATROL_DATA_DIR:-/var/lib/rdk-patrol}"
restart=1

if [[ "${1:-}" == "--no-restart" ]]; then
  restart=0
elif [[ $# -gt 0 ]]; then
  echo "用法：sudo $0 [--no-restart]" >&2
  exit 64
fi
if (( EUID != 0 )); then
  echo "ERROR: 请使用 sudo 运行。" >&2
  exit 77
fi
if [[ ! -d "$prefix/current/configs" || ! -d "$data_dir/site-config" ]]; then
  echo "ERROR: 未找到当前版本或现场配置目录。" >&2
  exit 66
fi

for site_file in system.yaml stereo_gs130w.yaml points.yaml; do
  source_path="$data_dir/site-config/$site_file"
  if [[ ! -s "$source_path" ]]; then
    echo "ERROR: 现场配置不存在或为空：$source_path" >&2
    exit 66
  fi
  install -m 0644 "$source_path" "$prefix/current/configs/$site_file"
  echo "已应用：$site_file"
done

if (( restart == 1 )); then
  systemctl restart rdk-patrol.service
  echo "已重启 rdk-patrol.service"
else
  echo "未重启服务；请在合适时间手动重启。"
fi
