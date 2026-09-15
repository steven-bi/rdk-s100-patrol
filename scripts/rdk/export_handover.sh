#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：
  scripts/rdk/export_handover.sh [--data-dir DIR] [--output FILE.tar.gz]

生成自包含离线报警台账、JSONL、配置快照、服务状态和校验文件。
本脚本不会确认、移动或删除任何报警，也不会导出 /etc/rdk-patrol.env。
EOF
}

prefix="${RDK_PATROL_PREFIX:-/opt/rdk-patrol}"
project_root="$prefix/current"
data_dir="${RDK_PATROL_DATA_DIR:-/var/lib/rdk-patrol}"
output=""

while (($#)); do
  case "$1" in
    --data-dir)
      data_dir="$2"
      shift 2
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: 未知参数：$1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ ! "$data_dir" =~ ^/[A-Za-z0-9._/-]+$ ]] || [[ ! -d "$data_dir" ]]; then
  echo "ERROR: 数据目录无效：$data_dir" >&2
  exit 64
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$output" ]]; then
  output="$data_dir/exports/rdk-patrol-handover-$timestamp.tar.gz"
fi
if [[ ! "$output" =~ ^/[A-Za-z0-9._/-]+[.]tar[.]gz$ ]]; then
  echo "ERROR: 输出必须是无空格的绝对 .tar.gz 路径。" >&2
  exit 64
fi
install -d -m 0750 "$(dirname -- "$output")"

stage="$(mktemp -d "$data_dir/exports/.handover.XXXXXX")"
cleanup() {
  case "$stage" in
    "$data_dir"/exports/.handover.*)
      rm -rf -- "$stage"
      ;;
  esac
}
trap cleanup EXIT

cli="$project_root/.venv/bin/rdk-patrol"
if [[ ! -x "$cli" ]]; then
  echo "ERROR: 找不到主程序 CLI：$cli" >&2
  exit 70
fi

"$cli" export-ledger \
  --data-dir "$data_dir" \
  --output "$stage/报警台账.html"

if [[ -r "$data_dir/alarms/records.jsonl" ]]; then
  cp -p "$data_dir/alarms/records.jsonl" "$stage/报警记录.jsonl"
fi
if [[ -r "$data_dir/state/health.json" ]]; then
  cp -p "$data_dir/state/health.json" "$stage/健康快照.json"
fi
install -d -m 0750 "$stage/现场配置"
for config_file in system.yaml points.yaml stereo_gs130w.yaml; do
  if [[ -r "$project_root/configs/$config_file" ]]; then
    cp -p "$project_root/configs/$config_file" "$stage/现场配置/$config_file"
  fi
done

{
  printf '导出时间（北京时间）：%s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %z')"
  printf '板端主机名：%s\n' "$(hostname)"
  printf '当前版本：%s\n' "$(readlink -f "$project_root" 2>/dev/null || echo "$project_root")"
  if [[ -r "$project_root/models/hbm/patrol_v2_yolo11s_static_640x320_rect_rgb_int8.hbm" ]]; then
    printf 'HBM模型SHA256：%s\n' "$(
      sha256sum "$project_root/models/hbm/patrol_v2_yolo11s_static_640x320_rect_rgb_int8.hbm" |
        awk '{print $1}'
    )"
  fi
  printf '数据目录：%s\n' "$data_dir"
  printf '说明：报警台账.html 为自包含文件，可在断网电脑上双击打开。\n'
  printf '说明：本次导出没有确认或删除任何板端报警。\n'
} > "$stage/导出说明.txt"

systemctl status rdk-patrol.service --no-pager > "$stage/服务状态.txt" 2>&1 || true
journalctl -u rdk-patrol.service --since '24 hours ago' --no-pager > "$stage/最近24小时日志.txt" 2>&1 || true

(
  cd "$stage"
  find . -type f ! -name SHA256SUMS -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS
)
tar -czf "$output" -C "$stage" .
chmod 0640 "$output"

printf 'EXPORT_PATH=%s\n' "$output"
printf '已生成拉回包；板端报警保持不变。\n'
