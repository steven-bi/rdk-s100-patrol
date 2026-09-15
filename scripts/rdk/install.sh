#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：
  sudo bash scripts/rdk/install.sh [选项]

选项：
  --source DIR          待部署项目目录（默认：脚本上两级目录）
  --prefix DIR          程序目录（默认：/opt/rdk-patrol）
  --data-dir DIR        数据目录（默认：/var/lib/rdk-patrol）
  --service-user USER   systemd 运行用户（默认：sunrise）
  --wheelhouse DIR      可选离线 wheel 目录
  --no-start            安装并启用服务，但不立即启动
  -h, --help            显示帮助

安装采用 release + current 符号链接，不删除旧版本，也不删除报警。
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_dir="$(cd -- "$script_dir/../.." && pwd -P)"
prefix="/opt/rdk-patrol"
data_dir="/var/lib/rdk-patrol"
service_user="sunrise"
wheelhouse=""
start_service=1

while (($#)); do
  case "$1" in
    --source)
      source_dir="$2"
      shift 2
      ;;
    --prefix)
      prefix="$2"
      shift 2
      ;;
    --data-dir)
      data_dir="$2"
      shift 2
      ;;
    --service-user)
      service_user="$2"
      shift 2
      ;;
    --wheelhouse)
      wheelhouse="$2"
      shift 2
      ;;
    --no-start)
      start_service=0
      shift
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

safe_application_path() {
  local candidate="$1"
  # Require at least two clean path components and an application-specific
  # marker.  This prevents a typo such as /etc, /usr or /opt from becoming the
  # target of install -d/chown.
  [[ "$candidate" =~ ^/[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)+$ ]] || return 1
  [[ "$candidate" =~ ([Rr][Dd][Kk]|[Pp][Aa][Tt][Rr][Oo][Ll]) ]] || return 1
  case "$candidate" in
    /var/lib/*)
      ;;
    /etc/*|/usr/*|/bin/*|/sbin/*|/lib/*|/lib32/*|/lib64/*|\
    /boot/*|/dev/*|/proc/*|/sys/*|/run/*|/root/*|/home/*|\
    /tmp/*|/var/*|/snap/*|/lost+found/*)
      return 1
      ;;
  esac
}

paths_overlap() {
  local first="$1"
  local second="$2"
  [[
    "$first" == "$second" ||
      "$first" == "$second/"* ||
      "$second" == "$first/"*
  ]]
}

if ! safe_application_path "$prefix" || ! safe_application_path "$data_dir"; then
  echo "ERROR: --prefix 和 --data-dir 必须是至少两级、名称含 rdk/patrol 的应用专用绝对目录，且不能位于系统关键目录。" >&2
  exit 64
fi
if ! command -v realpath >/dev/null 2>&1; then
  echo "ERROR: 缺少用于安全解析部署目录的 realpath。" >&2
  exit 69
fi
prefix="$(realpath -m -- "$prefix")"
data_dir="$(realpath -m -- "$data_dir")"
if ! safe_application_path "$prefix" || ! safe_application_path "$data_dir"; then
  echo "ERROR: 部署目录解析后落入非应用目录；请检查父目录符号链接。" >&2
  exit 64
fi
if paths_overlap "$prefix" "$data_dir"; then
  echo "ERROR: --prefix 与 --data-dir 不能相同，也不能互为父子目录。" >&2
  exit 64
fi
if (( EUID != 0 )); then
  echo "ERROR: 请使用 sudo 运行安装脚本。" >&2
  exit 77
fi
if [[ ! "$service_user" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] || ! id "$service_user" >/dev/null 2>&1; then
  echo "ERROR: 服务用户不存在或名称无效：$service_user" >&2
  exit 67
fi
source_dir="$(cd -- "$source_dir" && pwd -P)"
if [[ ! -r "$source_dir/pyproject.toml" || ! -s "$source_dir/models/hbm/patrol_v2_yolo11s_static_640x320_rect_rgb_int8.hbm" ]]; then
  echo "ERROR: 源目录不是完整交付包：$source_dir" >&2
  exit 66
fi

service_group="$(id -gn "$service_user")"
release_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
release_dir="$prefix/releases/$release_id"

install -d -m 0755 "$prefix" "$prefix/releases"
install -d -m 0755 "$release_dir"
cp -a "$source_dir/." "$release_dir/"
find "$release_dir" -type d -name __pycache__ -prune -exec rm -rf -- {} +
find "$release_dir" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

install -d -o "$service_user" -g "$service_group" -m 0750 \
  "$data_dir" \
  "$data_dir/alarms" \
  "$data_dir/exports" \
  "$data_dir/logs" \
  "$data_dir/logs/ros" \
  "$data_dir/outbox" \
  "$data_dir/recordings" \
  "$data_dir/recordings/raw" \
  "$data_dir/recordings/annotated" \
  "$data_dir/review_queue" \
  "$data_dir/site-config" \
  "$data_dir/state"

# 业务参数、现场标定和点位登记属于持久配置；升级时优先覆盖新
# release 中的占位配置。旧安装没有 system.yaml 时会从本 release 初始化。
for site_file in system.yaml stereo_gs130w.yaml points.yaml; do
  if [[ -s "$data_dir/site-config/$site_file" ]]; then
    install -m 0644 "$data_dir/site-config/$site_file" "$release_dir/configs/$site_file"
  else
    install -o "$service_user" -g "$service_group" -m 0640 \
      "$release_dir/configs/$site_file" "$data_dir/site-config/$site_file"
  fi
done

python3 -m venv --system-site-packages "$release_dir/.venv"
if [[ -n "$wheelhouse" ]]; then
  wheelhouse="$(cd -- "$wheelhouse" && pwd -P)"
  "$release_dir/.venv/bin/python" -m pip install \
    --no-index --find-links "$wheelhouse" -r "$release_dir/requirements.txt"
else
  if ! "$release_dir/.venv/bin/python" -c 'import cv2, numpy, PIL, yaml' >/dev/null 2>&1; then
    echo "ERROR: 板端缺少 cv2/numpy/Pillow/PyYAML。请提供 --wheelhouse，或先按部署文档安装板端依赖。" >&2
    exit 69
  fi
fi
if ! "$release_dir/.venv/bin/python" -c \
  'import cv2; assert hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_APRILTAG_36h11")' \
  >/dev/null 2>&1; then
  echo "ERROR: 板端 OpenCV 缺少 AprilTag tag36h11。请使用与 RDK aarch64/BPU 环境兼容的构建，禁止安装通用 x86 wheel。" >&2
  exit 69
fi
if ! "$release_dir/.venv/bin/python" -m pip install \
  --no-deps --no-build-isolation "$release_dir"; then
  echo "WARN: 板端 setuptools/wheel 无法完成离线构建，改用本地 .pth 启动方式。"
fi

# Some RDK system pip/setuptools combinations return success for a local
# install without creating the console-script entry point.  Always install a
# deterministic local-source path and wrapper, then verify it before switching
# the current symlink.
site_packages="$(
  "$release_dir/.venv/bin/python" -c \
    'import site; paths=site.getsitepackages(); print(paths[0])'
)"
printf '%s\n' "$release_dir/src" > "$site_packages/rdk_patrol_local.pth"
{
  printf '#!/usr/bin/env bash\n'
  printf 'exec "%s" -m rdk_patrol.cli "$@"\n' \
    "$release_dir/.venv/bin/python"
} > "$release_dir/.venv/bin/rdk-patrol"
chmod 0755 "$release_dir/.venv/bin/rdk-patrol"
if ! "$release_dir/.venv/bin/rdk-patrol" --version >/dev/null 2>&1; then
  echo "ERROR: 主程序入口创建后仍无法执行；拒绝切换 current。" >&2
  exit 70
fi

chmod 0755 \
  "$release_dir/scripts/rdk/install.sh" \
  "$release_dir/scripts/rdk/preflight.sh" \
  "$release_dir/scripts/rdk/run-service.sh" \
  "$release_dir/scripts/rdk/export_handover.sh" \
  "$release_dir/scripts/rdk/apply-site-config.sh"

unit_source="$release_dir/ops/systemd/rdk-patrol.service"
unit_tmp="$(mktemp)"
trap 'rm -f -- "$unit_tmp"' EXIT
sed \
  -e "s|/opt/rdk-patrol|$prefix|g" \
  -e "s|/var/lib/rdk-patrol|$data_dir|g" \
  -e "s|^User=sunrise$|User=$service_user|" \
  -e "s|^Group=sunrise$|Group=$service_group|" \
  "$unit_source" > "$unit_tmp"
install -m 0644 "$unit_tmp" /etc/systemd/system/rdk-patrol.service

if [[ ! -e /etc/rdk-patrol.env ]]; then
  install -o root -g "$service_group" -m 0640 \
    "$release_dir/ops/systemd/rdk-patrol.env.example" /etc/rdk-patrol.env
else
  chown root:"$service_group" /etc/rdk-patrol.env
  chmod 0640 /etc/rdk-patrol.env
  echo "保留现有 /etc/rdk-patrol.env（未覆盖 MiniMax Key）。"
fi

chown -R root:root "$release_dir"
tmp_link="$prefix/.current-$release_id"
ln -s "$release_dir" "$tmp_link"
mv -Tf "$tmp_link" "$prefix/current"

systemctl daemon-reload
systemctl enable rdk-patrol.service
if (( start_service == 1 )); then
  systemctl restart rdk-patrol.service
fi

printf '\n部署完成\n'
printf '  版本：%s\n' "$release_dir"
printf '  当前：%s/current\n' "$prefix"
printf '  数据：%s\n' "$data_dir"
printf '  服务：rdk-patrol.service\n'
printf '\n未删除任何旧版本、录像或报警。现场配置保存在 %s/site-config。\n' "$data_dir"
