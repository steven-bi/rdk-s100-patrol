#!/usr/bin/env bash
set -Eeuo pipefail

source_ros_environment() {
  local candidate
  for candidate in \
    /opt/tros/humble/setup.bash \
    /opt/tros/setup.bash \
    /opt/ros/humble/setup.bash; do
    if [[ -r "$candidate" ]]; then
      # ROS 2 setup scripts may reference variables that are initially unset.
      set +u
      # shellcheck disable=SC1090
      source "$candidate"
      set -u
      return 0
    fi
  done
  echo "ERROR: 找不到 ROS 2 环境脚本（/opt/tros 或 /opt/ros）。" >&2
  return 1
}

prefix="${RDK_PATROL_PREFIX:-/opt/rdk-patrol}"
project_root="$prefix/current"
data_dir="${RDK_PATROL_DATA_DIR:-/var/lib/rdk-patrol}"
config_path="${RDK_PATROL_CONFIG:-$project_root/configs/system.yaml}"
cli="$project_root/.venv/bin/rdk-patrol"

if [[ ! -x "$cli" ]]; then
  echo "ERROR: 主程序入口不存在：$cli" >&2
  exit 70
fi
if [[ ! -r "$config_path" ]]; then
  echo "ERROR: 配置文件不存在：$config_path" >&2
  exit 78
fi
if [[ ! -d "$data_dir" || ! -w "$data_dir" ]]; then
  echo "ERROR: 数据目录不存在或不可写：$data_dir" >&2
  exit 73
fi

source_ros_environment
exec "$cli" run --config "$config_path" --data-dir "$data_dir"
