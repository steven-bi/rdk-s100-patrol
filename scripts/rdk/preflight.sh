#!/usr/bin/env bash
set -u

prefix="${RDK_PATROL_PREFIX:-/opt/rdk-patrol}"
project_root="${RDK_PATROL_PROJECT_ROOT:-$prefix/current}"
data_dir="${RDK_PATROL_DATA_DIR:-/var/lib/rdk-patrol}"
config_path="${RDK_PATROL_CONFIG:-$project_root/configs/system.yaml}"
topic="${RDK_PATROL_CAMERA_TOPIC:-/image_combine_jpeg}"
web_port="${RDK_PATROL_WEB_PORT:-}"
failures=0
warnings=0

if [[ -z "$web_port" ]]; then
  web_port="$({
    python3 - "$config_path" <<'PY'
import sys

import yaml

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    document = yaml.safe_load(stream) or {}
print(int(document.get("web", {}).get("port", 8081)))
PY
  } 2>/dev/null || printf '8081')"
fi

load_service_environment() {
  local env_name line value
  [[ -r /etc/rdk-patrol.env ]] || return 0
  for env_name in MINIMAX_API_KEY ROS_DOMAIN_ID RMW_IMPLEMENTATION; do
    line="$(grep -m1 -E "^${env_name}=" /etc/rdk-patrol.env 2>/dev/null || true)"
    [[ -n "$line" ]] || continue
    value="${line#*=}"
    value="${value%$'\r'}"
    export "$env_name=$value"
  done
}

web_health() {
  local url="http://127.0.0.1:${web_port}/health"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "$url" >/dev/null 2>&1
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' "$url" >/dev/null 2>&1
    return
  fi
  return 127
}

ok() {
  printf '[OK]   %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
  warnings=$((warnings + 1))
}

fail() {
  printf '[FAIL] %s\n' "$1"
  failures=$((failures + 1))
}

source_ros_environment() {
  local candidate
  for candidate in \
    /opt/tros/humble/setup.bash \
    /opt/tros/setup.bash \
    /opt/ros/humble/setup.bash; do
    if [[ -r "$candidate" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "$candidate"
      set -u
      ok "ROS 2 环境：$candidate"
      return 0
    fi
  done
  fail "未找到 ROS 2 环境脚本"
  return 1
}

printf 'RDK S100 巡检系统预检\n'
printf '项目：%s\n数据：%s\n配置：%s\n\n' "$project_root" "$data_dir" "$config_path"
load_service_environment

if [[ "$(uname -m 2>/dev/null)" == "aarch64" ]]; then
  ok "CPU 架构为 aarch64"
else
  warn "当前架构为 $(uname -m 2>/dev/null || echo unknown)，生产板应为 aarch64"
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    ok "Python $(python3 -c 'import platform; print(platform.python_version())')"
  else
    fail "Python 必须不低于 3.9"
  fi
else
  fail "找不到 python3"
fi

for command_name in tar sha256sum systemctl timeout; do
  if command -v "$command_name" >/dev/null 2>&1; then
    ok "命令可用：$command_name"
  else
    fail "缺少命令：$command_name"
  fi
done

if [[ -r "$config_path" ]]; then
  ok "主配置可读"
else
  fail "主配置不可读：$config_path"
fi

model_path="$project_root/models/hbm/patrol_v2_yolo11s_static_640x320_rect_rgb_int8.hbm"
metadata_path="$project_root/models/hbm/model_metadata.yaml"
if [[ -s "$model_path" && -s "$metadata_path" ]]; then
  ok "HBM 模型和元数据存在"
else
  fail "HBM 模型或元数据缺失"
fi

if [[ -d "$data_dir" && -w "$data_dir" ]]; then
  ok "数据目录可写"
  free_kb="$(df -Pk "$data_dir" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ "$free_kb" =~ ^[0-9]+$ ]]; then
    free_gb=$((free_kb / 1024 / 1024))
    if (( free_gb >= 300 )); then
      ok "数据盘可用空间约 ${free_gb} GB"
    else
      warn "数据盘仅约 ${free_gb} GB；双路 72 小时录像建议至少预留 300 GB，整盘建议 512 GB"
    fi
  fi
else
  fail "数据目录不存在或不可写：$data_dir"
fi

timezone_name="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
if [[ "$timezone_name" == "Asia/Shanghai" ]]; then
  ok "系统时区为 Asia/Shanghai"
else
  warn "系统时区为 ${timezone_name:-未知}；报警时间必须使用北京时间"
fi
if timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -qx yes; then
  ok "系统时钟已同步"
else
  warn "系统时钟尚未确认同步"
fi

if source_ros_environment; then
  camera_service_state="$(systemctl is-active rdk-gs130w-camera.service 2>/dev/null || true)"
  if [[ "$camera_service_state" == "active" ]]; then
    ok "GS130W 相机服务为 active（单实例托管）"
  else
    fail "GS130W 相机服务不是 active：${camera_service_state:-unknown}"
  fi
  if command -v ros2 >/dev/null 2>&1; then
    topic_list="$(timeout 8 ros2 topic list 2>/dev/null || true)"
    if grep -Fxq "$topic" <<<"$topic_list"; then
      ok "相机话题存在：$topic"
      topic_type="$(timeout 5 ros2 topic type "$topic" 2>/dev/null || true)"
      ok "相机消息类型：${topic_type:-未知}"
    else
      fail "找不到相机话题：$topic"
    fi
  else
    fail "ROS 2 环境中找不到 ros2 命令"
  fi
fi

runtime_python="$project_root/.venv/bin/python"
if [[ -x "$runtime_python" ]]; then
  if "$runtime_python" -c 'import rclpy; import sensor_msgs.msg' >/dev/null 2>&1; then
    ok "板端 Python 可导入 rclpy 和 sensor_msgs"
  else
    fail "板端 Python 无法导入 rclpy/sensor_msgs；检查 ROS 2 环境和虚拟环境"
  fi
  if "$runtime_python" -c 'import hbm_runtime' >/dev/null 2>&1; then
    ok "板端 Python 可导入 hbm_runtime"
  else
    fail "板端 Python 无法导入 hbm_runtime"
  fi
  if "$runtime_python" -c 'import cv2; assert hasattr(cv2, "aruco") and hasattr(cv2.aruco, "DICT_APRILTAG_36h11")' >/dev/null 2>&1; then
    ok "OpenCV 支持 AprilTag tag36h11"
  else
    fail "OpenCV 缺少 cv2.aruco AprilTag tag36h11 支持"
  fi
fi

stereo_path="$project_root/configs/stereo_gs130w.yaml"
if [[ -r "$stereo_path" ]] && grep -Eq '^[[:space:]]*valid:[[:space:]]*true([[:space:]]|$)' "$stereo_path"; then
  ok "GS130W 双目标定已标记有效"
else
  warn "GS130W 双目标定尚未标记有效；明火仍报警，但距离会显示“不可用”"
fi

points_path="$project_root/configs/points.yaml"
if [[ -r "$points_path" ]] && grep -Eq 'registration_required:[[:space:]]*true' "$points_path"; then
  warn "仍有 AprilTag 点位需要现场登记"
else
  ok "未发现待登记的 AprilTag 点位标记"
fi

if [[ -r /etc/rdk-patrol.env ]] && grep -Eq '^MINIMAX_API_KEY=.+$' /etc/rdk-patrol.env; then
  ok "MiniMax Key 已配置（内容未显示）"
else
  warn "MiniMax Key 未配置；垃圾桶图片会进入持久化重试队列"
fi

if web_health; then
  ok "只读报警网页健康检查通过"
else
  warn "本机 ${web_port} 端口的报警网页尚未就绪"
fi

cli="$project_root/.venv/bin/rdk-patrol"
if [[ -x "$cli" ]]; then
  ok "主程序 CLI 已安装"
  if "$cli" preflight --config "$config_path" --data-dir "$data_dir"; then
    ok "主程序深度预检通过"
  else
    fail "主程序深度预检未通过"
  fi
else
  fail "主程序 CLI 不存在：$cli"
fi

printf '\n预检完成：%d 个失败，%d 个警告。\n' "$failures" "$warnings"
if (( failures > 0 )); then
  exit 1
fi
exit 0
