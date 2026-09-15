#!/usr/bin/env python3
"""Manually collect GS130W stereo chessboard pairs from a local web page.

The browser only requests a capture.  The ROS/OpenCV main loop remains the
single writer and saves a pair only when both boards are detected, the board
has stayed still, the full outer board is inside both images, and the pose is
different enough from every previously saved pair.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from collect_gs130w_pairs import (
    decode_image_message,
    find_board,
    put_label,
    rotate,
    split_combined,
)


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GS130W 手动双目标定采集</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #f9fafb; }
    main { max-width: 1120px; margin: auto; padding: 18px; }
    h1 { margin: 0 0 12px; font-size: 25px; }
    .layout { display: grid; grid-template-columns: minmax(0,2fr) minmax(280px,1fr); gap: 16px; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 14px; }
    #preview { width: 100%; max-height: 78vh; object-fit: contain; background: #030712; border-radius: 8px; }
    .counter { font-size: 30px; font-weight: 750; margin-bottom: 10px; }
    .check { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #374151; }
    .ok { color: #4ade80; font-weight: 700; }
    .bad { color: #f87171; font-weight: 700; }
    #message { min-height: 50px; line-height: 1.5; color: #fcd34d; }
    #action { min-height: 45px; line-height: 1.5; color: #93c5fd; }
    button { width: 100%; padding: 15px; border: 0; border-radius: 10px; font-size: 20px; font-weight: 750; color: white; background: #16a34a; cursor: pointer; }
    button:disabled { background: #4b5563; cursor: not-allowed; color: #d1d5db; }
    button.stop { margin-top: 10px; padding: 10px; font-size: 16px; background: #b91c1c; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; text-align: center; margin-top: 8px; }
    td { border: 1px solid #4b5563; height: 42px; font-size: 18px; }
    .hint { color: #d1d5db; line-height: 1.55; font-size: 14px; }
    @media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>GS130W 手动双目标定采集</h1>
  <div class="layout">
    <section class="card"><img id="preview" alt="等待摄像头画面"></section>
    <aside class="card">
      <div class="counter"><span id="saved">0</span>/<span id="target">0</span></div>
      <div class="check"><span>物理左目棋盘</span><span id="left" class="bad">未识别</span></div>
      <div class="check"><span>物理右目棋盘</span><span id="right" class="bad">未识别</span></div>
      <div class="check"><span>棋盘完整/大小</span><span id="quality" class="bad">未通过</span></div>
      <div class="check"><span>保持稳定</span><span id="stable" class="bad">未通过</span></div>
      <div class="check"><span>姿态不同</span><span id="unique" class="bad">未通过</span></div>
      <p id="message">等待摄像头画面……</p>
      <button id="capture" disabled>条件未满足</button>
      <button id="stop" class="stop">安全结束/稍后继续</button>
      <p id="action"></p>
      <h2>已保存位置（九宫格）</h2>
      <table>
        <tr><td id="c0">0</td><td id="c1">0</td><td id="c2">0</td></tr>
        <tr><td id="c3">0</td><td id="c4">0</td><td id="c5">0</td></tr>
        <tr><td id="c6">0</td><td id="c7">0</td><td id="c8">0</td></tr>
      </table>
      <p class="hint">
        只有五项全部通过后按钮才会变绿。点击一次只保存一组。保存后必须把棋盘移出画面，
        再改变位置、大小、旋转或透视角度；不要移动摄像头。
      </p>
    </aside>
  </div>
</main>
<script>
const byId = (id) => document.getElementById(id);
let stopping = false;
let statusTimer = null;
let previewTimer = null;
const setCheck = (id, ok, yes, no) => {
  const node = byId(id);
  node.textContent = ok ? yes : no;
  node.className = ok ? "ok" : "bad";
};
function stopPolling(removePreview = true) {
  stopping = true;
  if (statusTimer !== null) {
    clearInterval(statusTimer);
    statusTimer = null;
  }
  if (previewTimer !== null) {
    clearInterval(previewTimer);
    previewTimer = null;
  }
  if (removePreview) byId("preview").removeAttribute("src");
}
async function updateStatus() {
  if (stopping) return;
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    const s = await response.json();
    byId("saved").textContent = s.saved;
    byId("target").textContent = s.target;
    setCheck("left", s.left_found, "已识别", "未识别");
    setCheck("right", s.right_found, "已识别", "未识别");
    setCheck("quality", s.quality_ok, "已通过", "未通过");
    setCheck("stable", s.stable, "已稳定", "稳定 " + Number(s.stable_seconds || 0).toFixed(1) + " 秒");
    setCheck("unique", s.pose_unique, "新姿态", "与旧姿态相似");
    byId("message").textContent = s.message || "";
    byId("action").textContent = s.last_action || "";
    const button = byId("capture");
    button.disabled = !s.ready || s.capture_pending || s.completed;
    button.textContent = s.completed ? "采集已完成" : (s.capture_pending ? "正在保存……" : (s.ready ? "采集本组" : "条件未满足"));
    (s.coverage || []).forEach((value, index) => { byId("c" + index).textContent = value; });
    if (s.completed) stopPolling(false);
  } catch (error) {
    byId("message").textContent = "网页暂时无法读取板端状态，请确认SSH采集窗口仍在运行。";
    byId("capture").disabled = true;
  }
}
async function requestCapture() {
  const button = byId("capture");
  button.disabled = true;
  try {
    const response = await fetch("/api/capture", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: "{}"
    });
    const result = await response.json();
    if (!response.ok) byId("action").textContent = result.message || "当前不能采集。";
  } catch (error) {
    byId("action").textContent = "采集请求发送失败。";
  }
  await updateStatus();
}
async function requestStop() {
  if (!confirm("确认安全结束本次采集？已经保存的完整图片会保留，可用 --resume 继续。")) return;
  byId("capture").disabled = true;
  byId("stop").disabled = true;
  stopPolling();
  try {
    const response = await fetch("/api/stop", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: "{}"
    });
    const result = await response.json();
    byId("action").textContent = result.message || "正在安全结束……";
  } catch (error) {
    byId("action").textContent = "停止请求发送失败，请在命令窗口按 Ctrl+C。";
  }
}
byId("capture").addEventListener("click", requestCapture);
byId("stop").addEventListener("click", requestStop);
statusTimer = setInterval(updateStatus, 400);
previewTimer = setInterval(() => {
  if (!stopping) byId("preview").src = "/preview.jpg?t=" + Date.now();
}, 500);
updateStatus();
</script>
</body>
</html>
"""


@dataclass(frozen=True)
class ViewPose:
    center_x: float
    center_y: float
    log_area: float
    angle_rad: float
    perspective_tb: float
    perspective_lr: float
    axis_skew: float
    outer_margin_px: float
    area_fraction: float


@dataclass(frozen=True)
class PairPose:
    left: ViewPose
    right: ViewPose


@dataclass(frozen=True)
class PoseThresholds:
    center: float = 0.06
    log_area: float = 0.16
    angle_rad: float = math.radians(8.0)
    perspective: float = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过SSH隧道网页手动采集GS130W同步双目标定图。"
    )
    parser.add_argument("--topic", default="/image_combine_jpeg")
    parser.add_argument("--layout", choices=("vertical",), default="vertical")
    parser.add_argument(
        "--rotation",
        choices=("none", "cw90", "ccw90", "rotate180"),
        default="ccw90",
    )
    parser.add_argument(
        "--physical-left", choices=("top", "bottom"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board-cols", type=int, default=9)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stable-seconds", type=float, default=1.2)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--max-motion-px", type=float, default=3.5)
    parser.add_argument("--max-frame-age-seconds", type=float, default=2.0)
    parser.add_argument("--capture-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--min-area-fraction", type=float, default=0.006)
    parser.add_argument("--min-outer-margin-px", type=float, default=16.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="严格检查并继续目标目录中已有的连续完整图像对。",
    )
    return parser.parse_args()


class FreshFrameSubscriber:
    """ROS subscriber that exposes callback sequence and receive time."""

    def __init__(self, topic: str, wait_seconds: float) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
        except ImportError as exc:
            raise RuntimeError(
                "缺少 rclpy；请先 source 板端 ROS 2 环境。"
            ) from exc

        self.rclpy = rclpy
        rclpy.init(args=None)
        self.node = Node("gs130w_manual_pair_collector")
        self.frame: Optional[np.ndarray] = None
        self.sequence = 0
        self.received_at = 0.0
        self.error: Optional[str] = None

        deadline = time.monotonic() + wait_seconds
        topic_type = ""
        while time.monotonic() < deadline and not topic_type:
            for name, types in self.node.get_topic_names_and_types():
                if name == topic and types:
                    topic_type = types[0]
                    break
            if not topic_type:
                rclpy.spin_once(self.node, timeout_sec=0.2)
        if not topic_type:
            self.close()
            raise RuntimeError(f"{wait_seconds:g} 秒内未发现 ROS 2 话题：{topic}")

        if topic_type == "sensor_msgs/msg/CompressedImage":
            from sensor_msgs.msg import CompressedImage as MessageType
        elif topic_type == "sensor_msgs/msg/Image":
            from sensor_msgs.msg import Image as MessageType
        else:
            self.close()
            raise RuntimeError(f"不支持的话题类型：{topic_type}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        def callback(message: object) -> None:
            callback_received_at = time.monotonic()
            try:
                decoded = decode_image_message(message)
                self.frame = decoded
                self.sequence += 1
                self.received_at = callback_received_at
                self.error = None
            except Exception as exc:
                self.error = str(exc)

        self.subscription = self.node.create_subscription(
            MessageType, topic, callback, qos
        )
        print(f"已订阅 {topic}（{topic_type}）")

    def spin_once(
        self,
    ) -> Tuple[Optional[np.ndarray], int, float]:
        self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.error:
            raise RuntimeError(self.error)
        return self.frame, int(self.sequence), float(self.received_at)

    def close(self) -> None:
        node = getattr(self, "node", None)
        if node is not None:
            node.destroy_node()
        rclpy = getattr(self, "rclpy", None)
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(math.log(max(numerator, 1e-9) / max(denominator, 1e-9)))


def _finite_json_float(value: float) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


def _line_length(points: np.ndarray) -> float:
    differences = np.diff(points.astype(np.float64), axis=0)
    return float(np.linalg.norm(differences, axis=1).sum())


def _outer_margin(
    corners: np.ndarray,
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int],
) -> float:
    cols, rows = pattern_size
    image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    object_points = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    homography, _mask = cv2.findHomography(object_points, image_points, method=0)
    if homography is None:
        return float("-inf")
    outside = np.asarray(
        [[-1.0, -1.0], [float(cols), -1.0], [float(cols), float(rows)], [-1.0, float(rows)]],
        dtype=np.float32,
    ).reshape(1, -1, 2)
    projected = cv2.perspectiveTransform(outside, homography).reshape(-1, 2)
    width, height = image_size
    margins: List[float] = []
    for x, y in projected:
        margins.extend([float(x), float(width - 1 - x), float(y), float(height - 1 - y)])
    return float(min(margins))


def _canonical_corner_order(
    corners: np.ndarray, pattern_size: Tuple[int, int]
) -> np.ndarray:
    """Choose one of the two valid 180-degree chessboard orderings."""

    cols, rows = pattern_size
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    grid = points.reshape(rows, cols, 2)
    horizontal = grid[:, -1, :].mean(axis=0) - grid[:, 0, :].mean(axis=0)
    dominant = horizontal[0] if abs(horizontal[0]) >= abs(horizontal[1]) else horizontal[1]
    if dominant < 0:
        points = points[::-1].copy()
    return points


def view_pose_signature(
    corners: np.ndarray,
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int],
) -> ViewPose:
    cols, rows = pattern_size
    ordered = _canonical_corner_order(corners, pattern_size)
    grid = ordered.reshape(rows, cols, 2)
    flat = grid.reshape(-1, 2)
    width, height = image_size
    center = flat.mean(axis=0)
    hull = cv2.convexHull(flat.astype(np.float32))
    area_fraction = float(cv2.contourArea(hull) / float(width * height))

    horizontal = grid[:, -1, :].mean(axis=0) - grid[:, 0, :].mean(axis=0)
    vertical = grid[-1, :, :].mean(axis=0) - grid[0, :, :].mean(axis=0)
    horizontal_norm = max(float(np.linalg.norm(horizontal)), 1e-9)
    vertical_norm = max(float(np.linalg.norm(vertical)), 1e-9)
    top_width = _line_length(grid[0, :, :])
    bottom_width = _line_length(grid[-1, :, :])
    left_height = _line_length(grid[:, 0, :])
    right_height = _line_length(grid[:, -1, :])

    return ViewPose(
        center_x=float(center[0] / width),
        center_y=float(center[1] / height),
        log_area=float(math.log(max(area_fraction, 1e-12))),
        angle_rad=float(math.atan2(horizontal[1], horizontal[0])),
        perspective_tb=_safe_log_ratio(top_width, bottom_width),
        perspective_lr=_safe_log_ratio(left_height, right_height),
        axis_skew=float(np.dot(horizontal, vertical) / (horizontal_norm * vertical_norm)),
        outer_margin_px=_outer_margin(corners, image_size, pattern_size),
        area_fraction=area_fraction,
    )


def pair_pose_signature(
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int],
) -> PairPose:
    return PairPose(
        left=view_pose_signature(left_corners, image_size, pattern_size),
        right=view_pose_signature(right_corners, image_size, pattern_size),
    )


def _periodic_angle_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi / 2.0) % math.pi - math.pi / 2.0)


def pose_difference(
    current: PairPose,
    previous: PairPose,
    thresholds: PoseThresholds,
) -> Tuple[float, Dict[str, float]]:
    centers = []
    scales = []
    angles = []
    perspectives = []
    for current_view, previous_view in (
        (current.left, previous.left),
        (current.right, previous.right),
    ):
        centers.append(
            math.hypot(
                current_view.center_x - previous_view.center_x,
                current_view.center_y - previous_view.center_y,
            )
        )
        scales.append(abs(current_view.log_area - previous_view.log_area))
        angles.append(
            _periodic_angle_distance(current_view.angle_rad, previous_view.angle_rad)
        )
        perspectives.append(
            max(
                abs(current_view.perspective_tb - previous_view.perspective_tb),
                abs(current_view.perspective_lr - previous_view.perspective_lr),
                abs(current_view.axis_skew - previous_view.axis_skew),
            )
        )
    metrics = {
        "center": float(max(centers)),
        "log_area": float(max(scales)),
        "angle_rad": float(max(angles)),
        "perspective": float(max(perspectives)),
    }
    score = max(
        metrics["center"] / thresholds.center,
        metrics["log_area"] / thresholds.log_area,
        metrics["angle_rad"] / thresholds.angle_rad,
        metrics["perspective"] / thresholds.perspective,
    )
    return float(score), metrics


def nearest_saved_pose(
    current: PairPose,
    saved_poses: Sequence[PairPose],
    thresholds: PoseThresholds,
) -> Tuple[bool, Optional[int], float]:
    if not saved_poses:
        return True, None, float("inf")
    scored = [
        (pose_difference(current, previous, thresholds)[0], index + 1)
        for index, previous in enumerate(saved_poses)
    ]
    score, pair_number = min(scored)
    return bool(score >= 1.0), int(pair_number), float(score)


def coverage_grid(saved_poses: Sequence[PairPose]) -> List[int]:
    counts = [0] * 9
    for pose in saved_poses:
        center_x = (pose.left.center_x + pose.right.center_x) / 2.0
        center_y = (pose.left.center_y + pose.right.center_y) / 2.0
        column = min(2, max(0, int(center_x * 3.0)))
        row = min(2, max(0, int(center_y * 3.0)))
        counts[row * 3 + column] += 1
    return counts


def _aligned_corner_rms(current: np.ndarray, anchor: np.ndarray) -> float:
    current_points = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    anchor_points = np.asarray(anchor, dtype=np.float64).reshape(-1, 2)
    direct = float(
        np.sqrt(np.mean(np.sum((current_points - anchor_points) ** 2, axis=1)))
    )
    reversed_order = float(
        np.sqrt(
            np.mean(
                np.sum((current_points[::-1] - anchor_points) ** 2, axis=1)
            )
        )
    )
    return min(direct, reversed_order)


class StabilityTracker:
    def __init__(
        self, required_seconds: float, required_frames: int, max_motion_px: float
    ) -> None:
        self.required_seconds = float(required_seconds)
        self.required_frames = int(required_frames)
        self.max_motion_px = float(max_motion_px)
        self.reset()

    def reset(self) -> None:
        self.anchor_left: Optional[np.ndarray] = None
        self.anchor_right: Optional[np.ndarray] = None
        self.stable_since: Optional[float] = None
        self.stable_frames = 0

    def update(
        self,
        left_corners: Optional[np.ndarray],
        right_corners: Optional[np.ndarray],
        now: float,
    ) -> Tuple[bool, float, Optional[float], int]:
        if left_corners is None or right_corners is None:
            self.reset()
            return False, 0.0, None, 0
        if self.anchor_left is None or self.anchor_right is None:
            self.anchor_left = np.asarray(left_corners, dtype=np.float32).copy()
            self.anchor_right = np.asarray(right_corners, dtype=np.float32).copy()
            self.stable_since = float(now)
            self.stable_frames = 1
            stable = self.required_seconds <= 0 and self.required_frames <= 1
            return stable, 0.0, 0.0, self.stable_frames

        motion = max(
            _aligned_corner_rms(left_corners, self.anchor_left),
            _aligned_corner_rms(right_corners, self.anchor_right),
        )
        if motion > self.max_motion_px:
            self.anchor_left = np.asarray(left_corners, dtype=np.float32).copy()
            self.anchor_right = np.asarray(right_corners, dtype=np.float32).copy()
            self.stable_since = float(now)
            self.stable_frames = 1
            return False, 0.0, float(motion), self.stable_frames

        assert self.stable_since is not None
        self.stable_frames += 1
        seconds = max(0.0, float(now - self.stable_since))
        stable = (
            seconds >= self.required_seconds
            and self.stable_frames >= self.required_frames
        )
        return bool(stable), seconds, float(motion), self.stable_frames


class CaptureWebState:
    def __init__(self, target: int, capture_timeout_seconds: float = 2.0) -> None:
        self._lock = threading.Lock()
        self._preview: Optional[bytes] = None
        self._status: Dict[str, object] = {
            "saved": 0,
            "target": int(target),
            "left_found": False,
            "right_found": False,
            "quality_ok": False,
            "stable": False,
            "stable_seconds": 0.0,
            "pose_unique": False,
            "ready": False,
            "completed": False,
            "message": "等待摄像头画面……",
            "coverage": [0] * 9,
        }
        self._capture_requested = False
        self._capture_pending = False
        self._capture_requested_at: Optional[float] = None
        self._capture_after_frame = -1
        self._latest_received_frame = -1
        self._capture_claimed_frame: Optional[int] = None
        self._capture_timeout_seconds = float(capture_timeout_seconds)
        self._stop_requested = False
        self._last_action = ""

    def update(
        self, status: Dict[str, object], preview_jpeg: Optional[bytes] = None
    ) -> None:
        with self._lock:
            self._status.update(status)
            if preview_jpeg is not None:
                self._preview = bytes(preview_jpeg)

    def snapshot(self) -> Tuple[Dict[str, object], Optional[bytes]]:
        with self._lock:
            status = dict(self._status)
            status["capture_pending"] = self._capture_pending
            status["last_action"] = self._last_action
            return status, self._preview

    def request_capture(self) -> Tuple[bool, str]:
        with self._lock:
            if self._capture_pending:
                return False, "上一条采集请求仍在处理。"
            if not bool(self._status.get("ready")):
                return False, str(self._status.get("message") or "当前条件未满足。")
            self._capture_requested = True
            self._capture_pending = True
            self._capture_requested_at = time.monotonic()
            self._capture_claimed_frame = None
            self._capture_after_frame = max(
                int(self._status.get("frame_seq", -1)),
                self._latest_received_frame,
            )
            self._last_action = "已收到请求，正在用当前同步帧复核并保存……"
            return True, self._last_action

    def expire_capture_request(self, now: Optional[float] = None) -> bool:
        with self._lock:
            if (
                not self._capture_pending
                or self._capture_requested_at is None
                or self._capture_claimed_frame is not None
            ):
                return False
            current = time.monotonic() if now is None else float(now)
            if current - self._capture_requested_at <= self._capture_timeout_seconds:
                return False
            self._capture_requested = False
            self._capture_pending = False
            self._capture_requested_at = None
            self._last_action = "采集请求已超时，没有保存；请确认画面恢复后重新点击。"
            return True

    def consume_capture_request(self, frame_seq: int) -> bool:
        with self._lock:
            if self._stop_requested:
                return False
            if self._capture_claimed_frame != int(frame_seq):
                return False
            self._capture_claimed_frame = None
            return True

    def note_frame_received(
        self, frame_seq: int, received_at: Optional[float] = None
    ) -> None:
        with self._lock:
            frame_received_at = (
                time.monotonic()
                if received_at is None
                else float(received_at)
            )
            self._latest_received_frame = max(
                self._latest_received_frame, int(frame_seq)
            )
            if (
                self._capture_requested
                and self._capture_pending
                and self._capture_requested_at is not None
                and int(frame_seq) > self._capture_after_frame
                and frame_received_at > self._capture_requested_at
            ):
                age = frame_received_at - self._capture_requested_at
                if age <= self._capture_timeout_seconds:
                    self._capture_requested = False
                    self._capture_requested_at = None
                    self._capture_claimed_frame = int(frame_seq)
                    self._last_action = (
                        "已取得点击后的下一张新帧，正在复核角点和稳定性……"
                    )
                else:
                    self._capture_requested = False
                    self._capture_pending = False
                    self._capture_requested_at = None
                    self._capture_claimed_frame = None
                    self._last_action = (
                        "采集请求已超时，没有保存；请确认画面恢复后重新点击。"
                    )

    def finish_capture_request(self, message: str) -> None:
        with self._lock:
            self._capture_pending = False
            self._capture_requested_at = None
            self._capture_claimed_frame = None
            self._last_action = str(message)

    def request_stop(self) -> str:
        with self._lock:
            self._stop_requested = True
            self._capture_requested = False
            self._capture_pending = False
            self._capture_requested_at = None
            self._capture_claimed_frame = None
            self._status["ready"] = False
            self._last_action = "已请求安全结束，正在关闭ROS订阅和网页端口……"
            return self._last_action

    def stop_requested(self) -> bool:
        with self._lock:
            return bool(self._stop_requested)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler_for(state: CaptureWebState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GS130WCalibration/1.0"

        def _send_bytes(
            self, status: int, content_type: str, payload: bytes
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: int, payload: Dict[str, object]) -> None:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", encoded)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    WEB_PAGE.encode("utf-8"),
                )
                return
            status, preview = state.snapshot()
            if path == "/api/status":
                self._send_json(HTTPStatus.OK, status)
                return
            if path == "/preview.jpg":
                if preview is None:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"message": "摄像头画面尚未就绪。"},
                    )
                else:
                    self._send_bytes(HTTPStatus.OK, "image/jpeg", preview)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"message": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in {"/api/capture", "/api/stop"}:
                self._send_json(HTTPStatus.NOT_FOUND, {"message": "Not found"})
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"message": "Content-Type 必须为 application/json。"},
                )
                return
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length > 4096:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"message": "Request too large"},
                )
                return
            if content_length:
                self.rfile.read(content_length)
            if path == "/api/stop":
                message = state.request_stop()
                self._send_json(
                    HTTPStatus.ACCEPTED,
                    {"accepted": True, "message": message},
                )
                return
            accepted, message = state.request_capture()
            self._send_json(
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
                {"accepted": accepted, "message": message},
            )

        def do_HEAD(self) -> None:  # noqa: N802
            self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"message": "Method not allowed"})

        do_PUT = do_HEAD
        do_PATCH = do_HEAD
        do_DELETE = do_HEAD
        do_OPTIONS = do_HEAD

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


class ManualCaptureWebServer:
    def __init__(self, state: CaptureWebState, host: str, port: int) -> None:
        self._server = _ReusableThreadingHTTPServer(
            (host, int(port)), _handler_for(state)
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="gs130w-calibration-web",
            daemon=True,
        )

    def start(self) -> Tuple[str, int]:
        self._thread.start()
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


def _quality_result(
    pose: PairPose,
    minimum_area_fraction: float,
    minimum_outer_margin_px: float,
) -> Tuple[bool, str]:
    smallest_area = min(pose.left.area_fraction, pose.right.area_fraction)
    smallest_margin = min(pose.left.outer_margin_px, pose.right.outer_margin_px)
    if smallest_area < minimum_area_fraction:
        return (
            False,
            "棋盘太小，请靠近一些；最远姿态也要保证棋盘清晰可见。",
        )
    if smallest_margin < minimum_outer_margin_px:
        return (
            False,
            "棋盘外边缘太靠近画面边界或已经出框，请向画面内移动。",
        )
    return True, ""


def _status_message(
    left_found: bool,
    right_found: bool,
    quality_ok: bool,
    quality_message: str,
    stable: bool,
    stable_seconds: float,
    required_seconds: float,
    pose_unique: bool,
    nearest_pair: Optional[int],
) -> str:
    if not left_found or not right_found:
        return "请让完整棋盘同时出现在左右目画面中。"
    if not quality_ok:
        return quality_message
    if not stable:
        remaining = max(0.0, required_seconds - stable_seconds)
        return f"请保持棋盘完全静止，还需约 {remaining:.1f} 秒。"
    if not pose_unique:
        return (
            f"与已保存的第 {nearest_pair} 组过于相似；"
            "请改变位置、距离、旋转或俯仰角。"
        )
    return "条件全部通过，可以点击“采集本组”。"


def _build_preview(
    left_preview: np.ndarray,
    right_preview: np.ndarray,
    left_found: bool,
    right_found: bool,
    stable: bool,
    pose_unique: bool,
) -> bytes:
    put_label(
        left_preview,
        "PHYSICAL LEFT / BOARD={}".format("OK" if left_found else "NO"),
        (0, 255, 0) if left_found else (0, 0, 255),
    )
    put_label(
        right_preview,
        "PHYSICAL RIGHT / BOARD={}".format("OK" if right_found else "NO"),
        (0, 255, 0) if right_found else (0, 0, 255),
    )
    target_height = 720

    def resize(image: np.ndarray) -> np.ndarray:
        scale = target_height / float(image.shape[0])
        width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)

    preview = np.hstack([resize(left_preview), resize(right_preview)])
    summary = "STABLE={}  NEW_POSE={}".format(
        "YES" if stable else "NO", "YES" if pose_unique else "NO"
    )
    cv2.rectangle(preview, (0, preview.shape[0] - 42), (preview.shape[1], preview.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        preview,
        summary,
        (12, preview.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if stable and pose_unique else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(
        ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )
    if not ok:
        raise RuntimeError("无法编码网页预览图。")
    return encoded.tobytes()


def _pair_directories(root: Path) -> Tuple[Path, Path, Path]:
    return root / "left", root / "right", root / "combined"


def _existing_stems(directory: Path, suffixes: Sequence[str]) -> set[str]:
    if not directory.is_dir():
        return set()
    allowed = {suffix.lower() for suffix in suffixes}
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed and path.stem.isdigit()
    }


def _unexpected_capture_files(root: Path) -> List[Path]:
    unexpected: List[Path] = []
    allowed_suffix = {
        "left": {".png"},
        "right": {".png"},
        "combined": {".jpg", ".jpeg"},
    }
    if not root.exists():
        return unexpected
    if not root.is_dir():
        return [root]
    for path in root.iterdir():
        if path.name not in allowed_suffix:
            unexpected.append(path)
            continue
        if not path.is_dir():
            unexpected.append(path)
            continue
        for child in path.iterdir():
            if (
                not child.is_file()
                or not child.stem.isdigit()
                or len(child.stem) != 4
                or child.suffix.lower() not in allowed_suffix[path.name]
            ):
                unexpected.append(child)
    return unexpected


def _prepare_output(
    root: Path,
    resume: bool,
    count: int,
    pattern_size: Tuple[int, int],
) -> Tuple[int, List[PairPose]]:
    unexpected = _unexpected_capture_files(root)
    if unexpected:
        shown = ", ".join(str(path) for path in unexpected[:3])
        raise RuntimeError(f"目标目录包含非本工具生成的内容，拒绝继续：{shown}")
    left_dir, right_dir, combined_dir = _pair_directories(root)
    left_stems = _existing_stems(left_dir, (".png",))
    right_stems = _existing_stems(right_dir, (".png",))
    combined_stems = _existing_stems(combined_dir, (".jpg", ".jpeg"))
    any_existing = bool(left_stems or right_stems or combined_stems)
    if any_existing and not resume:
        raise RuntimeError(
            f"目标目录已有采集图，已拒绝覆盖：{root}。确认是同一次任务后使用 --resume。"
        )
    if not (left_stems == right_stems == combined_stems):
        raise RuntimeError("目标目录的 left/right/combined 文件不完整，拒绝继续。")

    ordered = sorted(left_stems)
    expected = [f"{index:04d}" for index in range(1, len(ordered) + 1)]
    if ordered != expected:
        raise RuntimeError("已有文件编号必须从0001开始连续，拒绝继续。")
    if len(ordered) > count:
        raise RuntimeError(f"已有 {len(ordered)} 组，超过本次目标 {count} 组。")

    for directory in (left_dir, right_dir, combined_dir):
        directory.mkdir(parents=True, exist_ok=True)

    poses: List[PairPose] = []
    for stem in ordered:
        left = cv2.imread(str(left_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise RuntimeError(f"无法恢复第 {stem} 组：左右图读取失败或尺寸不同。")
        left_found, left_corners, _ = find_board(left, pattern_size)
        right_found, right_corners, _ = find_board(right, pattern_size)
        if (
            not left_found
            or not right_found
            or left_corners is None
            or right_corners is None
        ):
            raise RuntimeError(f"无法恢复第 {stem} 组：棋盘角点复检失败。")
        poses.append(
            pair_pose_signature(
                left_corners,
                right_corners,
                (left.shape[1], left.shape[0]),
                pattern_size,
            )
        )
    return len(ordered), poses


def _atomic_save_image(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if temporary.exists():
        temporary.unlink()
    ok, encoded = cv2.imencode(path.suffix.lower(), image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    try:
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_pair(
    root: Path,
    stem: str,
    left: np.ndarray,
    right: np.ndarray,
    combined: np.ndarray,
) -> None:
    left_dir, right_dir, combined_dir = _pair_directories(root)
    destinations = (
        left_dir / f"{stem}.png",
        right_dir / f"{stem}.png",
        combined_dir / f"{stem}.jpg",
    )
    if any(path.exists() for path in destinations):
        raise RuntimeError(f"第 {stem} 组目标文件已存在，拒绝覆盖。")
    try:
        _atomic_save_image(destinations[0], left)
        _atomic_save_image(destinations[1], right)
        _atomic_save_image(destinations[2], combined)
    except BaseException:
        for path in destinations:
            path.unlink(missing_ok=True)
            path.with_name(path.stem + ".tmp" + path.suffix).unlink(
                missing_ok=True
            )
        raise


def main() -> int:
    args = parse_args()
    if (
        args.board_cols < 3
        or args.board_rows < 3
        or args.count < 1
        or not (1 <= args.port <= 65535)
        or args.stable_seconds < 0
        or args.stable_frames < 1
        or args.max_motion_px <= 0
        or args.max_frame_age_seconds <= 0
        or args.capture_timeout_seconds <= 0
        or args.min_area_fraction <= 0
        or args.min_outer_margin_px < 0
    ):
        print("ERROR: 参数无效。", file=sys.stderr)
        return 64
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: 网页未绑定回环地址；只允许在可信隔离局域网临时使用。",
            file=sys.stderr,
        )

    pattern_size = (args.board_cols, args.board_rows)
    try:
        saved, saved_poses = _prepare_output(
            args.output, args.resume, args.count, pattern_size
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 66
    if saved == args.count:
        print(f"目标目录已经完整包含 {saved}/{args.count} 组，无需继续。")
        return 0

    state = CaptureWebState(args.count, args.capture_timeout_seconds)
    state.update(
        {
            "saved": saved,
            "coverage": coverage_grid(saved_poses),
            "message": "正在等待ROS 2摄像头话题……",
        }
    )
    try:
        web = ManualCaptureWebServer(state, args.host, args.port)
        host, port = web.start()
    except OSError as exc:
        print(f"ERROR: 无法启动网页端口 {args.host}:{args.port}：{exc}", file=sys.stderr)
        return 69

    print(f"手动采集网页已启动：http://127.0.0.1:{port}/")
    print("此地址通过SSH -L隧道访问；只有点击网页按钮才会保存。")
    print(f"已恢复 {saved}/{args.count} 组；目标目录：{args.output}")
    subscriber: Optional[FreshFrameSubscriber] = None
    interrupted = False
    tracker = StabilityTracker(
        args.stable_seconds, args.stable_frames, args.max_motion_px
    )
    thresholds = PoseThresholds()
    last_source_sequence = 0
    last_fresh_frame_at = time.monotonic()
    frame_seq = 0
    try:
        subscriber = FreshFrameSubscriber(args.topic, args.wait_seconds)
        while saved < args.count:
            if state.stop_requested():
                interrupted = True
                break
            combined, source_sequence, received_at = subscriber.spin_once()
            if combined is None:
                continue
            now = time.monotonic()
            if source_sequence == last_source_sequence:
                state.expire_capture_request(now)
                if now - last_fresh_frame_at > args.max_frame_age_seconds:
                    tracker.reset()
                    state.update(
                        {
                            "fresh": False,
                            "ready": False,
                            "stable": False,
                            "stable_seconds": 0.0,
                            "message": "摄像头画面没有更新，已禁止采集；请检查相机服务。",
                        }
                    )
                continue
            last_source_sequence = source_sequence
            last_fresh_frame_at = now
            frame_seq = source_sequence
            state.note_frame_received(frame_seq, received_at)
            first_raw, second_raw, first_name, second_name = split_combined(
                combined, args.layout
            )
            mapping = {
                first_name: rotate(first_raw, args.rotation),
                second_name: rotate(second_raw, args.rotation),
            }
            left = mapping[args.physical_left]
            right_name = ({"top", "bottom"} - {args.physical_left}).pop()
            right = mapping[right_name]
            left_found, left_corners, left_preview = find_board(left, pattern_size)
            right_found, right_corners, right_preview = find_board(right, pattern_size)
            stable, stable_seconds, motion_px, stable_frames = tracker.update(
                left_corners if left_found else None,
                right_corners if right_found else None,
                now,
            )

            current_pose: Optional[PairPose] = None
            quality_ok = False
            quality_message = ""
            pose_unique = False
            nearest_pair: Optional[int] = None
            difference_score = 0.0
            if (
                left_found
                and right_found
                and left_corners is not None
                and right_corners is not None
            ):
                current_pose = pair_pose_signature(
                    left_corners,
                    right_corners,
                    (left.shape[1], left.shape[0]),
                    pattern_size,
                )
                quality_ok, quality_message = _quality_result(
                    current_pose,
                    args.min_area_fraction,
                    args.min_outer_margin_px,
                )
                pose_unique, nearest_pair, difference_score = nearest_saved_pose(
                    current_pose, saved_poses, thresholds
                )

            ready = bool(
                left_found
                and right_found
                and quality_ok
                and stable
                and pose_unique
                and current_pose is not None
            )
            message = _status_message(
                left_found,
                right_found,
                quality_ok,
                quality_message,
                stable,
                stable_seconds,
                args.stable_seconds,
                pose_unique,
                nearest_pair,
            )
            preview_jpeg = _build_preview(
                left_preview,
                right_preview,
                left_found,
                right_found,
                stable,
                pose_unique,
            )
            state.update(
                {
                    "saved": saved,
                    "target": args.count,
                    "fresh": True,
                    "frame_seq": frame_seq,
                    "left_found": left_found,
                    "right_found": right_found,
                    "quality_ok": quality_ok,
                    "stable": stable,
                    "stable_seconds": stable_seconds,
                    "stable_frames": stable_frames,
                    "motion_px": motion_px,
                    "pose_unique": pose_unique,
                    "nearest_pair": nearest_pair,
                    "pose_difference_score": _finite_json_float(difference_score),
                    "ready": ready,
                    "completed": False,
                    "message": message,
                    "coverage": coverage_grid(saved_poses),
                },
                preview_jpeg,
            )

            state.expire_capture_request()
            if not state.consume_capture_request(frame_seq):
                continue
            if state.stop_requested():
                interrupted = True
                state.finish_capture_request(
                    "已取消尚未写盘的采集请求，正在安全结束。"
                )
                break
            if not ready or current_pose is None:
                state.finish_capture_request(
                    "画面在点击后发生变化，本次没有保存；请重新稳定棋盘。"
                )
                continue

            stem = f"{saved + 1:04d}"
            try:
                _save_pair(args.output, stem, left, right, combined)
            except Exception as exc:
                state.finish_capture_request(f"保存失败：{exc}")
                raise
            saved_poses.append(current_pose)
            saved += 1
            tracker.reset()
            state.finish_capture_request(
                f"已保存第 {saved}/{args.count} 组。请把棋盘移出画面，再更换姿态。"
            )
            state.update(
                {
                    "saved": saved,
                    "ready": False,
                    "stable": False,
                    "stable_seconds": 0.0,
                    "pose_unique": False,
                    "message": "已保存；请先移出棋盘，再更换姿态。",
                    "coverage": coverage_grid(saved_poses),
                }
            )
            print(
                f"已采集 {saved}/{args.count}：物理左目={args.physical_left}，"
                f"物理右目={right_name}",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n已按用户要求停止；现有完整图像对已保留，可使用 --resume 继续。")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return_code = 2
    else:
        return_code = 0
    finally:
        if saved >= args.count:
            state.update(
                {
                    "saved": saved,
                    "ready": False,
                    "completed": True,
                    "message": f"采集完成：{saved}/{args.count}。",
                }
            )
            time.sleep(1.0)
        if subscriber is not None:
            subscriber.close()
        web.stop()

    if saved >= args.count:
        print(f"采集完成：{saved}/{args.count}")
        print(f"输出目录：{args.output}")
        return 0
    if interrupted:
        print(f"当前进度：{saved}/{args.count}")
        return 130
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
