#!/usr/bin/env python3
"""Inspect a combined GS130W ROS 2 stream and collect stereo chessboard pairs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检查 GS130W 上/下半帧，或从 /image_combine_jpeg 采集同步双目标定图。"
        )
    )
    parser.add_argument("--topic", default="/image_combine_jpeg")
    parser.add_argument(
        "--layout",
        choices=("vertical",),
        default="vertical",
        help="本项目 GS130W 交付链路固定为上下拼接。",
    )
    parser.add_argument(
        "--rotation",
        choices=("none", "cw90", "ccw90", "rotate180"),
        default="ccw90",
        help="拆分每个视图后采用的运行时旋转。",
    )
    parser.add_argument(
        "--mode",
        choices=("inspect", "collect"),
        default="inspect",
        help="inspect 用于遮挡法确认物理左右目；collect 用于采集标定对。",
    )
    parser.add_argument(
        "--physical-left",
        choices=("top", "bottom"),
        help="collect 模式必填，表示组合原图中哪一半来自物理左镜头。",
    )
    parser.add_argument("--output", type=Path, default=Path("gs130w_capture"))
    parser.add_argument("--board-cols", type=int, default=9, help="棋盘格内角点列数。")
    parser.add_argument("--board-rows", type=int, default=6, help="棋盘格内角点行数。")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开窗口；按 --auto-interval 自动保存，适合纯 SSH 板端。",
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.0,
        help="大于 0 时在双目均检出棋盘后自动按该秒数采集；默认按空格。",
    )
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    return parser.parse_args()


def rotate(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "rotate180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def split_combined(
    image: np.ndarray, layout: str
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    height, width = image.shape[:2]
    if layout == "vertical":
        if height % 2:
            raise ValueError(f"组合图高度必须为偶数，实际为 {height}")
        midpoint = height // 2
        return image[:midpoint], image[midpoint:], "top", "bottom"
    if width % 2:
        raise ValueError(f"组合图宽度必须为偶数，实际为 {width}")
    midpoint = width // 2
    return image[:, :midpoint], image[:, midpoint:], "left-half", "right-half"


def decode_image_message(message: object) -> np.ndarray:
    if hasattr(message, "format") and hasattr(message, "data"):
        buffer = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("CompressedImage JPEG 解码失败")
        return image

    encoding = str(getattr(message, "encoding", "")).lower()
    height = int(getattr(message, "height"))
    width = int(getattr(message, "width"))
    step = int(getattr(message, "step"))
    raw = np.frombuffer(getattr(message, "data"), dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("Image 数据长度小于 height*step")
    rows = raw[: height * step].reshape(height, step)
    if encoding in ("bgr8", "rgb8"):
        image = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
    if encoding in ("bgra8", "rgba8"):
        image = rows[:, : width * 4].reshape(height, width, 4)
        conversion = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, conversion)
    if encoding in ("mono8", "8uc1"):
        image = rows[:, :width].reshape(height, width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"暂不支持 sensor_msgs/Image 编码：{encoding}")


def find_board(
    image: np.ndarray, pattern_size: Tuple[int, int]
) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners: Optional[np.ndarray]
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
    else:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found and corners is not None:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
            )
    preview = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(preview, pattern_size, corners, bool(found))
    return bool(found), corners, preview


class LatestFrameSubscriber:
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
            raise RuntimeError("缺少 rclpy；请先 source 板端 ROS 2 环境。") from exc

        self.rclpy = rclpy
        rclpy.init(args=None)
        self.node = Node("gs130w_pair_collector")
        self.frame: Optional[np.ndarray] = None
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
            try:
                self.frame = decode_image_message(message)
                self.error = None
            except Exception as exc:  # Keep subscription alive and display the error.
                self.error = str(exc)

        self.subscription = self.node.create_subscription(
            MessageType, topic, callback, qos
        )
        print(f"已订阅 {topic}（{topic_type}）")

    def spin_once(self) -> Optional[np.ndarray]:
        self.rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.error:
            raise RuntimeError(self.error)
        return self.frame

    def close(self) -> None:
        node = getattr(self, "node", None)
        if node is not None:
            node.destroy_node()
        rclpy = getattr(self, "rclpy", None)
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


def put_label(image: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 560), 44), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    expected_halves = {"top", "bottom"}
    if args.mode == "collect" and args.physical_left not in expected_halves:
        print(
            f"ERROR: collect 模式下 --physical-left 必须是 "
            "top 或 bottom。",
            file=sys.stderr,
        )
        return 64
    if args.board_cols < 3 or args.board_rows < 3 or args.count < 1:
        print("ERROR: 棋盘参数和采集数量无效。", file=sys.stderr)
        return 64
    if args.headless and args.mode == "collect" and args.auto_interval <= 0:
        print(
            "ERROR: 无界面 collect 模式必须设置正数 --auto-interval。",
            file=sys.stderr,
        )
        return 64
    if args.headless and args.mode == "inspect" and args.auto_interval <= 0:
        args.auto_interval = 2.0

    args.output.mkdir(parents=True, exist_ok=True)
    subscriber = LatestFrameSubscriber(args.topic, args.wait_seconds)
    saved = 0
    last_auto = 0.0
    pattern_size = (args.board_cols, args.board_rows)
    if args.headless:
        print(f"无界面模式：每 {args.auto_interval:g} 秒自动检查/保存。")
    else:
        print("按键：空格=保存，s=检查模式快照，q/Esc=退出。")
    print("遮挡确认时一次只完全遮住一个镜头，观察 TOP/BOTTOM 哪一幅变黑。")

    try:
        while saved < args.count:
            combined = subscriber.spin_once()
            if combined is None:
                continue
            first_raw, second_raw, first_name, second_name = split_combined(
                combined, args.layout
            )
            first = rotate(first_raw, args.rotation)
            second = rotate(second_raw, args.rotation)
            key = -1

            if args.mode == "inspect":
                now = time.monotonic()
                auto_capture = (
                    args.headless
                    and now - last_auto >= args.auto_interval
                )
                if not args.headless:
                    first_preview = first.copy()
                    second_preview = second.copy()
                    put_label(first_preview, first_name.upper(), (0, 255, 255))
                    put_label(second_preview, second_name.upper(), (0, 255, 255))
                    preview = np.hstack(
                        [
                            cv2.resize(first_preview, (640, 480)),
                            cv2.resize(second_preview, (640, 480)),
                        ]
                    )
                    cv2.imshow("GS130W physical-view inspection", preview)
                    key = cv2.waitKey(1) & 0xFF
                if key in (ord("s"), ord(" ")) or auto_capture:
                    saved += 1
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    stem = f"{saved:04d}_{stamp}"
                    cv2.imwrite(str(args.output / f"{stem}_{first_name}.png"), first)
                    cv2.imwrite(str(args.output / f"{stem}_{second_name}.png"), second)
                    cv2.imwrite(str(args.output / f"{stem}_combined.png"), combined)
                    last_auto = now
                    print(f"已保存遮挡检查快照 {saved}/{args.count}：{stem}")
            else:
                mapping = {first_name: first, second_name: second}
                left = mapping[args.physical_left]
                right_name = (expected_halves - {args.physical_left}).pop()
                right = mapping[right_name]
                left_found, _, left_preview = find_board(left, pattern_size)
                right_found, _, right_preview = find_board(right, pattern_size)
                put_label(
                    left_preview,
                    f"PHYSICAL LEFT / board={'OK' if left_found else 'NO'}",
                    (0, 255, 0) if left_found else (0, 0, 255),
                )
                put_label(
                    right_preview,
                    f"PHYSICAL RIGHT / board={'OK' if right_found else 'NO'}",
                    (0, 255, 0) if right_found else (0, 0, 255),
                )
                if not args.headless:
                    preview = np.hstack(
                        [
                            cv2.resize(left_preview, (640, 480)),
                            cv2.resize(right_preview, (640, 480)),
                        ]
                    )
                    cv2.imshow("GS130W stereo calibration collector", preview)
                    key = cv2.waitKey(1) & 0xFF
                now = time.monotonic()
                auto_capture = (
                    args.auto_interval > 0
                    and left_found
                    and right_found
                    and now - last_auto >= args.auto_interval
                )
                if key == ord(" ") or auto_capture:
                    if not (left_found and right_found):
                        print("跳过：左右目必须同时完整检出棋盘内角点。")
                    else:
                        saved += 1
                        stem = f"{saved:04d}"
                        for directory in ("left", "right", "combined"):
                            (args.output / directory).mkdir(exist_ok=True)
                        cv2.imwrite(str(args.output / "left" / f"{stem}.png"), left)
                        cv2.imwrite(str(args.output / "right" / f"{stem}.png"), right)
                        cv2.imwrite(
                            str(args.output / "combined" / f"{stem}.jpg"), combined
                        )
                        last_auto = now
                        print(
                            f"已采集 {saved}/{args.count}：物理左目={args.physical_left}，"
                            f"物理右目={right_name}"
                        )
            if key in (27, ord("q")):
                break
    finally:
        subscriber.close()
        if not args.headless:
            cv2.destroyAllWindows()

    if args.mode == "collect" and saved < max(12, args.count // 2):
        print(f"ERROR: 仅采集 {saved} 对，建议至少 20 对且姿态/距离覆盖充分。", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
