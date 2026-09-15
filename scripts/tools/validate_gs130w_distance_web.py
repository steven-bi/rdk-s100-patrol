#!/usr/bin/env python3
"""Strict three-point GS130W distance validation through a local web page.

This tool is deliberately conservative:

* it accepts a ``valid: false`` calibration candidate but enables it only in
  memory for measurement;
* it captures a frame strictly after each browser click;
* each nominal distance (1 m, 3 m and 5 m) is an independent three-frame
  attempt, and every frame gets exactly one ROI submission;
* failed attempts and their source images remain in the audit trail;
* the candidate is never modified; only a fully passing session can atomically
  create a new ``valid: true`` YAML file.

The stereo matcher, rectification and distance filtering intentionally import
the deployed ``rdk_patrol.stereo.depth`` implementation so validation cannot
silently drift away from the patrol runtime.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Optional

import cv2
import numpy as np
try:
    import yaml
except ImportError:  # --help/--self-test remain available on a plain workstation.
    yaml = None  # type: ignore[assignment]


SCHEMA_VERSION = "rdk-patrol-gs130w-distance-validation/v1"
SLOT_ORDER = ("1m", "3m", "5m")
SLOT_NOMINAL_M = {"1m": 1.0, "3m": 3.0, "5m": 5.0}
FRAMES_PER_ATTEMPT = 3
BEIJING = dt.timezone(dt.timedelta(hours=8))

MIN_ROI_PIXELS = 400
MIN_VALID_PIXELS = 200
MIN_VALID_FRACTION = 0.35
MAX_RELATIVE_MAD = 0.10
MAX_CENTRAL80_SPREAD = 0.25
MAX_RELATIVE_ERROR = 0.10
MIN_PASSING_FRAMES = 2
MAX_PASSING_ESTIMATE_SPAN = 0.03
MIN_DISTANCE_SEPARATION = 0.05


class ValidationError(RuntimeError):
    """Expected validation or safety failure."""


def require_yaml() -> Any:
    if yaml is None:
        raise ValidationError("缺少 PyYAML；请使用项目 .venv 运行或安装 requirements.txt")
    return yaml


def now_beijing() -> str:
    return dt.datetime.now(BEIJING).isoformat(timespec="milliseconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace a mutable session file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Atomically create an immutable file and refuse every overwrite.

    A fully synced temporary inode is hard-linked into place.  ``os.link`` is
    an atomic no-replace operation on the Linux filesystem used by the board.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError(f"拒绝覆盖已有文件：{path}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_png_exclusive(path: Path, image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValidationError(f"PNG 编码失败：{path.name}")
    payload = encoded.tobytes()
    atomic_create_bytes(path, payload)
    return sha256_bytes(payload)


def encode_jpeg(image: np.ndarray, quality: int = 86) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise ValidationError("JPEG 预览编码失败")
    return encoded.tobytes()


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} 必须是有限数值") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} 必须是有限数值")
    return result


def decode_image_message(message: object) -> np.ndarray:
    if hasattr(message, "format") and hasattr(message, "data"):
        buffer = np.frombuffer(getattr(message, "data"), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValidationError("CompressedImage JPEG 解码失败")
        return image

    encoding = str(getattr(message, "encoding", "")).lower()
    height = int(getattr(message, "height"))
    width = int(getattr(message, "width"))
    step = int(getattr(message, "step"))
    raw = np.frombuffer(getattr(message, "data"), dtype=np.uint8)
    if raw.size < height * step:
        raise ValidationError("Image 数据长度小于 height*step")
    rows = raw[: height * step].reshape(height, step)
    if encoding in {"bgr8", "rgb8"}:
        image = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
    if encoding in {"bgra8", "rgba8"}:
        image = rows[:, : width * 4].reshape(height, width, 4)
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, code)
    if encoding in {"mono8", "8uc1"}:
        image = rows[:, :width].reshape(height, width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValidationError(f"不支持 sensor_msgs/Image 编码：{encoding}")


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    normalized = str(rotation).lower().replace("_", "")
    if normalized in {"", "none", "0"}:
        return image.copy()
    if normalized in {"ccw90", "90ccw"}:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if normalized in {"cw90", "90cw"}:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized in {"180", "rotate180"}:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValidationError(f"不支持的运行旋转：{rotation}")


def split_physical_views(
    combined: np.ndarray,
    *,
    layout: str,
    physical_left_view: str,
    physical_right_view: str,
    rotation: str,
) -> tuple[np.ndarray, np.ndarray]:
    if combined is None or combined.size == 0 or combined.ndim not in (2, 3):
        raise ValidationError("GS130W 组合帧为空或格式错误")
    if layout != "vertical":
        raise ValidationError("严格验证工具只支持 GS130W vertical 组合布局")
    height = combined.shape[0]
    if height < 2 or height % 2:
        raise ValidationError("vertical 组合帧高度必须为偶数")
    top, bottom = np.split(combined, 2, axis=0)
    views = {"top": top, "bottom": bottom}
    if {physical_left_view, physical_right_view} != {"top", "bottom"}:
        raise ValidationError("物理左右目映射必须是 top/bottom 的明确排列")
    return (
        rotate_image(views[physical_left_view], rotation),
        rotate_image(views[physical_right_view], rotation),
    )


class FreshFrameSubscriber:
    """ROS subscriber exposing callback sequence and monotonic receive time."""

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
            raise ValidationError("缺少 rclpy；请先 source 板端 ROS 2 环境") from exc

        self.rclpy = rclpy
        rclpy.init(args=None)
        self.node = Node("gs130w_strict_distance_validator")
        self.frame: Optional[np.ndarray] = None
        self.sequence = 0
        self.received_at = 0.0
        self.error: Optional[str] = None

        deadline = time.monotonic() + float(wait_seconds)
        topic_type = ""
        while time.monotonic() < deadline and not topic_type:
            for name, types in self.node.get_topic_names_and_types():
                if name == topic and types:
                    topic_type = str(types[0])
                    break
            if not topic_type:
                rclpy.spin_once(self.node, timeout_sec=0.2)
        if not topic_type:
            self.close()
            raise ValidationError(f"{wait_seconds:g} 秒内未发现 ROS 2 话题：{topic}")

        if topic_type == "sensor_msgs/msg/CompressedImage":
            from sensor_msgs.msg import CompressedImage as MessageType
        elif topic_type == "sensor_msgs/msg/Image":
            from sensor_msgs.msg import Image as MessageType
        else:
            self.close()
            raise ValidationError(f"不支持的 ROS 话题类型：{topic_type}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        def callback(message: object) -> None:
            received = time.monotonic()
            try:
                decoded = decode_image_message(message)
                self.frame = decoded
                self.sequence += 1
                self.received_at = received
                self.error = None
            except Exception as exc:  # callback must not escape into rclpy
                self.error = str(exc)

        self.subscription = self.node.create_subscription(
            MessageType, topic, callback, qos
        )
        self.topic_type = topic_type

    def spin_once(self) -> tuple[Optional[np.ndarray], int, float]:
        self.rclpy.spin_once(self.node, timeout_sec=0.10)
        if self.error:
            raise ValidationError(self.error)
        return self.frame, int(self.sequence), float(self.received_at)

    def close(self) -> None:
        node = getattr(self, "node", None)
        if node is not None:
            node.destroy_node()
            self.node = None
        rclpy = getattr(self, "rclpy", None)
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


@dataclass(frozen=True)
class StereoFrame:
    combined: np.ndarray
    left: np.ndarray
    right: np.ndarray
    left_rectified: np.ndarray
    right_rectified: np.ndarray
    sequence: int
    received_at: float


class RuntimeStereoEngine:
    """Use the deployed patrol depth implementation with an in-memory candidate."""

    def __init__(self, candidate: Path, project_root: Path) -> None:
        self.candidate_path = candidate.resolve()
        self.candidate_bytes = candidate.read_bytes()
        self.candidate_sha256 = sha256_bytes(self.candidate_bytes)
        try:
            raw = require_yaml().safe_load(self.candidate_bytes.decode("utf-8-sig"))
        except Exception as exc:
            raise ValidationError(f"候选 YAML 解析失败：{type(exc).__name__}") from exc
        if not isinstance(raw, Mapping):
            raise ValidationError("候选 YAML 顶层必须映射")
        self.pristine_payload = copy.deepcopy(dict(raw))
        nested = self.pristine_payload.get("calibration")
        source_payload = dict(nested) if isinstance(nested, Mapping) else dict(self.pristine_payload)
        if bool(source_payload.get("valid", self.pristine_payload.get("valid", False))):
            raise ValidationError("输入必须是尚未验证的 valid:false 候选文件")
        if not bool(source_payload.get("enabled", self.pristine_payload.get("enabled", True))):
            raise ValidationError("候选标定 enabled:false，拒绝测距验证")

        for key in (
            "physical_left_view",
            "physical_right_view",
            "runtime_rotation",
            "view_mapping",
        ):
            if key not in source_payload and key in self.pristine_payload:
                source_payload[key] = copy.deepcopy(self.pristine_payload[key])
        source_payload["valid"] = True  # memory only; pristine bytes are untouched

        src = project_root.resolve() / "src"
        if not src.is_dir():
            raise ValidationError(f"找不到部署项目 src 目录：{src}")
        sys.path.insert(0, str(src))
        try:
            from rdk_patrol.stereo.calibration import calibration_from_mapping
            from rdk_patrol.stereo.depth import StereoDepthEstimator
            from rdk_patrol.stereo import gs130w as _runtime_gs130w  # noqa: F401
        except Exception as exc:
            raise ValidationError(
                f"无法导入部署版双目测距实现：{type(exc).__name__}: {exc}"
            ) from exc

        # Archive the exact Python sources that produced every distance result.
        # Keeping only this validator is insufficient: a changed patrol depth
        # implementation could otherwise make a forged/old result impossible
        # to reproduce during the later installation review.
        self.runtime_source_files: dict[str, dict[str, Any]] = {}
        for key, module_name, archive_name in (
            ("depth", "rdk_patrol.stereo.depth", "runtime_depth.py"),
            ("calibration", "rdk_patrol.stereo.calibration", "runtime_calibration.py"),
            ("gs130w", "rdk_patrol.stereo.gs130w", "runtime_gs130w.py"),
        ):
            module = sys.modules.get(module_name)
            source_text = getattr(module, "__file__", None)
            if not source_text:
                raise ValidationError(f"无法定位已导入运行时源码：{module_name}")
            source_path = Path(str(source_text)).resolve()
            if source_path.suffix in {".pyc", ".pyo"}:
                candidate_source = source_path.with_suffix(".py")
                if candidate_source.is_file():
                    source_path = candidate_source
            if source_path.suffix != ".py" or not source_path.is_file():
                raise ValidationError(f"运行时模块不是可归档的 .py 源码：{source_path}")
            source_bytes = source_path.read_bytes()
            self.runtime_source_files[key] = {
                "source_path": str(source_path),
                "archive_file": archive_name,
                "sha256": sha256_bytes(source_bytes),
                "bytes": source_bytes,
            }

        calibration = calibration_from_mapping(source_payload)
        if not calibration.valid:
            raise ValidationError(f"候选标定参数不完整：{calibration.reason}")
        matcher = self.pristine_payload.get("matcher", source_payload.get("matcher", {}))
        if not isinstance(matcher, Mapping):
            matcher = {}

        def integer(name: str, default: int) -> int:
            try:
                return int(matcher.get(name, default))
            except (TypeError, ValueError, OverflowError):
                return int(default)

        def number(name: str, default: float) -> float:
            try:
                result = float(matcher.get(name, default))
            except (TypeError, ValueError, OverflowError):
                return float(default)
            return result if math.isfinite(result) else float(default)

        valid_disparity_min = number(
            "valid_disparity_min", number("min_disparity", 0.0) + 0.75
        )
        self.estimator = StereoDepthEstimator(
            calibration,
            min_valid_pixels=integer("minimum_valid_pixels", 24),
            min_disparity=valid_disparity_min,
            min_distance_m=number("min_distance_m", 0.15),
            max_distance_m=number("max_distance_m", 30.0),
            max_relative_mad=number("max_relative_mad", 0.40),
            matcher_min_disparity=integer("min_disparity", 0),
            num_disparities=integer("num_disparities", 128),
            block_size=integer("block_size", 5),
            uniqueness_ratio=integer("uniqueness_ratio", 10),
            speckle_window_size=integer("speckle_window_size", 80),
            speckle_range=integer("speckle_range", 2),
        )
        self.calibration = calibration
        self.layout = str(self.pristine_payload.get("combined_layout", "vertical"))
        self.rotation = str(calibration.runtime_rotation or "none")
        self.physical_left_view = str(calibration.physical_left_view)
        self.physical_right_view = str(calibration.physical_right_view)
        if {self.physical_left_view, self.physical_right_view} != {"top", "bottom"}:
            raise ValidationError("候选文件缺少明确物理左右目 top/bottom 映射")
        self.expected_size = tuple(int(v) for v in calibration.image_size or ())
        if len(self.expected_size) != 2:
            raise ValidationError("候选文件缺少有效 image_size")
        self._matcher = self.estimator._create_matcher(self.expected_size[0])

    def prepare_frame(
        self, combined: np.ndarray, sequence: int, received_at: float
    ) -> StereoFrame:
        left, right = split_physical_views(
            combined,
            layout=self.layout,
            physical_left_view=self.physical_left_view,
            physical_right_view=self.physical_right_view,
            rotation=self.rotation,
        )
        if left.shape[:2] != right.shape[:2]:
            raise ValidationError("物理左右目尺寸不一致")
        size = (left.shape[1], left.shape[0])
        if size != self.expected_size:
            raise ValidationError(
                f"当前单目尺寸 {size} 与候选标定 {self.expected_size} 不同"
            )
        try:
            left_rect, right_rect = self.estimator._rectify(left, right)
        except Exception as exc:
            raise ValidationError(f"双目校正失败：{type(exc).__name__}") from exc
        return StereoFrame(
            combined=combined.copy(),
            left=left,
            right=right,
            left_rectified=left_rect,
            right_rectified=right_rect,
            sequence=int(sequence),
            received_at=float(received_at),
        )

    def evaluate(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        roi: tuple[int, int, int, int],
        known_distance_m: float,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        if left_rectified.shape != right_rectified.shape:
            raise ValidationError("校正后左右图尺寸不一致")
        height, width = left_rectified.shape[:2]
        x, y, roi_width, roi_height = (int(v) for v in roi)
        if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
            raise ValidationError("ROI 坐标必须为图像内的正整数矩形")
        if x + roi_width > width or y + roi_height > height:
            raise ValidationError("ROI 超出校正后物理左目图像")
        if roi_width < 5 or roi_height < 5:
            raise ValidationError("ROI 宽高均必须至少 5 px")

        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        try:
            disparity = self._matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            xyz = cv2.reprojectImageTo3D(
                disparity,
                np.asarray(self.calibration.q_matrix, dtype=np.float64),
                handleMissingValues=False,
            )
        except (cv2.error, TypeError, ValueError) as exc:
            raise ValidationError(f"双目视差计算失败：{type(exc).__name__}") from exc

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y : y + roi_height, x : x + roi_width] = 1
        erosion = max(1, int(round(min(roi_width, roi_height) * 0.08)))
        if erosion > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion * 2 + 1, erosion * 2 + 1)
            )
            mask = cv2.erode(mask, kernel)
        roi_pixels = int(np.count_nonzero(mask))
        distances = np.linalg.norm(xyz, axis=2)
        valid = (
            (mask > 0)
            & np.isfinite(distances)
            & np.isfinite(disparity)
            & (disparity > self.estimator.min_disparity)
            & (distances >= self.estimator.min_distance_m)
            & (distances <= self.estimator.max_distance_m)
            & (xyz[:, :, 2] > 0)
        )
        values = distances[valid].astype(np.float64)
        disparities = disparity[valid].astype(np.float64)
        valid_pixels = int(values.size)
        valid_fraction = valid_pixels / max(roi_pixels, 1)

        estimate: Optional[float] = None
        relative_mad: Optional[float] = None
        central80_spread: Optional[float] = None
        relative_error: Optional[float] = None
        median_disparity: Optional[float] = None
        trimmed_pixels = 0
        low: Optional[float] = None
        high: Optional[float] = None
        if valid_pixels:
            low_value, high_value = np.percentile(values, [10.0, 90.0])
            low, high = float(low_value), float(high_value)
            trimmed = values[(values >= low_value) & (values <= high_value)]
            # Keep the runtime's exact 10–90% fallback rule; the stricter
            # 200-pixel acceptance gate remains an independent check below.
            if trimmed.size < self.estimator.min_valid_pixels:
                trimmed = values
            trimmed_pixels = int(trimmed.size)
            estimate = float(np.median(trimmed))
            mad = float(np.median(np.abs(trimmed - estimate)))
            relative_mad = mad / max(estimate, 1e-9)
            central80_spread = (high - low) / max(estimate, 1e-9)
            relative_error = abs(estimate - known_distance_m) / known_distance_m
            median_disparity = float(np.median(disparities))

        checks = {
            "roi_pixels": roi_pixels >= MIN_ROI_PIXELS,
            "valid_pixels": valid_pixels >= MIN_VALID_PIXELS,
            "valid_fraction": valid_fraction >= MIN_VALID_FRACTION,
            "relative_mad": relative_mad is not None and relative_mad <= MAX_RELATIVE_MAD,
            "central80_spread": central80_spread is not None
            and central80_spread <= MAX_CENTRAL80_SPREAD,
            "relative_error": relative_error is not None
            and relative_error <= MAX_RELATIVE_ERROR,
        }
        passed = all(checks.values())
        reasons = [name for name, ok in checks.items() if not ok]
        metrics: dict[str, Any] = {
            "passed": bool(passed),
            "failure_checks": reasons,
            "known_distance_m": float(known_distance_m),
            "estimated_distance_m": estimate,
            "relative_error": relative_error,
            "roi": {"x": x, "y": y, "width": roi_width, "height": roi_height},
            "roi_pixels_after_boundary_erosion": roi_pixels,
            "boundary_erosion_px": erosion,
            "valid_pixels": valid_pixels,
            "valid_fraction": float(valid_fraction),
            "trimmed_pixels": trimmed_pixels,
            "relative_mad": relative_mad,
            "central80_low_m": low,
            "central80_high_m": high,
            "central80_spread_over_median": central80_spread,
            "median_disparity_px": median_disparity,
            "thresholds": {
                "minimum_roi_pixels": MIN_ROI_PIXELS,
                "minimum_valid_pixels": MIN_VALID_PIXELS,
                "minimum_valid_fraction": MIN_VALID_FRACTION,
                "maximum_relative_mad": MAX_RELATIVE_MAD,
                "maximum_central80_spread_over_median": MAX_CENTRAL80_SPREAD,
                "maximum_relative_error": MAX_RELATIVE_ERROR,
                "minimum_valid_disparity_px": float(self.estimator.min_disparity),
                "minimum_distance_m": float(self.estimator.min_distance_m),
                "maximum_distance_m": float(self.estimator.max_distance_m),
            },
            "checks": checks,
            "distance_definition": "physical_left_camera_to_roi_euclidean_m",
        }

        annotated = left_rectified.copy()
        color = (0, 200, 0) if passed else (0, 0, 255)
        cv2.rectangle(
            annotated,
            (x, y),
            (x + roi_width - 1, y + roi_height - 1),
            color,
            3,
        )
        estimate_text = "unavailable" if estimate is None else f"{estimate:.3f}m"
        error_text = "n/a" if relative_error is None else f"{relative_error:.1%}"
        labels = [
            f"{'PASS' if passed else 'FAIL'} estimate={estimate_text} known={known_distance_m:.3f}m",
            f"valid={valid_pixels}/{roi_pixels} ({valid_fraction:.1%}) error={error_text}",
        ]
        for index, label in enumerate(labels):
            cv2.putText(
                annotated,
                label,
                (12, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        valid_disp = disparity[np.isfinite(disparity) & (disparity > self.estimator.min_disparity)]
        if valid_disp.size:
            dlow, dhigh = np.percentile(valid_disp, [2.0, 98.0])
            scale = 255.0 / max(float(dhigh - dlow), 1e-6)
            visual = np.clip((disparity - dlow) * scale, 0, 255).astype(np.uint8)
        else:
            visual = np.zeros(disparity.shape, dtype=np.uint8)
        disparity_color = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
        disparity_color[~np.isfinite(disparity)] = 0
        cv2.rectangle(
            disparity_color,
            (x, y),
            (x + roi_width - 1, y + roi_height - 1),
            color,
            2,
        )
        return metrics, annotated, disparity_color


class SessionStore:
    """Persistent, resumable and hash-audited validation session."""

    def __init__(
        self,
        engine: RuntimeStereoEngine,
        session_dir: Path,
        validated_output: Path,
        *,
        resume: bool,
    ) -> None:
        self.engine = engine
        self.root = session_dir.resolve()
        self.manifest_path = self.root / "session.json"
        self.audit_path = self.root / "audit.jsonl"
        self.pristine_path = self.root / "candidate_pristine.yaml"
        self.validator_archive_path = self.root / "validator_script.py"
        self.runtime_archive_paths = {
            "depth": self.root / "runtime_depth.py",
            "calibration": self.root / "runtime_calibration.py",
            "gs130w": self.root / "runtime_gs130w.py",
        }
        self.manifest_before_validation_path = self.root / "session_before_validation.json"
        self.audit_before_validation_path = self.root / "audit_before_validation.jsonl"
        self.finalize_transaction_path = self.root / "finalize_transaction.json"
        self.validated_output = validated_output.resolve()
        self._lock = threading.RLock()
        self._audit_tail = "0" * 64

        runtime_sources = getattr(self.engine, "runtime_source_files", None)
        if not isinstance(runtime_sources, Mapping) or set(runtime_sources) != {
            "depth",
            "calibration",
            "gs130w",
        }:
            raise ValidationError("测距引擎没有提供完整的 depth/calibration/gs130w 运行时源码")
        self.runtime_sources: dict[str, dict[str, Any]] = {}
        for key, expected_name in (
            ("depth", "runtime_depth.py"),
            ("calibration", "runtime_calibration.py"),
            ("gs130w", "runtime_gs130w.py"),
        ):
            record = runtime_sources[key]
            if not isinstance(record, Mapping):
                raise ValidationError(f"{key} 运行时源码记录格式错误")
            source_bytes = record.get("bytes")
            if not isinstance(source_bytes, bytes) or not source_bytes:
                raise ValidationError(f"{key} 运行时源码内容为空")
            claimed_hash = str(record.get("sha256") or "")
            if sha256_bytes(source_bytes) != claimed_hash:
                raise ValidationError(f"{key} 运行时源码 SHA256 自检失败")
            if str(record.get("archive_file")) != expected_name:
                raise ValidationError(f"{key} 运行时归档文件名不符合固定契约")
            self.runtime_sources[key] = {
                "source_path": str(record.get("source_path") or ""),
                "archive_file": expected_name,
                "sha256": claimed_hash,
                "bytes": source_bytes,
            }

        if self.validated_output.parent != self.root:
            raise ValidationError(
                "--validated-output 必须位于 --session-dir 内，以便独立审计原始候选副本"
            )
        if resume:
            self._load_existing()
        else:
            self._create_new()

    def _create_new(self) -> None:
        session_id = self.root.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", session_id):
            raise ValidationError(
                "session-dir 最后一级目录名即 session_id，只能使用 3–128 位英文字母、数字、._-"
            )
        if self.root.exists() and any(self.root.iterdir()):
            raise ValidationError(
                f"新会话目录非空，拒绝混用：{self.root}；继续请加 --resume"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        if self.validated_output.exists():
            raise ValidationError(f"验证输出已存在，拒绝覆盖：{self.validated_output}")
        atomic_create_bytes(self.pristine_path, self.engine.candidate_bytes)
        validator_bytes = Path(__file__).resolve().read_bytes()
        atomic_create_bytes(self.validator_archive_path, validator_bytes)
        for key, archive_path in self.runtime_archive_paths.items():
            atomic_create_bytes(archive_path, self.runtime_sources[key]["bytes"])
        atomic_create_bytes(self.audit_path, b"")
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "created_at": now_beijing(),
            "updated_at": now_beijing(),
            "status": "in_progress",
            "candidate_source_path": str(self.engine.candidate_path),
            "candidate_pristine_file": self.pristine_path.name,
            "candidate_sha256": self.engine.candidate_sha256,
            "validator_script_file": self.validator_archive_path.name,
            "validator_script_sha256": sha256_bytes(validator_bytes),
            "runtime_source_archives": {
                key: {
                    "file": self.runtime_archive_paths[key].name,
                    "sha256": self.runtime_sources[key]["sha256"],
                    "deployed_source_path": self.runtime_sources[key]["source_path"],
                }
                for key in ("depth", "calibration", "gs130w")
            },
            "validated_output_file": self.validated_output.name,
            "runtime_mapping": {
                "combined_layout": self.engine.layout,
                "runtime_rotation": self.engine.rotation,
                "physical_left_view": self.engine.physical_left_view,
                "physical_right_view": self.engine.physical_right_view,
                "image_size": list(self.engine.expected_size),
            },
            "thresholds": self.thresholds(),
            "slots": {
                slot: {
                    "nominal_distance_m": SLOT_NOMINAL_M[slot],
                    "locked": False,
                    "locked_attempt": None,
                    "aggregate": None,
                    "attempts": [],
                }
                for slot in SLOT_ORDER
            },
            "validated_yaml_sha256": None,
            "completion": None,
        }
        self._save_manifest()
        self.audit("session_created", {"candidate_sha256": self.engine.candidate_sha256})

    def _load_existing(self) -> None:
        if not self.root.is_dir() or not self.manifest_path.is_file():
            raise ValidationError(f"--resume 找不到完整会话：{self.root}")
        try:
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValidationError(f"会话清单解析失败：{type(exc).__name__}") from exc
        if not isinstance(loaded, dict) or loaded.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError("会话清单 schema_version 不匹配")
        self.data = loaded
        if str(loaded.get("session_id")) != self.root.name:
            raise ValidationError("manifest.session_id 必须与 session-dir 末级目录名完全一致")
        if str(loaded.get("candidate_sha256")) != self.engine.candidate_sha256:
            raise ValidationError("--resume 的候选 YAML 与原会话 SHA256 不同")
        if not self.pristine_path.is_file():
            raise ValidationError("会话缺少 candidate_pristine.yaml")
        if sha256_file(self.pristine_path) != self.engine.candidate_sha256:
            raise ValidationError("会话 pristine 候选副本 SHA256 异常")
        archived_validator_hash = str(loaded.get("validator_script_sha256") or "")
        if (
            not self.validator_archive_path.is_file()
            or sha256_file(self.validator_archive_path) != archived_validator_hash
        ):
            raise ValidationError("会话归档的 validator_script.py 缺失或 SHA256 异常")
        if sha256_file(Path(__file__).resolve()) != archived_validator_hash:
            raise ValidationError(
                "当前运行的验证工具与会话归档版本 SHA256 不同；必须用该会话内 validator_script.py 恢复"
            )
        archived_runtime = loaded.get("runtime_source_archives")
        if not isinstance(archived_runtime, Mapping) or set(archived_runtime) != {
            "depth",
            "calibration",
            "gs130w",
        }:
            raise ValidationError("会话缺少完整 runtime_source_archives")
        for key, archive_path in self.runtime_archive_paths.items():
            record = archived_runtime[key]
            if not isinstance(record, Mapping):
                raise ValidationError(f"会话 {key} 运行时源码记录格式错误")
            expected_name = self.runtime_sources[key]["archive_file"]
            expected_hash = self.runtime_sources[key]["sha256"]
            if str(record.get("file")) != expected_name:
                raise ValidationError(f"会话 {key} 运行时归档文件名异常")
            if str(record.get("sha256")) != expected_hash:
                raise ValidationError(f"当前部署的 {key} 运行时源码与原会话不同")
            if not archive_path.is_file() or sha256_file(archive_path) != expected_hash:
                raise ValidationError(f"会话归档的 {expected_name} 缺失或 SHA256 异常")
        if str(loaded.get("validated_output_file")) != self.validated_output.name:
            raise ValidationError("--validated-output 与原会话记录不同")
        self._verify_audit_chain()
        loaded_status = str(self.data.get("status"))
        if loaded_status not in {"finalizing", "completed"}:
            self._archive_uncommitted_directories()
            self._recover_committed_evaluations()
        self._verify_recorded_files()

        # The transaction itself contains the complete validated YAML.  It is
        # therefore the source of truth even if power failed just before the
        # manifest could be switched from in_progress to finalizing.
        if self.finalize_transaction_path.is_file() and str(self.data.get("status")) != "completed":
            self._recover_finalize_transaction()

        status = str(self.data.get("status"))
        if status == "completed":
            expected = str(self.data.get("validated_yaml_sha256") or "")
            if not self.validated_output.is_file() or sha256_file(self.validated_output) != expected:
                raise ValidationError("已完成会话的 validated YAML 缺失或 SHA256 不匹配")
            self._ensure_final_audit_event(recovered=True)
            checksums = self.root / "SESSION_SHA256SUMS.txt"
            if checksums.exists():
                self._verify_session_checksums()
            else:
                self._write_session_checksums()
        elif status == "finalizing":
            raise ValidationError("会话处于 finalizing，但完整 finalize_transaction.json 缺失")
        elif self.validated_output.exists():
            raise ValidationError("未完成会话却已存在 validated YAML，拒绝继续")
        elif self.manifest_before_validation_path.exists() or self.audit_before_validation_path.exists():
            # A crash may happen while the two immutable pre-validation
            # snapshots are being created.  Do not mutate manifest/audit here;
            # main() will call finalize() again and safely finish the plan.
            if not self.all_slots_locked():
                raise ValidationError("存在最终化快照，但三个距离槽并未全部锁定")
        else:
            self.data["status"] = "in_progress"
            self._save_manifest()
            self.audit("session_resumed", {})

    @staticmethod
    def thresholds() -> dict[str, Any]:
        return {
            "frames_per_attempt": FRAMES_PER_ATTEMPT,
            "minimum_roi_pixels_after_boundary_erosion": MIN_ROI_PIXELS,
            "minimum_valid_pixels": MIN_VALID_PIXELS,
            "minimum_valid_fraction": MIN_VALID_FRACTION,
            "maximum_relative_mad": MAX_RELATIVE_MAD,
            "maximum_central80_spread_over_median": MAX_CENTRAL80_SPREAD,
            "maximum_relative_error": MAX_RELATIVE_ERROR,
            "minimum_passing_frames": MIN_PASSING_FRAMES,
            "maximum_passing_estimate_span_over_median": MAX_PASSING_ESTIMATE_SPAN,
            "minimum_known_distance_separation": MIN_DISTANCE_SEPARATION,
        }

    def _verify_audit_chain(self) -> None:
        if not self.audit_path.is_file():
            raise ValidationError("会话缺少 audit.jsonl")
        raw_audit = self.audit_path.read_bytes()
        if raw_audit and not raw_audit.endswith(b"\n"):
            if (
                str(self.data.get("status")) == "completed"
                and (self.root / "SESSION_SHA256SUMS.txt").exists()
            ):
                raise ValidationError("已封存完成会话的 audit.jsonl 尾部损坏，拒绝自动修复")
            # Preserve, rather than discard, a proven incomplete final append.
            # Every valid event emitted by audit() is exactly one newline-
            # terminated JSON record, so only the non-newline tail is eligible.
            last_newline = raw_audit.rfind(b"\n")
            complete_prefix = raw_audit[: last_newline + 1] if last_newline >= 0 else b""
            incomplete_tail = raw_audit[last_newline + 1 :]
            tail_hash = sha256_bytes(incomplete_tail)
            evidence = self.root / f"audit_incomplete_tail_{tail_hash}.bin"
            if evidence.is_file():
                if sha256_file(evidence) != tail_hash:
                    raise ValidationError("已有审计尾部证据文件 SHA256 冲突")
            else:
                atomic_create_bytes(evidence, incomplete_tail)
            atomic_replace_bytes(self.audit_path, complete_prefix)
        previous = "0" * 64
        with self.audit_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"审计日志第 {line_number} 行损坏") from exc
                claimed = str(event.pop("event_sha256", ""))
                if event.get("previous_event_sha256") != previous:
                    raise ValidationError(f"审计日志第 {line_number} 行链接断裂")
                canonical = json.dumps(
                    event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                actual = sha256_bytes(canonical)
                if claimed != actual:
                    raise ValidationError(f"审计日志第 {line_number} 行 SHA256 错误")
                previous = actual
        self._audit_tail = previous

    def _ensure_final_audit_event(self, *, recovered: bool) -> None:
        """Idempotently repair only the one documented final-audit crash window."""

        try:
            output_payload = require_yaml().safe_load(
                self.validated_output.read_text(encoding="utf-8-sig")
            )
        except Exception as exc:
            raise ValidationError("无法读取已生成 validated YAML 的审计来源") from exc
        if not isinstance(output_payload, Mapping):
            raise ValidationError("validated YAML 顶层不是映射")
        nested = output_payload.get("calibration")
        target = nested if isinstance(nested, Mapping) else output_payload
        validation = target.get("validation", {})
        provenance = validation.get("distance_validation", {}) if isinstance(validation, Mapping) else {}
        if not isinstance(provenance, Mapping):
            raise ValidationError("validated YAML 缺少 distance_validation 来源")
        prefix_hash = str(provenance.get("audit_log_sha256_before_validation") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", prefix_hash):
            raise ValidationError("validated YAML 的 audit_log_sha256_before_validation 无效")
        current_file_hash = sha256_file(self.audit_path)
        output_hash = sha256_file(self.validated_output)
        expected_details = {
            "file": self.validated_output.name,
            "sha256": output_hash,
            "source_candidate_sha256": self.engine.candidate_sha256,
        }
        if current_file_hash == prefix_hash:
            details = dict(expected_details)
            if recovered:
                details["recovered_after_interruption"] = True
            self.audit("validated_yaml_created", details)
            return
        last_event: Optional[dict[str, Any]] = None
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last_event = json.loads(line)
        if not isinstance(last_event, dict) or last_event.get("event") != "validated_yaml_created":
            raise ValidationError(
                "validated YAML 已存在，但审计日志既非记录前前缀，末事件也非 validated_yaml_created"
            )
        details = last_event.get("details", {})
        if not isinstance(details, Mapping) or any(
            str(details.get(key)) != str(value) for key, value in expected_details.items()
        ):
            raise ValidationError("validated_yaml_created 最终审计事件与输出文件不匹配")

    def _verify_recorded_files(self) -> None:
        combined_hashes: set[str] = set()
        recorded_directories: set[str] = set()
        slots = self.data.get("slots", {})
        if not isinstance(slots, Mapping) or set(slots) != set(SLOT_ORDER):
            raise ValidationError("会话距离槽结构异常")
        for slot in SLOT_ORDER:
            attempts = slots[slot].get("attempts", [])
            if not isinstance(attempts, list):
                raise ValidationError(f"{slot} attempts 结构异常")
            for attempt in attempts:
                for frame in attempt.get("frames", []):
                    if frame.get("candidate_sha256") != self.engine.candidate_sha256:
                        raise ValidationError(f"{slot} 历史帧使用了不同候选参数")
                    relative_dir = Path(str(frame.get("relative_dir", "")))
                    if relative_dir.is_absolute() or ".." in relative_dir.parts:
                        raise ValidationError(f"{slot} 历史帧目录不安全")
                    frame_dir = self.root / relative_dir
                    recorded_directories.add(relative_dir.as_posix())
                    hashes = frame.get("file_sha256", {})
                    if not isinstance(hashes, Mapping):
                        raise ValidationError(f"{slot} 历史帧缺少文件哈希")
                    for filename, expected in hashes.items():
                        path = frame_dir / str(filename)
                        if not path.is_file() or sha256_file(path) != str(expected):
                            raise ValidationError(f"历史文件缺失或被改动：{path}")
                    combined_hash = str(frame.get("combined_pixel_sha256") or "")
                    if len(combined_hash) != 64 or combined_hash in combined_hashes:
                        raise ValidationError("历史同步帧图像哈希缺失或重复")
                    combined_hashes.add(combined_hash)
        slots_root = self.root / "slots"
        actual_directories = {
            path.relative_to(self.root).as_posix()
            for path in slots_root.glob("*/attempt_*/frame_*")
            if path.is_dir()
        } if slots_root.is_dir() else set()
        orphaned = sorted(actual_directories - recorded_directories)
        temporary = sorted(
            path.relative_to(self.root).as_posix()
            for path in slots_root.rglob(".frame_*.tmp-*")
            if path.is_dir()
        ) if slots_root.is_dir() else []
        if orphaned or temporary:
            paths = ", ".join(orphaned + temporary)
            raise ValidationError(f"会话存在未入 manifest 的孤立/临时帧目录，拒绝自动删除：{paths}")

    def _archive_uncommitted_directories(self) -> None:
        """Preserve crash-window directories under unique orphan names."""

        slots_root = self.root / "slots"
        if not slots_root.is_dir():
            return
        recorded = {
            str(frame.get("relative_dir"))
            for slot in SLOT_ORDER
            for attempt in self.data["slots"][slot]["attempts"]
            for frame in attempt.get("frames", [])
        }
        candidates: list[tuple[Path, str]] = []
        for attempt_dir in slots_root.glob("*/attempt_*"):
            if not attempt_dir.is_dir():
                continue
            for path in attempt_dir.glob("frame_*"):
                if path.is_dir() and path.relative_to(self.root).as_posix() not in recorded:
                    candidates.append((path, "orphaned_capture"))
            for path in attempt_dir.glob(".frame_*.tmp-*"):
                if path.is_dir():
                    candidates.append((path, "orphaned_partial_capture"))
            for path in attempt_dir.glob(".evaluation*.tmp-*"):
                if path.is_dir():
                    candidates.append((path, "orphaned_partial_evaluation"))
        for source, prefix in candidates:
            destination = source.parent / (
                f"{prefix}_{dt.datetime.now(BEIJING):%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
            )
            os.rename(source, destination)
            _fsync_directory(source.parent)
            self.audit(
                "uncommitted_directory_archived",
                {
                    "source_name": source.name,
                    "archived_name": destination.name,
                    "reason": prefix,
                },
            )

    def _recover_committed_evaluations(self) -> None:
        """Finish an attempt-level evaluation transaction after a crash."""

        for slot in SLOT_ORDER:
            for attempt in self.data["slots"][slot]["attempts"]:
                attempt_number = int(attempt["number"])
                attempt_dir = self.root / "slots" / slot / f"attempt_{attempt_number:02d}"
                evaluation_dir = attempt_dir / "evaluation"
                if not evaluation_dir.is_dir():
                    continue
                transaction_path = evaluation_dir / "transaction.json"
                if not transaction_path.is_file():
                    raise ValidationError(f"已提交 evaluation 缺少 transaction.json：{evaluation_dir}")
                try:
                    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ValidationError(f"evaluation transaction 解析失败：{evaluation_dir}") from exc
                if (
                    not isinstance(transaction, dict)
                    or transaction.get("session_id") != self.data["session_id"]
                    or transaction.get("candidate_sha256") != self.engine.candidate_sha256
                    or transaction.get("slot") != slot
                    or int(transaction.get("attempt", -1)) != attempt_number
                ):
                    raise ValidationError(f"evaluation transaction 来源不匹配：{evaluation_dir}")
                transaction_frames = transaction.get("frames", [])
                if not isinstance(transaction_frames, list) or len(transaction_frames) != 3:
                    raise ValidationError("evaluation transaction 必须恰好包含三帧")
                selected = max(
                    attempt["frames"], key=lambda frame: float(frame.get("sharpness_score", 0.0))
                )
                if int(transaction.get("roi_source_frame", -1)) != int(selected["index"]):
                    raise ValidationError("evaluation transaction ROI 源帧不是三帧中清晰度最高帧")
                needs_manifest_recovery = all(
                    frame.get("status") == "awaiting_roi" for frame in attempt["frames"]
                )
                if not needs_manifest_recovery and not all(
                    frame.get("status") in {"passed", "failed"} for frame in attempt["frames"]
                ):
                    raise ValidationError("evaluation transaction 与 manifest 处于混合状态")
                for entry in transaction_frames:
                    index = int(entry.get("index", -1))
                    frame = next(
                        (item for item in attempt["frames"] if int(item["index"]) == index),
                        None,
                    )
                    if frame is None or entry.get("combined_pixel_sha256") != frame.get("combined_pixel_sha256"):
                        raise ValidationError("evaluation transaction 帧索引/图像 SHA256 不匹配")
                    staged_frame = evaluation_dir / f"frame_{index:02d}"
                    file_hashes = entry.get("file_sha256", {})
                    if not isinstance(file_hashes, Mapping):
                        raise ValidationError("evaluation transaction 缺少结果文件 SHA256")
                    frame_dir = self.root / frame["relative_dir"]
                    for filename, expected_hash in file_hashes.items():
                        source = staged_frame / str(filename)
                        destination = frame_dir / str(filename)
                        if not source.is_file() or sha256_file(source) != str(expected_hash):
                            raise ValidationError(f"evaluation 已提交文件缺失/损坏：{source}")
                        if destination.exists():
                            if sha256_file(destination) != str(expected_hash):
                                raise ValidationError(f"evaluation 目标文件 SHA256 冲突：{destination}")
                        else:
                            os.link(source, destination)
                    _fsync_directory(frame_dir)
                    if needs_manifest_recovery:
                        metrics = entry.get("metrics")
                        if not isinstance(metrics, Mapping):
                            raise ValidationError("evaluation transaction 缺少帧 metrics")
                        frame["file_sha256"].update(dict(file_hashes))
                        frame["status"] = str(entry["status"])
                        frame["roi"] = copy.deepcopy(metrics["roi"])
                        frame["metrics"] = copy.deepcopy(dict(metrics))
                        frame["evaluated_at"] = transaction["evaluated_at"]
                if needs_manifest_recovery:
                    self._complete_attempt(slot, attempt)
                    self._save_manifest()
                    self.audit(
                        "evaluation_transaction_recovered",
                        {
                            "slot": slot,
                            "attempt": attempt_number,
                            "roi_source_frame": int(selected["index"]),
                            "shared_roi": transaction.get("shared_roi"),
                            **attempt["summary"],
                        },
                    )

    def _save_manifest(self) -> None:
        self.data["updated_at"] = now_beijing()
        atomic_replace_bytes(self.manifest_path, json_bytes(self.data))

    def audit(self, kind: str, details: Mapping[str, Any]) -> None:
        with self._lock:
            event: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": now_beijing(),
                "session_id": self.data["session_id"],
                "event": str(kind),
                "details": copy.deepcopy(dict(details)),
                "previous_event_sha256": self._audit_tail,
            }
            canonical = json.dumps(
                event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            event_hash = sha256_bytes(canonical)
            event["event_sha256"] = event_hash
            line = (
                json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            descriptor = os.open(str(self.audit_path), os.O_WRONLY | os.O_APPEND)
            try:
                view = memoryview(line)
                written = 0
                while written < len(view):
                    count = os.write(descriptor, view[written:])
                    if count <= 0:
                        raise OSError("audit.jsonl append returned a zero-length write")
                    written += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._audit_tail = event_hash

    def _assert_candidate_unchanged(self) -> None:
        if sha256_file(self.engine.candidate_path) != self.engine.candidate_sha256:
            raise ValidationError("候选 YAML 在验证期间发生变化，立即停止")
        if sha256_file(self.pristine_path) != self.engine.candidate_sha256:
            raise ValidationError("pristine 候选副本在验证期间发生变化")

    def current_attempt(self, slot: str, *, create: bool = False) -> dict[str, Any]:
        slot_data = self._slot(slot)
        attempts = slot_data["attempts"]
        if not attempts:
            if not create:
                raise ValidationError(f"{slot} 尚未开始")
            attempts.append(
                {
                    "number": 1,
                    "status": "in_progress",
                    "known_distance_m": None,
                    "created_at": now_beijing(),
                    "completed_at": None,
                    "frames": [],
                    "summary": None,
                }
            )
            self._save_manifest()
            self.audit("attempt_started", {"slot": slot, "attempt": 1})
        return attempts[-1]

    def _slot(self, slot: str) -> dict[str, Any]:
        if slot not in SLOT_ORDER:
            raise ValidationError("距离槽必须是 1m、3m 或 5m")
        value = self.data["slots"][slot]
        if not isinstance(value, dict):
            raise ValidationError(f"{slot} 会话数据异常")
        return value

    @staticmethod
    def _validate_known_distance(slot: str, known_mm: Any) -> float:
        millimetres = finite_float(known_mm, "实测距离(mm)")
        known_m = millimetres / 1000.0
        ranges = {"1m": (0.8, 1.2), "3m": (2.5, 3.5), "5m": (4.3, 5.7)}
        lower, upper = ranges[slot]
        if not lower <= known_m <= upper:
            raise ValidationError(
                f"{slot} 槽实测值应在 {lower:.1f}–{upper:.1f} m，当前为 {known_m:.3f} m"
            )
        return known_m

    def can_capture(
        self, slot: str, known_mm: Any, *, internal_batch: bool = False
    ) -> tuple[dict[str, Any], float]:
        with self._lock:
            self._assert_candidate_unchanged()
            if self.data["status"] != "in_progress":
                raise ValidationError("会话当前不可采集")
            slot_data = self._slot(slot)
            if bool(slot_data["locked"]):
                raise ValidationError(f"{slot} 已锁定，不能再采集")
            first_unlocked = next(
                (name for name in SLOT_ORDER if not self.data["slots"][name]["locked"]),
                None,
            )
            if slot != first_unlocked:
                raise ValidationError(
                    f"必须按 1m → 3m → 5m 顺序验证；当前应先完成 {first_unlocked}"
                )
            attempt = self.current_attempt(slot, create=True)
            if attempt["status"] == "failed":
                raise ValidationError(f"{slot} 本轮已失败，请点击“新建重试”")
            if attempt["status"] != "in_progress":
                raise ValidationError(f"{slot} 当前尝试状态不可采集")
            frames = attempt["frames"]
            if len(frames) >= FRAMES_PER_ATTEMPT:
                raise ValidationError(f"{slot} 本轮已采集 3 帧")
            if not internal_batch and frames and any(
                frame.get("status") != "awaiting_roi" for frame in frames
            ):
                raise ValidationError("本轮历史帧状态不一致，拒绝继续采集")
            known_m = self._validate_known_distance(slot, known_mm)
            existing = attempt.get("known_distance_m")
            if existing is None:
                attempt["known_distance_m"] = known_m
                self._save_manifest()
            elif abs(float(existing) - known_m) > 0.0005:
                raise ValidationError(
                    f"本轮已固定实测距离 {float(existing)*1000:.1f} mm，三帧必须使用同一值"
                )
            return attempt, known_m

    @staticmethod
    def _pixel_hash(image: np.ndarray) -> str:
        header = f"{image.dtype}|{image.shape}".encode("ascii")
        return sha256_bytes(header + image.tobytes(order="C"))

    def all_combined_hashes(self) -> set[str]:
        hashes: set[str] = set()
        for slot in SLOT_ORDER:
            for attempt in self.data["slots"][slot]["attempts"]:
                for frame in attempt["frames"]:
                    value = str(frame.get("combined_pixel_sha256") or "")
                    if value:
                        hashes.add(value)
        return hashes

    def save_capture(
        self, slot: str, known_mm: Any, stereo_frame: StereoFrame
    ) -> dict[str, Any]:
        with self._lock:
            attempt, known_m = self.can_capture(slot, known_mm, internal_batch=True)
            pixel_hash = self._pixel_hash(stereo_frame.combined)
            if pixel_hash in self.all_combined_hashes():
                raise ValidationError("当前同步帧与历史帧完全相同，拒绝保存；请检查相机是否停帧")
            frame_index = len(attempt["frames"]) + 1
            attempt_number = int(attempt["number"])
            relative_dir = Path("slots") / slot / f"attempt_{attempt_number:02d}" / f"frame_{frame_index:02d}"
            frame_dir = self.root / relative_dir
            if frame_dir.exists():
                raise ValidationError(f"帧目录已存在，拒绝覆盖：{frame_dir}")
            frame_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".frame_{frame_index:02d}.tmp-", dir=str(frame_dir.parent)
                )
            )
            try:
                file_hashes: dict[str, str] = {}
                images = {
                    "combined.png": stereo_frame.combined,
                    "physical_left.png": stereo_frame.left,
                    "physical_right.png": stereo_frame.right,
                    "rectified_left.png": stereo_frame.left_rectified,
                    "rectified_right.png": stereo_frame.right_rectified,
                }
                for filename, image in images.items():
                    file_hashes[filename] = write_png_exclusive(temporary_dir / filename, image)
                captured_at = now_beijing()
                capture_document = {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": self.data["session_id"],
                    "slot": slot,
                    "attempt": attempt_number,
                    "frame": frame_index,
                    "captured_at": captured_at,
                    "known_distance_m": known_m,
                    "ros_frame_sequence": stereo_frame.sequence,
                    "candidate_sha256": self.engine.candidate_sha256,
                    "combined_pixel_sha256": pixel_hash,
                    "physical_left_view": self.engine.physical_left_view,
                    "physical_right_view": self.engine.physical_right_view,
                    "runtime_rotation": self.engine.rotation,
                }
                atomic_create_bytes(temporary_dir / "capture.json", json_bytes(capture_document))
                file_hashes["capture.json"] = sha256_file(temporary_dir / "capture.json")
                hash_document = {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_sha256": self.engine.candidate_sha256,
                    "combined_pixel_sha256": pixel_hash,
                    "files": file_hashes,
                }
                atomic_create_bytes(temporary_dir / "hashes.json", json_bytes(hash_document))
                file_hashes["hashes.json"] = sha256_file(temporary_dir / "hashes.json")
                _fsync_directory(temporary_dir)
                os.rename(temporary_dir, frame_dir)
                _fsync_directory(frame_dir.parent)
            except Exception:
                if temporary_dir.exists():
                    shutil.rmtree(temporary_dir)
                raise
            record = {
                "index": frame_index,
                "status": "awaiting_roi",
                "captured_at": captured_at,
                "known_distance_m": known_m,
                "relative_dir": relative_dir.as_posix(),
                "ros_frame_sequence": stereo_frame.sequence,
                "candidate_sha256": self.engine.candidate_sha256,
                "combined_pixel_sha256": pixel_hash,
                "sharpness_score": float(
                    cv2.Laplacian(
                        cv2.cvtColor(stereo_frame.left_rectified, cv2.COLOR_BGR2GRAY),
                        cv2.CV_64F,
                    ).var()
                ),
                "file_sha256": file_hashes,
                "roi": None,
                "metrics": None,
            }
            attempt["frames"].append(record)
            self._save_manifest()
            self.audit(
                "frame_captured",
                {
                    "slot": slot,
                    "attempt": attempt_number,
                    "frame": frame_index,
                    "known_distance_m": known_m,
                    "combined_pixel_sha256": pixel_hash,
                },
            )
            return copy.deepcopy(record)

    def pending_attempt(
        self,
    ) -> Optional[tuple[str, dict[str, Any], dict[str, Any]]]:
        with self._lock:
            for slot in SLOT_ORDER:
                attempts = self.data["slots"][slot]["attempts"]
                if not attempts:
                    continue
                attempt = attempts[-1]
                frames = attempt["frames"]
                if len(frames) == FRAMES_PER_ATTEMPT and all(
                    frame.get("status") == "awaiting_roi" for frame in frames
                ):
                    selected = max(
                        frames, key=lambda frame: float(frame.get("sharpness_score", 0.0))
                    )
                    return slot, attempt, selected
            return None

    def submit_roi(
        self, slot: str, roi: tuple[int, int, int, int]
    ) -> dict[str, Any]:
        with self._lock:
            self._assert_candidate_unchanged()
            pending = self.pending_attempt()
            if pending is None:
                raise ValidationError("当前没有已集齐三帧且等待 ROI 的尝试")
            pending_slot, attempt, selected_frame = pending
            if slot != pending_slot:
                raise ValidationError(f"当前必须先处理 {pending_slot} 槽的三帧尝试")

            evaluated: list[tuple[dict[str, Any], Path, dict[str, Any], np.ndarray, np.ndarray]] = []
            for frame in attempt["frames"]:
                frame_dir = self.root / frame["relative_dir"]
                if (frame_dir / "result.json").exists():
                    raise ValidationError("该轮 ROI 已提交过，不允许反复换框")
                left_rect = cv2.imread(str(frame_dir / "rectified_left.png"), cv2.IMREAD_COLOR)
                right_rect = cv2.imread(str(frame_dir / "rectified_right.png"), cv2.IMREAD_COLOR)
                if left_rect is None or right_rect is None:
                    raise ValidationError("无法读取待评估的校正图")
                metrics, annotated, disparity = self.engine.evaluate(
                    left_rect,
                    right_rect,
                    roi,
                    float(frame["known_distance_m"]),
                )
                evaluated.append((frame, frame_dir, metrics, annotated, disparity))

            evaluated_at = now_beijing()
            attempt_dir = (self.root / attempt["frames"][0]["relative_dir"]).parent
            evaluation_dir = attempt_dir / "evaluation"
            if evaluation_dir.exists():
                raise ValidationError(
                    "本轮存在已提交的 evaluation 事务；请安全结束并用 --resume 自动恢复"
                )
            staging = Path(
                tempfile.mkdtemp(prefix=".evaluation.tmp-", dir=str(attempt_dir))
            )
            transaction_frames: list[dict[str, Any]] = []
            try:
                for frame, _frame_dir, metrics, annotated, disparity in evaluated:
                    staged_frame = staging / f"frame_{int(frame['index']):02d}"
                    staged_frame.mkdir()
                    added = {
                        "annotated.png": write_png_exclusive(staged_frame / "annotated.png", annotated),
                        "disparity.png": write_png_exclusive(staged_frame / "disparity.png", disparity),
                    }
                    result_document = {
                        "schema_version": SCHEMA_VERSION,
                        "session_id": self.data["session_id"],
                        "slot": slot,
                        "attempt": int(attempt["number"]),
                        "frame": int(frame["index"]),
                        "evaluated_at": evaluated_at,
                        "candidate_sha256": self.engine.candidate_sha256,
                        "combined_pixel_sha256": frame["combined_pixel_sha256"],
                        "roi_source_frame": int(selected_frame["index"]),
                        "shared_roi_for_three_frames": True,
                        "metrics": metrics,
                    }
                    atomic_create_bytes(staged_frame / "result.json", json_bytes(result_document))
                    added["result.json"] = sha256_file(staged_frame / "result.json")
                    transaction_frames.append(
                        {
                            "index": int(frame["index"]),
                            "combined_pixel_sha256": frame["combined_pixel_sha256"],
                            "status": "passed" if metrics["passed"] else "failed",
                            "metrics": metrics,
                            "file_sha256": added,
                        }
                    )
                    _fsync_directory(staged_frame)
                transaction = {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": self.data["session_id"],
                    "slot": slot,
                    "attempt": int(attempt["number"]),
                    "evaluated_at": evaluated_at,
                    "candidate_sha256": self.engine.candidate_sha256,
                    "shared_roi": evaluated[0][2]["roi"],
                    "roi_source_frame": int(selected_frame["index"]),
                    "frames": transaction_frames,
                }
                atomic_create_bytes(staging / "transaction.json", json_bytes(transaction))
                _fsync_directory(staging)
                os.rename(staging, evaluation_dir)
                _fsync_directory(attempt_dir)
            except Exception:
                if staging.exists():
                    # Preserve, never delete, a partially staged evaluation.
                    orphan = attempt_dir / (
                        f"orphaned_partial_evaluation_{dt.datetime.now(BEIJING):%Y%m%d_%H%M%S}_"
                        f"{secrets.token_hex(3)}"
                    )
                    os.rename(staging, orphan)
                    _fsync_directory(attempt_dir)
                raise

            for transaction_frame in transaction_frames:
                frame = next(
                    item for item in attempt["frames"] if int(item["index"]) == int(transaction_frame["index"])
                )
                frame_dir = self.root / frame["relative_dir"]
                staged_frame = evaluation_dir / f"frame_{int(frame['index']):02d}"
                for filename, expected_hash in transaction_frame["file_sha256"].items():
                    source = staged_frame / filename
                    destination = frame_dir / filename
                    if destination.exists():
                        if sha256_file(destination) != expected_hash:
                            raise ValidationError(f"已有评估文件 SHA256 冲突：{destination}")
                    else:
                        os.link(source, destination)
                _fsync_directory(frame_dir)
                frame["file_sha256"].update(transaction_frame["file_sha256"])
                frame["status"] = transaction_frame["status"]
                frame["roi"] = transaction_frame["metrics"]["roi"]
                frame["metrics"] = transaction_frame["metrics"]
                frame["evaluated_at"] = evaluated_at

            self._complete_attempt(slot, attempt)
            self._save_manifest()
            for frame, _frame_dir, metrics, _annotated, _disparity in evaluated:
                self.audit(
                    "frame_evaluated",
                    {
                        "slot": slot,
                        "attempt": int(attempt["number"]),
                        "frame": int(frame["index"]),
                        "shared_roi": metrics["roi"],
                        "roi_source_frame": int(selected_frame["index"]),
                        "passed": bool(metrics["passed"]),
                        "estimated_distance_m": metrics["estimated_distance_m"],
                        "relative_error": metrics["relative_error"],
                        "valid_pixels": metrics["valid_pixels"],
                        "valid_fraction": metrics["valid_fraction"],
                        "relative_mad": metrics["relative_mad"],
                        "central80_spread_over_median": metrics[
                            "central80_spread_over_median"
                        ],
                    },
                )
            self.audit(
                "attempt_completed",
                {
                    "slot": slot,
                    "attempt": int(attempt["number"]),
                    "shared_roi": evaluated[0][2]["roi"],
                    "roi_source_frame": int(selected_frame["index"]),
                    **attempt["summary"],
                },
            )
            return {
                "frames": copy.deepcopy(attempt["frames"]),
                "attempt_status": attempt["status"],
                "attempt_summary": copy.deepcopy(attempt.get("summary")),
                "slot_locked": bool(self.data["slots"][slot]["locked"]),
                "all_locked": self.all_slots_locked(),
            }

    def _complete_attempt(self, slot: str, attempt: dict[str, Any]) -> None:
        passed_frames = [frame for frame in attempt["frames"] if frame["metrics"]["passed"]]
        estimates = [float(frame["metrics"]["estimated_distance_m"]) for frame in passed_frames]
        span_ratio: Optional[float] = None
        if estimates:
            median_estimate = float(np.median(estimates))
            span_ratio = (max(estimates) - min(estimates)) / max(median_estimate, 1e-9)
        passed = (
            len(passed_frames) >= MIN_PASSING_FRAMES
            and span_ratio is not None
            and span_ratio <= MAX_PASSING_ESTIMATE_SPAN
        )
        summary = {
            "passed": bool(passed),
            "passing_frames": len(passed_frames),
            "total_frames": FRAMES_PER_ATTEMPT,
            "passing_estimate_span_over_median": span_ratio,
            "maximum_allowed_span_over_median": MAX_PASSING_ESTIMATE_SPAN,
            "failure_reason": None
            if passed
            else (
                "fewer_than_two_passing_frames"
                if len(passed_frames) < MIN_PASSING_FRAMES
                else "passing_frame_estimates_inconsistent"
            ),
        }
        attempt["summary"] = summary
        attempt["completed_at"] = now_beijing()
        attempt["status"] = "locked" if passed else "failed"
        slot_data = self.data["slots"][slot]
        if passed:
            known = float(attempt["known_distance_m"])
            estimate = float(np.median(estimates))
            aggregate = {
                "slot": slot,
                "nominal_distance_m": SLOT_NOMINAL_M[slot],
                "known_distance_m": known,
                "estimated_distance_m": estimate,
                "relative_error": abs(estimate - known) / known,
                "passed": True,
                "passing_frames": len(passed_frames),
                "total_frames": FRAMES_PER_ATTEMPT,
                "passing_estimate_span_over_median": span_ratio,
                "minimum_valid_pixels": min(
                    int(frame["metrics"]["valid_pixels"]) for frame in passed_frames
                ),
                "minimum_valid_fraction": min(
                    float(frame["metrics"]["valid_fraction"]) for frame in passed_frames
                ),
                "maximum_relative_mad": max(
                    float(frame["metrics"]["relative_mad"]) for frame in passed_frames
                ),
                "maximum_central80_spread_over_median": max(
                    float(frame["metrics"]["central80_spread_over_median"])
                    for frame in passed_frames
                ),
                "combined_image_sha256": [
                    frame["combined_pixel_sha256"] for frame in attempt["frames"]
                ],
                "frame_result_sha256": [
                    frame["file_sha256"]["result.json"] for frame in attempt["frames"]
                ],
                "session_relative_path": (
                    Path("slots") / slot / f"attempt_{int(attempt['number']):02d}"
                ).as_posix(),
            }
            slot_data["locked"] = True
            slot_data["locked_attempt"] = int(attempt["number"])
            slot_data["aggregate"] = aggregate

    def retry_slot(self, slot: str) -> int:
        with self._lock:
            slot_data = self._slot(slot)
            if slot_data["locked"]:
                raise ValidationError(f"{slot} 已锁定，不允许重试")
            attempts = slot_data["attempts"]
            if not attempts or attempts[-1]["status"] != "failed":
                raise ValidationError("只有完成且失败的三帧尝试才能新建重试")
            number = len(attempts) + 1
            attempts.append(
                {
                    "number": number,
                    "status": "in_progress",
                    "known_distance_m": None,
                    "created_at": now_beijing(),
                    "completed_at": None,
                    "frames": [],
                    "summary": None,
                }
            )
            self._save_manifest()
            self.audit("attempt_started", {"slot": slot, "attempt": number, "retry": True})
            return number

    def all_slots_locked(self) -> bool:
        return all(bool(self.data["slots"][slot]["locked"]) for slot in SLOT_ORDER)

    def _validate_final_invariants(self) -> list[dict[str, Any]]:
        self._assert_candidate_unchanged()
        if not self.all_slots_locked():
            raise ValidationError("三个距离槽尚未全部锁定")
        aggregates = [copy.deepcopy(self.data["slots"][slot]["aggregate"]) for slot in SLOT_ORDER]
        if len(aggregates) != 3 or not all(item and item.get("passed") for item in aggregates):
            raise ValidationError("最终聚合结果不是三个严格通过点")
        known = sorted(float(item["known_distance_m"]) for item in aggregates)
        for first, second in zip(known, known[1:]):
            separation = abs(second - first) / max(first, second)
            if separation < MIN_DISTANCE_SEPARATION:
                raise ValidationError("相邻实测距离相差小于 5%")
        hashes: list[str] = []
        candidate_hashes: set[str] = set()
        for slot in SLOT_ORDER:
            slot_data = self.data["slots"][slot]
            attempt_number = int(slot_data["locked_attempt"])
            attempt = next(
                item for item in slot_data["attempts"] if int(item["number"]) == attempt_number
            )
            if len(attempt["frames"]) != FRAMES_PER_ATTEMPT:
                raise ValidationError(f"{slot} 锁定尝试不是三帧")
            hashes.extend(str(frame["combined_pixel_sha256"]) for frame in attempt["frames"])
            candidate_hashes.update(str(frame["candidate_sha256"]) for frame in attempt["frames"])
        if len(hashes) != 9 or len(set(hashes)) != 9:
            raise ValidationError("最终九帧同步图像 SHA256 必须全部不同")
        if candidate_hashes != {self.engine.candidate_sha256}:
            raise ValidationError("最终九帧不是基于同一候选标定")
        validation = self.engine.pristine_payload.get("validation", {})
        if not isinstance(validation, Mapping):
            raise ValidationError("候选 YAML 缺少 validation 指标")
        if finite_float(validation.get("stereo_rms_px"), "stereo_rms_px") > 1.0:
            raise ValidationError("候选双目 RMS 大于 1.0 px")
        if finite_float(
            validation.get("median_epipolar_error_px"), "median_epipolar_error_px"
        ) > 1.0:
            raise ValidationError("候选中位极线误差大于 1.0 px")
        return aggregates

    def _test_fault(self, point: str) -> None:
        """Deterministic in-process crash hook used only by offline self-tests."""

        if getattr(self, "_test_fault_at", None) == point:
            self._test_fault_at = None
            raise RuntimeError(f"TEST_CRASH_AT:{point}")

    def _ensure_prevalidation_snapshots(self) -> tuple[bytes, bytes]:
        """Create/reuse immutable snapshots without requiring byte equality to a later manifest."""

        current_manifest = self.manifest_path.read_bytes()
        if self.manifest_before_validation_path.is_file():
            manifest_bytes = self.manifest_before_validation_path.read_bytes()
            try:
                snapshot = json.loads(manifest_bytes.decode("utf-8"))
            except Exception as exc:
                raise ValidationError("session_before_validation.json 无法解析") from exc
            if (
                not isinstance(snapshot, Mapping)
                or snapshot.get("schema_version") != SCHEMA_VERSION
                or str(snapshot.get("session_id")) != self.data["session_id"]
                or str(snapshot.get("candidate_sha256")) != self.engine.candidate_sha256
                or snapshot.get("slots") != self.data.get("slots")
            ):
                raise ValidationError("已有 session_before_validation.json 与当前会话不一致")
        else:
            manifest_bytes = current_manifest
            atomic_create_bytes(self.manifest_before_validation_path, manifest_bytes)
        self._test_fault("after_manifest_snapshot")

        current_audit = self.audit_path.read_bytes()
        if self.audit_before_validation_path.is_file():
            audit_bytes = self.audit_before_validation_path.read_bytes()
            if not current_audit.startswith(audit_bytes):
                raise ValidationError("已有 audit_before_validation.jsonl 不是当前审计链前缀")
        else:
            audit_bytes = current_audit
            atomic_create_bytes(self.audit_before_validation_path, audit_bytes)
        self._test_fault("after_audit_snapshot")
        return manifest_bytes, audit_bytes

    def _build_validated_yaml(
        self,
        aggregates: list[dict[str, Any]],
        *,
        manifest_hash_before: str,
        audit_hash_before: str,
        validated_at: str,
    ) -> bytes:
        payload = copy.deepcopy(self.engine.pristine_payload)
        nested = payload.get("calibration")
        target = nested if isinstance(nested, dict) else payload
        target["valid"] = True
        if target is not payload:
            payload["valid"] = True
        validation = target.setdefault("validation", {})
        if not isinstance(validation, dict):
            raise ValidationError("候选 validation 不是映射，拒绝生成")
        validation["known_distance_results"] = aggregates
        validation["distance_validation"] = {
            "schema_version": SCHEMA_VERSION,
            "validated_at": validated_at,
            "session_id": self.data["session_id"],
            "source_candidate_sha256": self.engine.candidate_sha256,
            "pristine_candidate_filename": self.pristine_path.name,
            "session_manifest_filename": self.manifest_path.name,
            "session_manifest_sha256_before_validation": manifest_hash_before,
            "session_manifest_before_validation_filename": self.manifest_before_validation_path.name,
            "audit_log_filename": self.audit_path.name,
            "audit_log_sha256_before_validation": audit_hash_before,
            "audit_log_before_validation_filename": self.audit_before_validation_path.name,
            "validator_script_sha256": sha256_file(self.validator_archive_path),
            "validator_script_filename": self.validator_archive_path.name,
            "runtime_source_archives": {
                key: {
                    "file": self.runtime_archive_paths[key].name,
                    "sha256": sha256_file(self.runtime_archive_paths[key]),
                }
                for key in ("depth", "calibration", "gs130w")
            },
            "thresholds": self.thresholds(),
            "note": (
                "1m/3m/5m three-frame strict validation passed; candidate remained "
                "immutable and valid:true was created only in this new file."
            ),
        }
        return require_yaml().safe_dump(
            payload, allow_unicode=True, sort_keys=False, width=120
        ).encode("utf-8")

    def _read_finalize_transaction(
        self,
    ) -> tuple[dict[str, Any], bytes, str]:
        transaction_bytes = self.finalize_transaction_path.read_bytes()
        transaction_hash = sha256_bytes(transaction_bytes)
        try:
            transaction = json.loads(transaction_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValidationError("finalize_transaction.json 无法解析") from exc
        if not isinstance(transaction, dict):
            raise ValidationError("finalize_transaction.json 顶层不是映射")
        if transaction.get("transaction_schema") != "gs130w-finalize-transaction/v1":
            raise ValidationError("最终化事务 schema 不匹配")
        if (
            str(transaction.get("session_id")) != self.data["session_id"]
            or str(transaction.get("candidate_sha256")) != self.engine.candidate_sha256
            or str(transaction.get("validated_output_file")) != self.validated_output.name
        ):
            raise ValidationError("最终化事务来源与当前会话不匹配")
        manifest_transaction_hash = str(self.data.get("finalize_transaction_sha256") or "")
        if manifest_transaction_hash and manifest_transaction_hash != transaction_hash:
            raise ValidationError("manifest 记录的最终化事务 SHA256 不匹配")

        snapshots = transaction.get("snapshots")
        if not isinstance(snapshots, Mapping):
            raise ValidationError("最终化事务缺少快照记录")
        for key, path in (
            ("manifest_before_validation", self.manifest_before_validation_path),
            ("audit_before_validation", self.audit_before_validation_path),
        ):
            record = snapshots.get(key)
            if (
                not isinstance(record, Mapping)
                or str(record.get("file")) != path.name
                or not path.is_file()
                or sha256_file(path) != str(record.get("sha256"))
            ):
                raise ValidationError(f"最终化事务的 {key} 快照缺失或 SHA256 错误")

        runtime_records = transaction.get("runtime_source_archives")
        if not isinstance(runtime_records, Mapping) or set(runtime_records) != {
            "depth",
            "calibration",
            "gs130w",
        }:
            raise ValidationError("最终化事务缺少运行时源码记录")
        for key, path in self.runtime_archive_paths.items():
            record = runtime_records[key]
            if (
                not isinstance(record, Mapping)
                or str(record.get("file")) != path.name
                or str(record.get("sha256")) != sha256_file(path)
            ):
                raise ValidationError(f"最终化事务的 {key} 运行时源码记录错误")

        yaml_text = transaction.get("validated_yaml_utf8")
        if not isinstance(yaml_text, str):
            raise ValidationError("最终化事务没有完整 validated YAML payload")
        encoded = yaml_text.encode("utf-8")
        planned_hash = str(transaction.get("validated_yaml_sha256") or "")
        if sha256_bytes(encoded) != planned_hash:
            raise ValidationError("最终化事务内 YAML payload 的 SHA256 错误")
        aggregates = self._validate_final_invariants()
        expected = self._build_validated_yaml(
            aggregates,
            manifest_hash_before=str(snapshots["manifest_before_validation"]["sha256"]),
            audit_hash_before=str(snapshots["audit_before_validation"]["sha256"]),
            validated_at=str(transaction.get("validated_at") or ""),
        )
        if expected != encoded:
            raise ValidationError("最终化事务内 YAML 无法由会话数据确定性重建")
        return transaction, encoded, transaction_hash

    def _create_finalize_transaction(
        self, aggregates: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], bytes, str]:
        manifest_before, audit_before = self._ensure_prevalidation_snapshots()
        validated_at = now_beijing()
        encoded = self._build_validated_yaml(
            aggregates,
            manifest_hash_before=sha256_bytes(manifest_before),
            audit_hash_before=sha256_bytes(audit_before),
            validated_at=validated_at,
        )
        transaction = {
            "transaction_schema": "gs130w-finalize-transaction/v1",
            "session_id": self.data["session_id"],
            "candidate_sha256": self.engine.candidate_sha256,
            "validated_output_file": self.validated_output.name,
            "validated_at": validated_at,
            "validator_script": {
                "file": self.validator_archive_path.name,
                "sha256": sha256_file(self.validator_archive_path),
            },
            "runtime_source_archives": {
                key: {
                    "file": self.runtime_archive_paths[key].name,
                    "sha256": sha256_file(self.runtime_archive_paths[key]),
                }
                for key in ("depth", "calibration", "gs130w")
            },
            "snapshots": {
                "manifest_before_validation": {
                    "file": self.manifest_before_validation_path.name,
                    "sha256": sha256_bytes(manifest_before),
                },
                "audit_before_validation": {
                    "file": self.audit_before_validation_path.name,
                    "sha256": sha256_bytes(audit_before),
                },
            },
            "validated_yaml_sha256": sha256_bytes(encoded),
            "validated_yaml_utf8": encoded.decode("utf-8"),
        }
        atomic_create_bytes(self.finalize_transaction_path, json_bytes(transaction))
        self._test_fault("after_finalize_transaction")
        return self._read_finalize_transaction()

    def _commit_finalize_transaction(self, *, recovered: bool) -> Path:
        _transaction, encoded, transaction_hash = self._read_finalize_transaction()
        planned_hash = sha256_bytes(encoded)
        self.data["status"] = "finalizing"
        self.data["pending_validated_yaml_sha256"] = planned_hash
        self.data["finalize_transaction_file"] = self.finalize_transaction_path.name
        self.data["finalize_transaction_sha256"] = transaction_hash
        self._save_manifest()
        self._test_fault("after_finalizing_manifest")

        if self.validated_output.is_file():
            if sha256_file(self.validated_output) != planned_hash:
                raise ValidationError("中断后的 validated YAML 与最终化事务不一致")
        else:
            atomic_create_bytes(self.validated_output, encoded)
        self._test_fault("after_validated_yaml")

        self.data["status"] = "completed"
        self.data["validated_yaml_sha256"] = planned_hash
        self.data.pop("pending_validated_yaml_sha256", None)
        self.data["completion"] = {
            "completed_at": now_beijing(),
            "validated_output_file": self.validated_output.name,
            "validated_yaml_sha256": planned_hash,
            "source_candidate_sha256": self.engine.candidate_sha256,
            "recovered_after_interruption": bool(recovered),
        }
        self._save_manifest()
        self._test_fault("after_completed_manifest")
        self._ensure_final_audit_event(recovered=recovered)
        self._test_fault("after_final_audit")
        checksums = self.root / "SESSION_SHA256SUMS.txt"
        if checksums.exists():
            self._verify_session_checksums()
        else:
            self._write_session_checksums()
        return self.validated_output

    def _recover_finalize_transaction(self) -> Path:
        return self._commit_finalize_transaction(recovered=True)

    def finalize(self) -> Path:
        with self._lock:
            if self.data["status"] == "completed":
                return self.validated_output
            aggregates = self._validate_final_invariants()
            if self.validated_output.exists() and not self.finalize_transaction_path.is_file():
                raise ValidationError(f"最终 YAML 已存在但没有事务，拒绝覆盖：{self.validated_output}")
            if not self.finalize_transaction_path.is_file():
                self._create_finalize_transaction(aggregates)
            return self._commit_finalize_transaction(recovered=False)

    def _write_session_checksums(self) -> None:
        excluded = {"session.json", "audit.jsonl", "SESSION_SHA256SUMS.txt"}
        lines: list[str] = []
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            relative = path.relative_to(self.root).as_posix()
            if relative in excluded:
                continue
            lines.append(f"{sha256_file(path)}  {relative}")
        atomic_create_bytes(
            self.root / "SESSION_SHA256SUMS.txt", ("\n".join(lines) + "\n").encode("utf-8")
        )

    def _verify_session_checksums(self) -> None:
        checksum_path = self.root / "SESSION_SHA256SUMS.txt"
        try:
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            raise ValidationError("SESSION_SHA256SUMS.txt 无法读取") from exc
        if not lines:
            raise ValidationError("SESSION_SHA256SUMS.txt 为空")
        for line_number, line in enumerate(lines, 1):
            if "  " not in line:
                raise ValidationError(f"SESSION_SHA256SUMS.txt 第 {line_number} 行格式错误")
            expected, relative_text = line.split("  ", 1)
            relative = Path(relative_text)
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValidationError(f"SESSION_SHA256SUMS.txt 第 {line_number} 行 SHA256 错误")
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationError(f"SESSION_SHA256SUMS.txt 第 {line_number} 行路径不安全")
            path = self.root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValidationError(f"审计文件缺失或 SHA256 错误：{relative_text}")

    def audit_bundle(self) -> Path:
        """Create a read-only transport ZIP after completion, never before."""

        with self._lock:
            if self.data.get("status") != "completed":
                raise ValidationError("只有完成后才能下载完整审计包")
            bundle = self.root / "GS130W_distance_validation_audit.zip"
            if bundle.is_file():
                return bundle
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".audit_bundle.", suffix=".zip", dir=str(self.root)
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with zipfile.ZipFile(
                    temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
                ) as archive:
                    for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
                        if path == temporary or path == bundle:
                            continue
                        archive.write(path, path.relative_to(self.root).as_posix())
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, bundle)
                except FileExistsError:
                    pass
                _fsync_directory(self.root)
            finally:
                temporary.unlink(missing_ok=True)
            return bundle

    def pause(self, reason: str) -> None:
        with self._lock:
            if self.data.get("status") == "completed":
                return
            self.data["status"] = "paused"
            self.audit("session_paused", {"reason": str(reason)})
            self._save_manifest()

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            slots: dict[str, Any] = {}
            for slot in SLOT_ORDER:
                source = self.data["slots"][slot]
                attempts = source["attempts"]
                current = attempts[-1] if attempts else None
                frames = [] if current is None else current["frames"]
                slots[slot] = {
                    "nominal_distance_m": source["nominal_distance_m"],
                    "locked": bool(source["locked"]),
                    "locked_attempt": source["locked_attempt"],
                    "aggregate": copy.deepcopy(source["aggregate"]),
                    "attempt_number": None if current is None else current["number"],
                    "attempt_status": None if current is None else current["status"],
                    "known_distance_mm": None
                    if current is None or current["known_distance_m"] is None
                    else float(current["known_distance_m"]) * 1000.0,
                    "frames": [
                        {
                            "index": frame["index"],
                            "status": frame["status"],
                            "metrics": copy.deepcopy(frame.get("metrics")),
                        }
                        for frame in frames
                    ],
                    "summary": None if current is None else copy.deepcopy(current.get("summary")),
                    "can_retry": bool(current and current["status"] == "failed"),
                }
            pending = self.pending_attempt()
            return {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.data["session_id"],
                "status": self.data["status"],
                "candidate_sha256": self.engine.candidate_sha256,
                "validated_output": str(self.validated_output),
                "validated_yaml_sha256": self.data.get("validated_yaml_sha256"),
                "slots": slots,
                "pending_roi": None
                if pending is None
                else {
                    "slot": pending[0],
                    "attempt": pending[1]["number"],
                    "frame": pending[2]["index"],
                    "roi_source_frame": pending[2]["index"],
                    "shared_for_all_three_frames": True,
                    "image_width": self.engine.expected_size[0],
                    "image_height": self.engine.expected_size[1],
                },
                "thresholds": self.thresholds(),
            }


WEB_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GS130W 三点严格测距验证</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui,"Microsoft YaHei",sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #f3f4f6; background: #0b1020; }
    main { max-width: 1440px; margin: auto; padding: 16px; }
    h1 { margin: 0 0 10px; font-size: 25px; }
    h2 { margin: 0 0 10px; font-size: 19px; }
    .notice { background:#172554; border:1px solid #3b82f6; padding:12px; border-radius:10px; line-height:1.65; }
    .danger { color:#fecaca; }
    .grid { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(360px,1fr); gap:14px; margin-top:14px; }
    .card { background:#151d30; border:1px solid #334155; border-radius:11px; padding:12px; }
    .previews { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
    figure { margin:0; }
    figcaption { color:#bfdbfe; margin-bottom:5px; }
    figure img { display:block; width:100%; max-height:56vh; object-fit:contain; background:#020617; border-radius:7px; }
    .slot { border:1px solid #475569; border-radius:9px; padding:10px; margin-bottom:10px; }
    .slot.active { border-color:#38bdf8; box-shadow:0 0 0 1px #38bdf8 inset; }
    .slot.locked { border-color:#22c55e; background:#052e20; }
    .slot.failed { border-color:#ef4444; }
    .slot h3 { margin:0 0 8px; display:flex; justify-content:space-between; }
    input { width:150px; padding:9px; border-radius:7px; border:1px solid #64748b; background:#0f172a; color:white; font-size:17px; }
    button, a.button { padding:10px 12px; border:0; border-radius:8px; background:#0284c7; color:white; font-weight:700; cursor:pointer; text-decoration:none; display:inline-block; }
    button:disabled { background:#475569; color:#cbd5e1; cursor:not-allowed; }
    button.red { background:#b91c1c; }
    button.green, a.green { background:#15803d; }
    button.retry { background:#a16207; }
    .frames { display:flex; gap:6px; margin:8px 0; }
    .frame { flex:1; text-align:center; padding:6px 3px; border-radius:6px; background:#334155; font-size:13px; }
    .frame.pass { background:#166534; }
    .frame.fail { background:#991b1b; }
    .frame.wait { background:#854d0e; }
    .small { color:#cbd5e1; font-size:13px; line-height:1.55; }
    #message { min-height:48px; color:#fde68a; line-height:1.55; white-space:pre-wrap; }
    #roiPanel { display:none; margin-top:14px; }
    #canvasWrap { max-height:75vh; overflow:auto; background:#020617; border:1px solid #475569; position:relative; }
    #roiCanvas { display:block; cursor:crosshair; image-rendering:auto; }
    .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:8px 0; }
    .complete { background:#052e20; border:1px solid #22c55e; padding:12px; border-radius:9px; display:none; }
    code { overflow-wrap:anywhere; color:#bae6fd; }
    @media (max-width:950px) { .grid { grid-template-columns:1fr; } .previews { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <h1>GS130W 三点严格测距验证</h1>
  <div class="notice">
    <b>量距基准：</b>从<b>物理左镜头前表面中心</b>量到<b>靶纸中心标记</b>的空间直线距离；不要量地面水平距离、外壳边缘或支架。<br>
    摄像头和靶纸摆稳后，输入卷尺实测毫米数；每个距离点只点击一次，工具会自动取得 3 个新同步帧，再在质量最好的一帧上框一次 ROI，同一 ROI 自动严格评估三帧。<br>
    <span class="danger"><b>5 m 注意：</b>A3 靶理论上可能只约 47×33 px。若提示 ROI/有效像素不足，请换 A1 或更大随机纹理靶并重新做该轮，不能降低阈值。</span>
  </div>
  <div class="grid">
    <section>
      <div class="card">
        <h2>实时物理双目（仅预览，不自动保存）</h2>
        <div class="previews">
          <figure><figcaption>物理左目（量距基准镜头）</figcaption><img id="leftPreview" alt="等待物理左目画面"></figure>
          <figure><figcaption>物理右目</figcaption><img id="rightPreview" alt="等待物理右目画面"></figure>
        </div>
      </div>
      <div id="roiPanel" class="card">
        <h2>在校正后物理左目中框选随机纹理 ROI</h2>
        <p class="small">只框靶纸内部、中心标记附近且纹理丰富的区域；不要包含靶纸边缘、墙面、阴影或反光。一次提交后不可换框。可放大后精确拖动，服务端会反算并复核原始 640×1280 坐标。</p>
        <div class="toolbar">
          <label>显示倍率
            <select id="zoom"><option value="0.5">50%</option><option value="0.75">75%</option><option value="1" selected>100%</option><option value="1.5">150%</option><option value="2">200%</option></select>
          </label>
          <span id="roiText">尚未框选</span>
          <button id="submitRoi" disabled>提交本轮共用 ROI（仅一次）</button>
        </div>
        <div id="canvasWrap"><canvas id="roiCanvas"></canvas></div>
      </div>
    </section>
    <aside>
      <div class="card">
        <h2>固定顺序：1 m → 3 m → 5 m</h2>
        <div id="slots"></div>
        <div id="message">正在等待状态……</div>
        <div id="complete" class="complete">
          <b>三点严格验证完成，已生成新的 valid:true YAML。</b><br>
          Windows 启动器会自动下载并校验完整审计包，然后安全结束；以下链接用于手动备用下载。<br>
          <code id="validatedPath"></code><br><br>
          <a id="download" class="button green" href="/download/validated.yaml">下载 validated YAML</a>
          <a class="button" href="/download/session.json">下载 session.json</a>
          <a class="button" href="/download/audit.zip">下载完整审计包 ZIP</a>
          <button id="finish" class="green">下载确认后安全结束</button>
        </div>
        <button id="stop" class="red" style="margin-top:12px">安全暂停（稍后 --resume）</button>
        <p class="small">失败不会修改候选 YAML，也不会删除图片。该距离点完成三帧后可“新建重试”，旧尝试仍保留在审计目录。</p>
      </div>
    </aside>
  </div>
</main>
<script>
const slotsOrder = ["1m","3m","5m"];
const ranges = {"1m":[800,1200],"3m":[2500,3500],"5m":[4300,5700]};
const $ = id => document.getElementById(id);
let state = null, roiImage = null, roi = null, dragging = false, start = null, pendingKey = "";

function esc(value) { const d=document.createElement("div"); d.textContent=String(value??""); return d.innerHTML; }
function activeSlot() { if(!state) return null; return slotsOrder.find(s=>!state.slots[s].locked) || null; }
function metricText(f) {
  if (!f.metrics) return f.status === "awaiting_roi" ? "待框选" : "未采集";
  const m=f.metrics, estimate=m.estimated_distance_m==null?"不可用":Number(m.estimated_distance_m).toFixed(3)+"m";
  const error=m.relative_error==null?"n/a":(Number(m.relative_error)*100).toFixed(1)+"%";
  return estimate+" / 误差"+error;
}
function renderSlots() {
  const active=activeSlot();
  $("slots").innerHTML=slotsOrder.map(slot=>{
    const s=state.slots[slot], range=ranges[slot];
    const cls=s.locked?"locked":(s.attempt_status==="failed"?"failed":(slot===active?"active":""));
    const frames=[0,1,2].map(i=>{
      const f=s.frames[i]; if(!f) return `<div class="frame">帧${i+1}<br>未采集</div>`;
      const fc=f.status==="passed"?"pass":(f.status==="failed"?"fail":"wait");
      return `<div class="frame ${fc}">帧${i+1}<br>${esc(metricText(f))}</div>`;
    }).join("");
    const fixed=s.known_distance_mm!=null;
    const disabled=(slot!==active || s.locked || s.attempt_status==="failed" || state.pending_roi || state.capture_pending || !state.camera_fresh || state.status==="completed");
    const inputValue=fixed?Number(s.known_distance_mm).toFixed(1):"";
    const summary=s.summary ? (s.summary.passed?"本轮一致性通过":`本轮失败：${s.summary.failure_reason}`) : "";
    return `<div class="slot ${cls}"><h3><span>${slot} 距离槽</span><span>${s.locked?"已锁定":(slot===active?"当前":"等待")}</span></h3>
      <div><input id="mm_${slot}" type="number" min="${range[0]}" max="${range[1]}" step="0.1" value="${inputValue}" ${fixed?"disabled":""}> mm
      <button onclick="capture('${slot}')" ${disabled?"disabled":""}>${s.frames.length?"继续自动补齐三帧":"自动采集本轮三帧"}</button></div>
      <div class="small">允许输入 ${range[0]}–${range[1]} mm；一次点击自动抓取三帧并使用同一实测值。</div>
      <div class="frames">${frames}</div><div class="small">${esc(summary)}</div>
      ${s.can_retry?`<button class="retry" onclick="retrySlot('${slot}')">保留失败记录并新建重试</button>`:""}</div>`;
  }).join("");
}
async function api(path, body) {
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
  const result=await response.json();
  if(!response.ok) throw new Error(result.message||"请求失败");
  return result;
}
async function capture(slot) {
  try {
    const input=$("mm_"+slot); const mm=Number(input.value);
    if(!Number.isFinite(mm)) throw new Error("请先输入卷尺实测毫米数");
    const result=await api("/api/capture",{slot,known_mm:mm});
    $("message").textContent=result.message;
  } catch(e) { $("message").textContent="未采集："+e.message; }
  await updateStatus();
}
async function retrySlot(slot) {
  if(!confirm("确认新建该距离点的三帧重试？旧图和失败审计会完整保留。")) return;
  try { const r=await api("/api/retry",{slot}); $("message").textContent=r.message; }
  catch(e){ $("message").textContent=e.message; }
  await updateStatus();
}
function canvasPoint(event) {
  const canvas=$("roiCanvas"), rect=canvas.getBoundingClientRect();
  return {x:Math.max(0,Math.min(canvas.width-1,(event.clientX-rect.left)*canvas.width/rect.width)), y:Math.max(0,Math.min(canvas.height-1,(event.clientY-rect.top)*canvas.height/rect.height))};
}
function drawCanvas() {
  if(!roiImage) return; const canvas=$("roiCanvas"), ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(roiImage,0,0);
  if(roi){ ctx.strokeStyle="#00ff66";ctx.lineWidth=3;ctx.strokeRect(roi.x,roi.y,roi.width,roi.height);ctx.fillStyle="rgba(0,255,102,.13)";ctx.fillRect(roi.x,roi.y,roi.width,roi.height); }
}
function applyZoom() { if(!roiImage)return; const z=Number($("zoom").value); const c=$("roiCanvas"); c.style.width=(c.width*z)+"px"; c.style.height=(c.height*z)+"px"; }
async function loadPendingImage(key) {
  pendingKey=key; roi=null; $("submitRoi").disabled=true; $("roiText").textContent="尚未框选";
  const image=new Image(); image.onload=()=>{ roiImage=image; const c=$("roiCanvas");c.width=image.naturalWidth;c.height=image.naturalHeight;applyZoom();drawCanvas(); };
  image.src="/captured.jpg?t="+Date.now();
}
async function submitRoi() {
  if(!roi || !state.pending_roi) return;
  try {
    const result=await api("/api/roi",{slot:state.pending_roi.slot,roi});
    $("message").textContent=result.message; roi=null; roiImage=null;
  } catch(e){ $("message").textContent="ROI提交失败："+e.message; }
  await updateStatus();
}
async function updateStatus() {
  try {
    const response=await fetch("/api/status",{cache:"no-store"}); state=await response.json();
    renderSlots(); $("message").textContent=state.message||"";
    const pending=state.pending_roi;
    $("roiPanel").style.display=pending?"block":"none";
    if(pending){ const key=pending.slot+":"+pending.attempt+":"+pending.frame; if(key!==pendingKey) loadPendingImage(key); }
    else { pendingKey=""; roi=null; roiImage=null; }
    const done=state.status==="completed"; $("complete").style.display=done?"block":"none";
    if(done) $("validatedPath").textContent=state.validated_output+"\nSHA256: "+state.validated_yaml_sha256;
    $("stop").style.display=done?"none":"block";
  } catch(e){ $("message").textContent="网页暂时无法读取板端状态："+e.message; }
}
function refreshPreviews(){ const t=Date.now(); $("leftPreview").src="/preview/left.jpg?t="+t; $("rightPreview").src="/preview/right.jpg?t="+t; }
$("roiCanvas").addEventListener("pointerdown",e=>{if(!roiImage)return;dragging=true;start=canvasPoint(e);roi=null;drawCanvas();});
$("roiCanvas").addEventListener("pointermove",e=>{if(!dragging)return;const p=canvasPoint(e);const x=Math.floor(Math.min(start.x,p.x)),y=Math.floor(Math.min(start.y,p.y));roi={x,y,width:Math.max(1,Math.ceil(Math.abs(p.x-start.x))),height:Math.max(1,Math.ceil(Math.abs(p.y-start.y)))};drawCanvas();$("roiText").textContent=`原图ROI: x=${roi.x}, y=${roi.y}, w=${roi.width}, h=${roi.height}`;});
window.addEventListener("pointerup",()=>{if(dragging){dragging=false;$("submitRoi").disabled=!(roi&&roi.width>=5&&roi.height>=5);}});
$("zoom").addEventListener("change",applyZoom); $("submitRoi").addEventListener("click",submitRoi);
$("stop").addEventListener("click",async()=>{if(!confirm("确认安全暂停？已保存内容保留，后续同一目录加 --resume。"))return;try{await api("/api/stop",{});}catch(e){} });
$("finish").addEventListener("click",async()=>{if(!confirm("请先下载结果。确认关闭验证网页并安全结束进程？"))return;try{await api("/api/finish",{});}catch(e){} });
setInterval(updateStatus,500); setInterval(refreshPreviews,700); updateStatus(); refreshPreviews();
</script>
</body></html>
"""


class ValidationApplication:
    def __init__(
        self,
        engine: RuntimeStereoEngine,
        session: SessionStore,
        *,
        max_frame_age_seconds: float,
        capture_timeout_seconds: float,
    ) -> None:
        self.engine = engine
        self.session = session
        self.max_frame_age_seconds = float(max_frame_age_seconds)
        self.capture_timeout_seconds = float(capture_timeout_seconds)
        self.minimum_batch_interval_seconds = 0.22
        self.maximum_batch_translation_px = 3.0
        self._lock = threading.RLock()
        self._left_preview: Optional[bytes] = None
        self._right_preview: Optional[bytes] = None
        self._latest_frame: Optional[StereoFrame] = None
        self._latest_sequence = -1
        self._last_new_frame_monotonic = 0.0
        self._capture_request: Optional[dict[str, Any]] = None
        self._stop_requested = False
        self._finish_requested = False
        self._message = (
            "三点验证已完成；请下载结果并点击安全结束。"
            if self.session.data["status"] == "completed"
            else "等待 GS130W 实时新帧……"
        )
        self._fatal_error: Optional[str] = None

    def process_frame(self, combined: np.ndarray, sequence: int, received_at: float) -> None:
        frame = self.engine.prepare_frame(combined, sequence, received_at)
        left_preview = encode_jpeg(frame.left)
        right_preview = encode_jpeg(frame.right)
        with self._lock:
            if sequence <= self._latest_sequence:
                return
            self._latest_frame = frame
            self._latest_sequence = int(sequence)
            self._last_new_frame_monotonic = time.monotonic()
            self._left_preview = left_preview
            self._right_preview = right_preview
            request = self._capture_request
            if request is None:
                if self.session.pending_attempt() is None:
                    self._message = "相机新帧正常；摆稳设备和靶纸后手动采集。"
                return
            if sequence <= int(request["after_sequence"]):
                return
            if received_at <= float(request["requested_at_monotonic"]):
                return
            now = time.monotonic()
            if now > float(request["deadline_monotonic"]):
                self._capture_request = None
                self._message = "三帧批量采集超时；已保存的完整新帧保留，摆稳后点击继续补齐。"
                return
            if now - float(request["last_saved_monotonic"]) < self.minimum_batch_interval_seconds:
                return
            gray = cv2.cvtColor(frame.left_rectified, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
            anchor = request.get("anchor_gray")
            if anchor is not None:
                try:
                    shift, response = cv2.phaseCorrelate(anchor, small)
                    motion_px = math.hypot(float(shift[0]), float(shift[1])) / 0.5
                except cv2.error:
                    motion_px, response = float("inf"), 0.0
                if response < 0.05 or motion_px > self.maximum_batch_translation_px:
                    request["after_sequence"] = sequence
                    self._message = (
                        f"检测到批量帧不稳定（位移约 {motion_px:.1f}px），"
                        "本帧不保存；请停止触碰设备和靶纸。"
                    )
                    return
            try:
                record = self.session.save_capture(request["slot"], request["known_mm"], frame)
            except Exception as exc:
                self._capture_request = None
                self._message = f"本次未保存：{exc}"
                return
            request["after_sequence"] = sequence
            request["last_saved_monotonic"] = now
            if request.get("anchor_gray") is None:
                request["anchor_gray"] = small
            if int(record["index"]) >= FRAMES_PER_ATTEMPT:
                self._capture_request = None
                self._message = (
                    f"已自动保存 {request['slot']} 的 3/3 个新同步帧。"
                    "请在质量最好的校正后物理左目图上框选一次 ROI。"
                )
            else:
                self._message = (
                    f"已保存 {request['slot']} 第 {record['index']}/3 帧，"
                    "设备稳定，正在自动等待下一个新帧……"
                )

    def expire_capture(self) -> None:
        with self._lock:
            if self._capture_request is None:
                return
            if time.monotonic() > float(self._capture_request["deadline_monotonic"]):
                self._capture_request = None
                self._message = "三帧采集超时；已完整保存的帧保留，可再点一次继续补齐。"

    def set_fatal_error(self, message: str) -> None:
        with self._lock:
            self._fatal_error = str(message)
            self._message = f"严重错误：{message}"

    def camera_fresh(self) -> bool:
        return bool(
            self._latest_frame is not None
            and time.monotonic() - self._last_new_frame_monotonic <= self.max_frame_age_seconds
        )

    def request_capture(self, slot: str, known_mm: Any) -> str:
        with self._lock:
            if self._fatal_error:
                raise ValidationError(self._fatal_error)
            if self.session.pending_attempt() is not None:
                raise ValidationError("请先框选并提交上一轮三帧的共用 ROI")
            if self._capture_request is not None:
                raise ValidationError("上一个采集请求正在等待新帧")
            if not self.camera_fresh():
                raise ValidationError("相机画面不是实时新帧，禁止采集")
            attempt, _known_m = self.session.can_capture(slot, known_mm)
            requested = time.monotonic()
            anchor_gray: Optional[np.ndarray] = None
            if attempt["frames"]:
                latest_dir = self.session.root / attempt["frames"][-1]["relative_dir"]
                latest_image = cv2.imread(str(latest_dir / "rectified_left.png"), cv2.IMREAD_GRAYSCALE)
                if latest_image is not None:
                    anchor_gray = cv2.resize(
                        latest_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                    ).astype(np.float32)
            self._capture_request = {
                "slot": slot,
                "known_mm": finite_float(known_mm, "实测距离(mm)"),
                "after_sequence": self._latest_sequence,
                "requested_at_monotonic": requested,
                "deadline_monotonic": requested + self.capture_timeout_seconds,
                "last_saved_monotonic": requested - self.minimum_batch_interval_seconds,
                "anchor_gray": anchor_gray,
            }
            existing = len(attempt["frames"])
            self._message = (
                f"已接受手动请求；当前 {existing}/3，"
                "只会自动保存点击之后、间隔不小于 0.22 秒且稳定的新同步帧……"
            )
            return self._message

    def submit_roi(self, slot: str, roi_payload: Any) -> dict[str, Any]:
        if not isinstance(roi_payload, Mapping):
            raise ValidationError("roi 必须包含 x/y/width/height")
        try:
            roi = tuple(
                int(round(finite_float(roi_payload[name], f"roi.{name}")))
                for name in ("x", "y", "width", "height")
            )
        except KeyError as exc:
            raise ValidationError(f"ROI 缺少字段：{exc.args[0]}") from exc
        with self._lock:
            result = self.session.submit_roi(slot, roi)  # type: ignore[arg-type]
            frames = result["frames"]
            passing = sum(bool(frame["metrics"]["passed"]) for frame in frames)
            details = []
            for frame in frames:
                metrics = frame["metrics"]
                estimate = metrics["estimated_distance_m"]
                details.append(
                    f"帧{frame['index']}={'通过' if metrics['passed'] else '失败'}"
                    + ("" if estimate is None else f"({estimate:.3f}m)")
                )
            message = f"同一 ROI 已评估三帧：{passing}/3 通过；" + "，".join(details)
            if result["attempt_status"] == "failed":
                message += "\n本轮三帧未锁定；请新建重试，不会删除旧数据。"
            elif result["slot_locked"]:
                message += f"\n{slot} 三帧一致性通过，已锁定。"
            if result["all_locked"]:
                output = self.session.finalize()
                message += f"\n三点全部通过，已生成：{output}"
            self._message = message
            return {"message": message, **result}

    def retry(self, slot: str) -> str:
        with self._lock:
            if self.session.pending_attempt() is not None or self._capture_request is not None:
                raise ValidationError("当前有未提交 ROI 的三帧或正在采集，不能新建重试")
            number = self.session.retry_slot(slot)
            self._message = f"{slot} 已新建第 {number} 轮三帧尝试；旧数据已保留。"
            return self._message

    def request_stop(self, completed_finish: bool = False) -> str:
        with self._lock:
            if completed_finish:
                if self.session.data.get("status") != "completed":
                    raise ValidationError("只有完成验证后才能执行最终结束")
                self._finish_requested = True
                self._stop_requested = True
                self._message = "验证已完成，正在安全关闭网页和 ROS 订阅……"
            else:
                self.session.pause("web_safe_pause")
                self._stop_requested = True
                self._message = "已安全暂停；下次用同一目录加 --resume 继续。"
            return self._message

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = self.session.public_status()
            status.update(
                {
                    "message": self._message,
                    "camera_fresh": self.camera_fresh(),
                    "latest_frame_sequence": self._latest_sequence,
                    "capture_pending": self._capture_request is not None,
                    "fatal_error": self._fatal_error,
                    "session_dir": str(self.session.root),
                    "result_paths": {
                        "session_dir": str(self.session.root),
                        "manifest": str(self.session.manifest_path),
                        "audit": str(self.session.audit_path),
                        "pristine_candidate": str(self.session.pristine_path),
                        "validated_yaml": str(self.session.validated_output),
                    },
                }
            )
            return status

    def preview(self, side: str) -> Optional[bytes]:
        with self._lock:
            return self._left_preview if side == "left" else self._right_preview

    def captured_preview(self) -> Optional[bytes]:
        pending = self.session.pending_attempt()
        if pending is None:
            return None
        frame_dir = self.session.root / pending[2]["relative_dir"]
        image = cv2.imread(str(frame_dir / "rectified_left.png"), cv2.IMREAD_COLOR)
        return None if image is None else encode_jpeg(image, quality=92)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def handler_for(application: ValidationApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GS130WDistanceValidator/1.0"

        def send_bytes(
            self,
            status: int,
            content_type: str,
            payload: bytes,
            *,
            disposition: Optional[str] = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline' 'self'; script-src 'unsafe-inline' 'self'; img-src 'self' data:")
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            try:
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError):
                encoded = b'{"message":"internal JSON serialization failure"}'
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            self.send_bytes(status, "application/json; charset=utf-8", encoded)

        def send_path(self, path: Path, content_type: str, download_name: str) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    self.wfile.write(block)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", WEB_PAGE.encode("utf-8"))
                return
            if path == "/api/status":
                self.send_json(HTTPStatus.OK, application.snapshot())
                return
            if path == "/healthz":
                snapshot = application.snapshot()
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "status": snapshot["status"],
                        "camera_fresh": snapshot["camera_fresh"],
                        "session_dir": snapshot["session_dir"],
                        "validated_output": snapshot["validated_output"],
                    },
                )
                return
            if path in {"/preview/left.jpg", "/preview/right.jpg"}:
                side = "left" if "left" in path else "right"
                preview = application.preview(side)
                if preview is None:
                    self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"message": "实时预览尚未就绪"})
                else:
                    self.send_bytes(HTTPStatus.OK, "image/jpeg", preview)
                return
            if path == "/captured.jpg":
                preview = application.captured_preview()
                if preview is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"message": "当前没有待框选帧"})
                else:
                    self.send_bytes(HTTPStatus.OK, "image/jpeg", preview)
                return
            if path == "/download/validated.yaml":
                output = application.session.validated_output
                if not output.is_file() or application.session.data.get("status") != "completed":
                    self.send_json(HTTPStatus.NOT_FOUND, {"message": "validated YAML 尚未生成"})
                else:
                    self.send_bytes(
                        HTTPStatus.OK,
                        "application/yaml; charset=utf-8",
                        output.read_bytes(),
                        disposition=f'attachment; filename="{output.name}"',
                    )
                return
            if path == "/download/session.json":
                manifest = application.session.manifest_path
                self.send_bytes(
                    HTTPStatus.OK,
                    "application/json; charset=utf-8",
                    manifest.read_bytes(),
                    disposition='attachment; filename="session.json"',
                )
                return
            if path in {"/download/audit.zip", "/api/audit.zip"}:
                try:
                    bundle = application.session.audit_bundle()
                except ValidationError as exc:
                    self.send_json(HTTPStatus.CONFLICT, {"message": str(exc)})
                else:
                    self.send_path(bundle, "application/zip", bundle.name)
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"message": "Not found"})

        def read_json(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                raise ValidationError("Content-Type 必须为 application/json")
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError as exc:
                raise ValidationError("Content-Length 无效") from exc
            if length < 0 or length > 8192:
                raise ValidationError("请求体过大")
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise ValidationError("JSON 请求体无效") from exc
            if not isinstance(value, dict):
                raise ValidationError("JSON 请求体必须是对象")
            return value

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                payload = self.read_json()
                if path == "/api/capture":
                    message = application.request_capture(str(payload.get("slot", "")), payload.get("known_mm"))
                    result = {"accepted": True, "message": message}
                elif path == "/api/roi":
                    result = application.submit_roi(str(payload.get("slot", "")), payload.get("roi"))
                elif path == "/api/retry":
                    message = application.retry(str(payload.get("slot", "")))
                    result = {"accepted": True, "message": message}
                elif path == "/api/stop":
                    message = application.request_stop(False)
                    result = {"accepted": True, "message": message}
                elif path == "/api/finish":
                    message = application.request_stop(True)
                    result = {"accepted": True, "message": message}
                else:
                    self.send_json(HTTPStatus.NOT_FOUND, {"message": "Not found"})
                    return
                self.send_json(HTTPStatus.ACCEPTED, result)
            except ValidationError as exc:
                self.send_json(HTTPStatus.CONFLICT, {"accepted": False, "message": str(exc)})
            except Exception as exc:
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"accepted": False, "message": f"内部错误：{type(exc).__name__}: {exc}"},
                )

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"message": "Method not allowed"})

        do_PUT = do_HEAD
        do_PATCH = do_HEAD
        do_DELETE = do_HEAD
        do_OPTIONS = do_HEAD

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


class ValidationWebServer:
    def __init__(self, application: ValidationApplication, host: str, port: int) -> None:
        self.server = ReusableThreadingHTTPServer((host, int(port)), handler_for(application))
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="gs130w-distance-validation-web",
            daemon=True,
        )

    def start(self) -> tuple[str, int]:
        self.thread.start()
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GS130W 1m/3m/5m 三点严格双目测距验证网页；"
            "候选 YAML 始终只读，三点全通过后另建 valid:true YAML。"
        )
    )
    parser.add_argument(
        "--candidate-yaml",
        "--calibration",
        "--candidate",
        dest="candidate_yaml",
        type=Path,
        help="板端 valid:false 候选双目 YAML（只读）",
    )
    parser.add_argument(
        "--session-dir",
        "--output",
        dest="session_dir",
        type=Path,
        help=(
            "新建/继续的独立审计目录；末级目录名就是 session_id，"
            "只能含英文字母、数字、._-"
        ),
    )
    parser.add_argument(
        "--validated-output",
        type=Path,
        help="最终新 YAML；默认 <session-dir>/stereo_gs130w_validated.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=Path("/opt/rdk-patrol/current"))
    parser.add_argument("--topic", default="/image_combine_jpeg")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    parser.add_argument("--max-frame-age-seconds", type=float, default=2.0)
    parser.add_argument(
        "--capture-timeout-seconds",
        type=float,
        default=12.0,
        help="每次自动集齐三帧的总超时，默认 12 秒",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="严格校验 session-dir 内的 manifest/哈希/审计链后继续",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行不需要 ROS/摄像头的离线安全性自测并退出",
    )
    return parser.parse_args(argv)


def run_self_test() -> None:
    """Small offline test covering no-overwrite, mapping, audit and resume."""

    # The workstation safety test should remain runnable even when only the
    # board project venv has PyYAML.  JSON is a strict YAML subset and is enough
    # for the synthetic payloads below; production execution still requires
    # PyYAML through require_yaml().
    global yaml
    if yaml is None:
        class _SelfTestJsonYaml:
            @staticmethod
            def safe_dump(payload: Any, **_kwargs: Any) -> str:
                return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

            @staticmethod
            def safe_load(text: str) -> Any:
                return json.loads(text)

        yaml = _SelfTestJsonYaml()  # type: ignore[assignment]

    with tempfile.TemporaryDirectory(prefix="gs130w-validator-selftest-") as temporary_text:
        temporary = Path(temporary_text)
        create_target = temporary / "exclusive.bin"
        atomic_create_bytes(create_target, b"first")
        try:
            atomic_create_bytes(create_target, b"second")
        except ValidationError:
            pass
        else:
            raise AssertionError("atomic_create_bytes overwrote an existing file")
        assert create_target.read_bytes() == b"first"

        top = np.zeros((4, 6, 3), dtype=np.uint8)
        bottom = np.full((4, 6, 3), 200, dtype=np.uint8)
        combined = np.vstack([top, bottom])
        left, right = split_physical_views(
            combined,
            layout="vertical",
            physical_left_view="bottom",
            physical_right_view="top",
            rotation="ccw90",
        )
        assert left.shape == (6, 4, 3) and right.shape == (6, 4, 3)
        assert int(np.median(left)) == 200 and int(np.median(right)) == 0

        candidate = temporary / "candidate.yaml"
        candidate.write_text("valid: false\n", encoding="utf-8")
        dummy_depth = b"# self-test depth runtime\n"
        dummy_calibration = b"# self-test calibration runtime\n"
        dummy_gs130w = b"# self-test gs130w runtime\n"

        class DummyEngine:
            candidate_path = candidate
            candidate_bytes = candidate.read_bytes()
            candidate_sha256 = sha256_bytes(candidate_bytes)
            pristine_payload = {"valid": False}
            layout = "vertical"
            rotation = "ccw90"
            physical_left_view = "bottom"
            physical_right_view = "top"
            expected_size = (640, 1280)
            runtime_source_files = {
                "depth": {
                    "source_path": "self-test/depth.py",
                    "archive_file": "runtime_depth.py",
                    "sha256": sha256_bytes(dummy_depth),
                    "bytes": dummy_depth,
                },
                "calibration": {
                    "source_path": "self-test/calibration.py",
                    "archive_file": "runtime_calibration.py",
                    "sha256": sha256_bytes(dummy_calibration),
                    "bytes": dummy_calibration,
                },
                "gs130w": {
                    "source_path": "self-test/gs130w.py",
                    "archive_file": "runtime_gs130w.py",
                    "sha256": sha256_bytes(dummy_gs130w),
                    "bytes": dummy_gs130w,
                },
            }

        session_root = temporary / "gs130w_3point_selftest"
        output = session_root / "stereo_gs130w_validated.yaml"
        store = SessionStore(DummyEngine(), session_root, output, resume=False)  # type: ignore[arg-type]
        assert store.data["session_id"] == session_root.name
        store._verify_audit_chain()
        assert math.isclose(store._validate_known_distance("1m", 1000), 1.0)
        for slot, invalid in (("1m", 799), ("3m", 2400), ("5m", 5800)):
            try:
                store._validate_known_distance(slot, invalid)
            except ValidationError:
                pass
            else:
                raise AssertionError(f"invalid range accepted for {slot}")
        store.pause("self_test")
        partial_tail = b'{"simulated_short_write":'
        with store.audit_path.open("ab") as stream:
            stream.write(partial_tail)
            stream.flush()
            os.fsync(stream.fileno())
        resumed = SessionStore(DummyEngine(), session_root, output, resume=True)  # type: ignore[arg-type]
        assert resumed.data["status"] == "in_progress"
        resumed._verify_audit_chain()
        tail_evidence = session_root / f"audit_incomplete_tail_{sha256_bytes(partial_tail)}.bin"
        assert tail_evidence.read_bytes() == partial_tail

        class TransactionEngine(DummyEngine):
            expected_size = (64, 48)
            pristine_payload = {
                "valid": False,
                "validation": {
                    "stereo_rms_px": 0.33,
                    "median_epipolar_error_px": 0.23,
                },
            }

            def evaluate(
                self,
                left_rectified: np.ndarray,
                _right_rectified: np.ndarray,
                roi: tuple[int, int, int, int],
                known_distance_m: float,
            ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
                metrics = {
                    "passed": True,
                    "failure_checks": [],
                    "known_distance_m": known_distance_m,
                    "estimated_distance_m": known_distance_m * 1.001,
                    "relative_error": 0.001,
                    "roi": {
                        "x": roi[0],
                        "y": roi[1],
                        "width": roi[2],
                        "height": roi[3],
                    },
                    "valid_pixels": 300,
                    "valid_fraction": 0.75,
                    "relative_mad": 0.01,
                    "central80_spread_over_median": 0.05,
                }
                return metrics, left_rectified.copy(), np.zeros_like(left_rectified)

        transaction_root = temporary / "gs130w_3point_transaction_test"
        transaction_output = transaction_root / "stereo_gs130w_validated.yaml"
        transaction_store = SessionStore(
            TransactionEngine(), transaction_root, transaction_output, resume=False
        )  # type: ignore[arg-type]
        global_sequence = 0
        for slot, known_mm in (("1m", 1000), ("3m", 3000), ("5m", 5000)):
            for _frame_index in range(3):
                global_sequence += 1
                left_image = np.zeros((48, 64, 3), dtype=np.uint8)
                cv2.line(
                    left_image,
                    (0, (global_sequence * 5) % 47),
                    (63, (47 - global_sequence * 3) % 47),
                    (255, 255, 255),
                    1,
                )
                combined_image = np.zeros((96, 64, 3), dtype=np.uint8)
                combined_image[global_sequence, global_sequence] = global_sequence
                frame = StereoFrame(
                    combined=combined_image,
                    left=left_image,
                    right=left_image.copy(),
                    left_rectified=left_image.copy(),
                    right_rectified=left_image.copy(),
                    sequence=global_sequence,
                    received_at=float(global_sequence),
                )
                transaction_store.save_capture(slot, known_mm, frame)
            pending = transaction_store.pending_attempt()
            assert pending is not None and pending[0] == slot
            transaction_result = transaction_store.submit_roi(slot, (10, 10, 30, 25))
            assert transaction_result["slot_locked"] is True
            assert len(transaction_result["frames"]) == 3
            for frame_record in transaction_result["frames"]:
                frame_dir = transaction_root / frame_record["relative_dir"]
                result_payload = json.loads(
                    (frame_dir / "result.json").read_text(encoding="utf-8")
                )
                assert result_payload["shared_roi_for_three_frames"] is True
        assert transaction_store.all_slots_locked()

        # Every durable write boundary of finalization must be replayable.  All
        # clones keep the same final directory name because session_id is bound
        # to that name by design.
        fault_points = (
            "after_manifest_snapshot",
            "after_audit_snapshot",
            "after_finalize_transaction",
            "after_finalizing_manifest",
            "after_validated_yaml",
            "after_completed_manifest",
            "after_final_audit",
        )
        for fault_index, fault_point in enumerate(fault_points, 1):
            clone_parent = temporary / f"fault_case_{fault_index:02d}"
            clone_root = clone_parent / transaction_root.name
            shutil.copytree(transaction_root, clone_root)
            clone_output = clone_root / transaction_output.name
            crashing = SessionStore(
                TransactionEngine(), clone_root, clone_output, resume=True
            )  # type: ignore[arg-type]
            crashing._test_fault_at = fault_point
            try:
                crashing.finalize()
            except RuntimeError as exc:
                assert str(exc) == f"TEST_CRASH_AT:{fault_point}"
            else:
                raise AssertionError(f"fault injection did not fire: {fault_point}")
            recovered_store = SessionStore(
                TransactionEngine(), clone_root, clone_output, resume=True
            )  # type: ignore[arg-type]
            if recovered_store.data["status"] != "completed":
                recovered_store.finalize()
            assert recovered_store.data["status"] == "completed"
            assert clone_output.is_file()
            assert recovered_store.finalize_transaction_path.is_file()
            assert sha256_file(clone_output) == recovered_store.data["validated_yaml_sha256"]
            recovered_store._verify_audit_chain()
            recovered_store._verify_session_checksums()
    print("SELF_TEST_OK", flush=True)


def validate_cli(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.candidate_yaml is None:
        raise ValidationError("缺少 --candidate-yaml")
    if args.session_dir is None:
        raise ValidationError("缺少 --session-dir")
    candidate = args.candidate_yaml.expanduser().resolve()
    session_dir = args.session_dir.expanduser().resolve()
    validated_output = (
        args.validated_output.expanduser().resolve()
        if args.validated_output is not None
        else session_dir / "stereo_gs130w_validated.yaml"
    )
    if not candidate.is_file():
        raise ValidationError(f"候选 YAML 不存在：{candidate}")
    if not 1 <= int(args.port) <= 65535:
        raise ValidationError("--port 必须在 1–65535")
    for name in ("wait_seconds", "max_frame_age_seconds", "capture_timeout_seconds"):
        value = finite_float(getattr(args, name), f"--{name.replace('_', '-')}")
        if value <= 0:
            raise ValidationError(f"--{name.replace('_', '-')} 必须大于 0")
    if float(args.capture_timeout_seconds) < 2.0:
        raise ValidationError("--capture-timeout-seconds 至少为 2 秒")
    return candidate, session_dir, validated_output


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            run_self_test()
        except Exception as exc:
            print(f"SELF_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        return 0

    try:
        candidate, session_dir, validated_output = validate_cli(args)
        engine = RuntimeStereoEngine(candidate, args.project_root)
        session = SessionStore(
            engine,
            session_dir,
            validated_output,
            resume=bool(args.resume),
        )
        if session.all_slots_locked() and session.data.get("status") != "completed":
            session.finalize()
        application = ValidationApplication(
            engine,
            session,
            max_frame_age_seconds=args.max_frame_age_seconds,
            capture_timeout_seconds=args.capture_timeout_seconds,
        )
        web = ValidationWebServer(application, args.host, args.port)
        bound_host, bound_port = web.start()
    except (ValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    print(f"WEB_READY: http://127.0.0.1:{bound_port}/", flush=True)
    print(f"HEALTH_URL: http://127.0.0.1:{bound_port}/healthz", flush=True)
    print(f"SESSION_DIR: {session.root}", flush=True)
    print(f"RESULT_YAML: {session.validated_output}", flush=True)
    print(
        "请通过 SSH -L 隧道打开网页；每个距离点只需一次自动三帧采集和一次 ROI。",
        flush=True,
    )

    subscriber: Optional[FreshFrameSubscriber] = None
    last_sequence = -1
    completion_announced = False
    interrupted = False
    fatal = False
    try:
        if session.data.get("status") != "completed":
            subscriber = FreshFrameSubscriber(args.topic, args.wait_seconds)
            print(f"CAMERA_SUBSCRIBED: {args.topic} ({subscriber.topic_type})", flush=True)
        while not application.stop_requested():
            if session.data.get("status") == "completed":
                if not completion_announced:
                    print("DISTANCE_VALIDATION_COMPLETE", flush=True)
                    print(f"SESSION_DIR: {session.root}", flush=True)
                    print(f"RESULT_YAML: {session.validated_output}", flush=True)
                    print(f"VALIDATED_YAML_READY: {session.validated_output}", flush=True)
                    print(
                        f"AUDIT_ZIP_URL: http://127.0.0.1:{bound_port}/download/audit.zip",
                        flush=True,
                    )
                    print("网页保持开启；下载完成后点击“安全结束”。", flush=True)
                    completion_announced = True
                time.sleep(0.15)
                continue
            assert subscriber is not None
            combined, sequence, received_at = subscriber.spin_once()
            application.expire_capture()
            if combined is None or sequence == last_sequence:
                continue
            last_sequence = sequence
            application.process_frame(combined, sequence, received_at)
    except KeyboardInterrupt:
        interrupted = True
        print("\nINTERRUPTED_BY_USER", flush=True)
        session.pause("keyboard_interrupt")
    except Exception as exc:
        fatal = True
        application.set_fatal_error(str(exc))
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if subscriber is not None:
            subscriber.close()
        web.stop()

    if session.data.get("status") == "completed" and not fatal:
        print("VALIDATION_PROCESS_EXIT_OK", flush=True)
        return 0
    if not interrupted and not fatal and session.data.get("status") != "completed":
        # Web safe-pause already persisted the paused state.
        interrupted = True
    if interrupted:
        print(f"VALIDATION_PAUSED_RESUMABLE: {session.root}", flush=True)
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
