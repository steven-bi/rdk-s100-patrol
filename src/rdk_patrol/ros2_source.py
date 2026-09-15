from __future__ import annotations

"""ROS 2 camera input for the unified patrol runtime.

The module deliberately imports ROS packages only when ``start`` is called so
configuration validation and replay tests remain usable on a Windows
development computer.  One subscription publishes into one ``FrameHub``; all
detectors consume the same envelope and never create their own camera
subscription.
"""

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from .io.frame_hub import FrameHub


COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
RAW_IMAGE_TYPE = "sensor_msgs/msg/Image"
SUPPORTED_IMAGE_TYPES = frozenset({COMPRESSED_IMAGE_TYPE, RAW_IMAGE_TYPE})


def message_stamp_seconds(message: Any) -> float | None:
    try:
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError):
        return None


def decode_compressed_image(message: Any) -> np.ndarray:
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("OpenCV could not decode the compressed ROS image")
    return frame


def decode_raw_image(message: Any) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError("raw ROS image has invalid dimensions")

    encoding = str(message.encoding).strip().lower()
    payload = np.frombuffer(message.data, dtype=np.uint8)
    expected = height * step
    if payload.size < expected:
        raise ValueError(
            f"raw ROS image is truncated: received={payload.size}, expected={expected}"
        )
    rows = payload[:expected].reshape(height, step)

    if encoding in {"bgr8", "rgb8"}:
        required = width * 3
        if step < required:
            raise ValueError("raw three-channel ROS image step is too small")
        frame = rows[:, :required].reshape(height, width, 3)
        if encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif encoding in {"bgra8", "rgba8"}:
        required = width * 4
        if step < required:
            raise ValueError("raw four-channel ROS image step is too small")
        frame4 = rows[:, :required].reshape(height, width, 4)
        conversion = (
            cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        )
        frame = cv2.cvtColor(frame4, conversion)
    elif encoding in {"mono8", "8uc1"}:
        if step < width:
            raise ValueError("raw mono ROS image step is too small")
        frame = cv2.cvtColor(rows[:, :width], cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"unsupported raw ROS image encoding: {message.encoding}")
    return np.ascontiguousarray(frame)


def decode_ros_image(message: Any, message_type: str) -> np.ndarray:
    if message_type == COMPRESSED_IMAGE_TYPE:
        return decode_compressed_image(message)
    if message_type == RAW_IMAGE_TYPE:
        return decode_raw_image(message)
    raise ValueError(f"unsupported ROS image type: {message_type}")


@dataclass(frozen=True)
class Ros2SourceStats:
    running: bool
    topic: str
    message_type: str
    received: int
    decode_failures: int
    last_received_monotonic: float | None
    last_error: str | None


class Ros2FrameSource:
    """Own the process's only ROS camera subscription."""

    def __init__(
        self,
        frame_hub: FrameHub,
        *,
        topic: str,
        message_type: str = "auto",
        discovery_timeout_seconds: float = 8.0,
        node_name: str = "rdk_patrol_unified_camera",
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.frame_hub = frame_hub
        self.topic = str(topic).strip()
        if not self.topic:
            raise ValueError("ROS camera topic cannot be empty")
        requested_type = str(message_type).strip()
        self.requested_message_type = requested_type or "auto"
        if (
            self.requested_message_type != "auto"
            and self.requested_message_type not in SUPPORTED_IMAGE_TYPES
        ):
            raise ValueError(
                f"unsupported ROS image type: {self.requested_message_type}"
            )
        self.discovery_timeout_seconds = max(
            0.1, float(discovery_timeout_seconds)
        )
        self.node_name = str(node_name)
        self.on_error = on_error

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: Any = None
        self._executor: Any = None
        self._node: Any = None
        self._message_type = ""
        self._received = 0
        self._decode_failures = 0
        self._last_received_monotonic: float | None = None
        self._last_error: str | None = None
        self._startup_error: BaseException | None = None
        self._started = threading.Event()

    def start(self, *, startup_timeout_seconds: float = 12.0) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._started.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._spin_entry,
                name="rdk-ros2-camera",
                daemon=True,
            )
            self._thread.start()

        if not self._started.wait(max(0.1, float(startup_timeout_seconds))):
            self.stop()
            raise TimeoutError("timed out while starting the ROS 2 camera source")
        if self._startup_error is not None:
            error = self._startup_error
            self.stop()
            raise RuntimeError(f"failed to start ROS 2 camera source: {error}") from error

    def stop(self, *, timeout_seconds: float = 3.0) -> None:
        self._stop.set()
        executor = self._executor
        if executor is not None:
            try:
                executor.wake()
            except Exception:
                pass
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(max(0.0, float(timeout_seconds)))

    def stats(self) -> Ros2SourceStats:
        with self._lock:
            thread = self._thread
            return Ros2SourceStats(
                running=bool(thread is not None and thread.is_alive()),
                topic=self.topic,
                message_type=self._message_type or self.requested_message_type,
                received=self._received,
                decode_failures=self._decode_failures,
                last_received_monotonic=self._last_received_monotonic,
                last_error=self._last_error,
            )

    def _spin_entry(self) -> None:
        context = None
        executor = None
        node = None
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from sensor_msgs.msg import CompressedImage, Image

            context = Context()
            rclpy.init(args=None, context=context)
            node = Node(self.node_name, context=context)
            message_type = self.requested_message_type
            if message_type == "auto":
                message_type = self._discover_message_type(node)
            if message_type not in SUPPORTED_IMAGE_TYPES:
                raise RuntimeError(
                    f"topic {self.topic!r} did not expose a supported image type; "
                    f"resolved={message_type!r}"
                )

            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )

            def callback(message: Any) -> None:
                self._handle_message(message, message_type)

            ros_class = (
                CompressedImage
                if message_type == COMPRESSED_IMAGE_TYPE
                else Image
            )
            node.create_subscription(ros_class, self.topic, callback, qos)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            with self._lock:
                self._context = context
                self._executor = executor
                self._node = node
                self._message_type = message_type
                self._last_error = None
            self._started.set()

            while not self._stop.is_set() and context.ok():
                executor.spin_once(timeout_sec=0.05)
        except BaseException as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                if not self._started.is_set():
                    self._startup_error = exc
            self._report_error(self._last_error)
            self._started.set()
        finally:
            if executor is not None and node is not None:
                try:
                    executor.remove_node(node)
                except Exception:
                    pass
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:
                    pass
            if context is not None:
                try:
                    context.shutdown()
                except Exception:
                    pass
            with self._lock:
                self._context = None
                self._executor = None
                self._node = None

    def _discover_message_type(self, node: Any) -> str:
        deadline = time.monotonic() + self.discovery_timeout_seconds
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                topics = dict(node.get_topic_names_and_types())
            except Exception:
                topics = {}
            offered = list(topics.get(self.topic) or [])
            supported = [item for item in offered if item in SUPPORTED_IMAGE_TYPES]
            if len(supported) == 1:
                return supported[0]
            if len(supported) > 1:
                # Prefer compressed data to avoid transporting an unnecessarily
                # large stacked stereo frame through Python.
                if COMPRESSED_IMAGE_TYPE in supported:
                    return COMPRESSED_IMAGE_TYPE
                return supported[0]
            time.sleep(0.05)
        raise RuntimeError(
            f"unable to discover ROS type for {self.topic!r} within "
            f"{self.discovery_timeout_seconds:.1f}s"
        )

    def _handle_message(self, message: Any, message_type: str) -> None:
        try:
            frame = decode_ros_image(message, message_type)
            source_ts = message_stamp_seconds(message)
            received_at = time.monotonic()
            self.frame_hub.publish(
                frame,
                source_ts=source_ts,
                monotonic_ts=received_at,
                metadata={
                    "camera_topic": self.topic,
                    "camera_message_type": message_type,
                },
            )
            with self._lock:
                self._received += 1
                self._last_received_monotonic = received_at
                self._last_error = None
        except Exception as exc:
            message_text = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._decode_failures += 1
                self._last_error = message_text
            self._report_error(message_text)

    def _report_error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(str(message))
            except Exception:
                pass
