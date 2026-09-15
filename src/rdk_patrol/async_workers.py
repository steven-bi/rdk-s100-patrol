from __future__ import annotations

"""Bounded background workers for non-HBM vision work.

AprilTag/fingerprint localization and stereo disparity must not stall the
single 15 FPS inference loop.  Both workers below use a replace-on-submit
mailbox, so a slow calculation drops obsolete frames instead of building
latency.
"""

from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol

import numpy as np

from .contracts import Box, DepthEstimate, PointContext


class PointResolverProtocol(Protocol):
    def resolve(
        self, frame_bgr: np.ndarray, observed_at: float
    ) -> PointContext | None:
        ...


class StereoEstimatorProtocol(Protocol):
    def estimate(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fire_box: Box,
        computed_at: float,
    ) -> DepthEstimate:
        ...


@dataclass(frozen=True)
class AsyncWorkerStats:
    running: bool
    submitted: int
    replaced: int
    completed: int
    failures: int
    last_completed_at: float | None
    last_error: str | None


class AsyncPointResolver:
    def __init__(
        self,
        resolver: PointResolverProtocol,
        *,
        minimum_interval_seconds: float = 0.10,
        copy_frame: bool = True,
    ) -> None:
        self.resolver = resolver
        self.minimum_interval_seconds = max(
            0.0, float(minimum_interval_seconds)
        )
        self.copy_frame = bool(copy_frame)
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, float] | None = None
        self._latest: PointContext | None = None
        self._latest_processed_at: float | None = None
        self._latest_observed_at: float | None = None
        self._last_submit_at: float | None = None
        self._last_error: str | None = None
        self._submitted = 0
        self._replaced = 0
        self._completed = 0
        self._failures = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._run,
                name="rdk-point-resolver",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout_seconds: float = 2.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(max(0.0, float(timeout_seconds)))

    def submit(self, frame_bgr: np.ndarray, observed_at: float) -> bool:
        observed = float(observed_at)
        with self._condition:
            if (
                self._last_submit_at is not None
                and observed - self._last_submit_at
                < self.minimum_interval_seconds
            ):
                return False
            if self._pending is not None:
                self._replaced += 1
            frame = (
                np.ascontiguousarray(frame_bgr).copy()
                if self.copy_frame
                else np.asarray(frame_bgr)
            )
            self._pending = (frame, observed)
            self._last_submit_at = observed
            self._submitted += 1
            self._condition.notify()
            return True

    def latest(
        self,
        *,
        now: float | None = None,
        max_age_seconds: float = 2.0,
    ) -> PointContext | None:
        current = time.monotonic() if now is None else float(now)
        with self._condition:
            if (
                self._latest_processed_at is None
                or current - self._latest_processed_at
                > max(0.0, float(max_age_seconds))
            ):
                return None
            return self._latest

    def stats(self) -> AsyncWorkerStats:
        with self._condition:
            thread = self._thread
            return AsyncWorkerStats(
                running=bool(thread is not None and thread.is_alive()),
                submitted=self._submitted,
                replaced=self._replaced,
                completed=self._completed,
                failures=self._failures,
                last_completed_at=self._latest_processed_at,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                frame, observed = self._pending
                self._pending = None
            try:
                result = self.resolver.resolve(frame, observed)
                error = None
            except Exception as exc:
                result = None
                error = f"{type(exc).__name__}: {exc}"
            completed_at = time.monotonic()
            with self._condition:
                self._latest = result
                self._latest_observed_at = observed
                self._latest_processed_at = completed_at
                self._completed += 1
                self._last_error = error
                if error is not None:
                    self._failures += 1


class AsyncDepthEstimator:
    def __init__(
        self,
        estimator: StereoEstimatorProtocol,
        *,
        minimum_interval_seconds: float = 0.5,
        copy_frames: bool = True,
    ) -> None:
        self.estimator = estimator
        self.minimum_interval_seconds = max(
            0.0, float(minimum_interval_seconds)
        )
        self.copy_frames = bool(copy_frames)
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, np.ndarray, Box, float] | None = None
        self._latest = DepthEstimate.unavailable("not_computed")
        self._latest_box: Box | None = None
        self._latest_completed_at: float | None = None
        self._last_submit_at: float | None = None
        self._last_error: str | None = None
        self._submitted = 0
        self._replaced = 0
        self._completed = 0
        self._failures = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._run,
                name="rdk-stereo-depth",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout_seconds: float = 2.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(max(0.0, float(timeout_seconds)))

    def submit(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fire_box: Box,
        computed_at: float,
    ) -> bool:
        submitted_at = float(computed_at)
        with self._condition:
            if (
                self._last_submit_at is not None
                and submitted_at - self._last_submit_at
                < self.minimum_interval_seconds
            ):
                return False
            if self._pending is not None:
                self._replaced += 1
            if self.copy_frames:
                left = np.ascontiguousarray(left_bgr).copy()
                right = np.ascontiguousarray(right_bgr).copy()
            else:
                left = np.asarray(left_bgr)
                right = np.asarray(right_bgr)
            box = tuple(float(value) for value in fire_box)
            self._pending = (left, right, box, submitted_at)  # type: ignore[arg-type]
            self._last_submit_at = submitted_at
            self._submitted += 1
            self._condition.notify()
            return True

    def latest(
        self,
        *,
        now: float | None = None,
        max_age_seconds: float = 1.0,
        fire_box: Box | None = None,
        minimum_iou: float = 0.05,
    ) -> DepthEstimate:
        current = time.monotonic() if now is None else float(now)
        with self._condition:
            if (
                self._latest_completed_at is None
                or current - self._latest_completed_at
                > max(0.0, float(max_age_seconds))
            ):
                return DepthEstimate.unavailable(
                    "stereo_result_stale_or_missing", computed_at=current
                )
            if (
                fire_box is not None
                and self._latest_box is not None
                and _box_iou(fire_box, self._latest_box) < float(minimum_iou)
            ):
                return DepthEstimate.unavailable(
                    "stereo_result_box_mismatch", computed_at=current
                )
            return self._latest

    def stats(self) -> AsyncWorkerStats:
        with self._condition:
            thread = self._thread
            return AsyncWorkerStats(
                running=bool(thread is not None and thread.is_alive()),
                submitted=self._submitted,
                replaced=self._replaced,
                completed=self._completed,
                failures=self._failures,
                last_completed_at=self._latest_completed_at,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                left, right, box, computed_at = self._pending
                self._pending = None
            try:
                result = self.estimator.estimate(
                    left, right, box, computed_at
                )
                if not isinstance(result, DepthEstimate):
                    raise TypeError("stereo estimator did not return DepthEstimate")
                error = None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result = DepthEstimate.unavailable(
                    f"estimator_error: {error}", computed_at=computed_at
                )
            completed_at = time.monotonic()
            with self._condition:
                self._latest = result
                self._latest_box = box
                self._latest_completed_at = completed_at
                self._completed += 1
                self._last_error = error
                if error is not None:
                    self._failures += 1


def _box_iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union
