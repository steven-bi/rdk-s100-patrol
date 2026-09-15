from __future__ import annotations

"""Runtime coordination around the durable five-second garbage reviewer."""

from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping

import numpy as np

from .contracts import PointContext
from .review import GarbageReviewCollector, GarbageReviewWorker, PendingReview


@dataclass(frozen=True)
class GarbageCoordinatorState:
    active_point_id: str | None
    last_point_id: str | None
    windows_started: int
    reviews_enqueued: int
    windows_aborted: int
    last_error: str | None


class GarbagePointCoordinator:
    """Start one review window per visual point visit.

    Once started, a window keeps collecting the full main patrol view for five
    seconds even if an AprilTag is briefly occluded.  A new visit is armed only
    after the previous point has been absent for ``leave_grace_seconds``.
    """

    def __init__(
        self,
        collector: GarbageReviewCollector,
        point_capabilities: Mapping[str, set[str] | frozenset[str]],
        *,
        leave_grace_seconds: float = 2.0,
    ) -> None:
        self.collector = collector
        self.point_capabilities = {
            str(point_id): frozenset(str(value) for value in values)
            for point_id, values in point_capabilities.items()
        }
        self.leave_grace_seconds = max(0.0, float(leave_grace_seconds))
        self._active_point_id: str | None = None
        self._last_point_id: str | None = None
        self._last_seen_at: float | None = None
        self._windows_started = 0
        self._reviews_enqueued = 0
        self._windows_aborted = 0
        self._last_error: str | None = None

    def process(
        self,
        point_context: PointContext | None,
        full_frame_bgr: np.ndarray,
        *,
        now_monotonic: float,
        occurred_at: float,
    ) -> PendingReview | None:
        current = float(now_monotonic)
        point_id = (
            point_context.point_id
            if point_context is not None and point_context.valid
            else None
        )
        if point_id is not None:
            self._last_seen_at = current

        if self._active_point_id is not None:
            active_point_id = self._active_point_id
            changed_point = (
                point_id is not None and point_id != active_point_id
            )
            context_expired = (
                point_id is None
                and self._last_seen_at is not None
                and current - self._last_seen_at >= self.leave_grace_seconds
            )
            if changed_point or context_expired:
                self.collector.cancel(active_point_id)
                self._active_point_id = None
                self._windows_aborted += 1
                self._last_error = (
                    "point_changed_before_window_completed"
                    if changed_point
                    else "point_context_lost_before_window_completed"
                )
                if context_expired:
                    self._last_point_id = None
                    return None
                # A different, positively identified point may start its own
                # window below; no frame is ever offered to the old point.

        if self._active_point_id is not None:
            try:
                self.collector.offer(
                    self._active_point_id,
                    full_frame_bgr,
                    captured_monotonic=current,
                    captured_epoch=float(occurred_at),
                )
                review = self.collector.finalize(
                    self._active_point_id,
                    now_monotonic=current,
                )
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return None
            if review is not None:
                self._reviews_enqueued += 1
                self._active_point_id = None
                self._last_error = None
                return review
            return None

        if point_id is None:
            if (
                self._last_point_id is not None
                and self._last_seen_at is not None
                and current - self._last_seen_at >= self.leave_grace_seconds
            ):
                self._last_point_id = None
            return None

        if (
            point_id == self._last_point_id
            or "trash_review" not in self.point_capabilities.get(point_id, ())
        ):
            return None
        try:
            self.collector.start(
                point_id,
                point_context.point_name,
                now_monotonic=current,
            )
            self.collector.offer(
                point_id,
                full_frame_bgr,
                captured_monotonic=current,
                captured_epoch=float(occurred_at),
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None
        self._active_point_id = point_id
        self._last_point_id = point_id
        self._windows_started += 1
        self._last_error = None
        return None

    def state(self) -> GarbageCoordinatorState:
        return GarbageCoordinatorState(
            active_point_id=self._active_point_id,
            last_point_id=self._last_point_id,
            windows_started=self._windows_started,
            reviews_enqueued=self._reviews_enqueued,
            windows_aborted=self._windows_aborted,
            last_error=self._last_error,
        )


@dataclass(frozen=True)
class ReviewWorkerStats:
    running: bool
    processed: int
    retries: int
    alarms_saved: int
    not_full: int
    last_result: Mapping[str, Any] | None
    last_error: str | None


class GarbageReviewWorkerThread:
    def __init__(
        self,
        worker: GarbageReviewWorker,
        *,
        idle_seconds: float = 1.0,
        busy_seconds: float = 0.05,
    ) -> None:
        self.worker = worker
        self.idle_seconds = max(0.05, float(idle_seconds))
        self.busy_seconds = max(0.0, float(busy_seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._processed = 0
        self._retries = 0
        self._alarms_saved = 0
        self._not_full = 0
        self._last_result: Mapping[str, Any] | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="rdk-garbage-review",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout_seconds: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(max(0.0, float(timeout_seconds)))

    def stats(self) -> ReviewWorkerStats:
        with self._lock:
            thread = self._thread
            return ReviewWorkerStats(
                running=bool(thread is not None and thread.is_alive()),
                processed=self._processed,
                retries=self._retries,
                alarms_saved=self._alarms_saved,
                not_full=self._not_full,
                last_result=self._last_result,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = self.worker.process_one()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(self.idle_seconds)
                continue
            if result is None:
                self._stop.wait(self.idle_seconds)
                continue
            status = str(result.get("status") or "")
            with self._lock:
                self._processed += 1
                self._last_result = dict(result)
                self._last_error = None
                if status in {"retry", "cooldown"}:
                    self._retries += 1
                elif status == "alarm_saved":
                    self._alarms_saved += 1
                elif status == "not_full":
                    self._not_full += 1
            self._stop.wait(self.busy_seconds)
