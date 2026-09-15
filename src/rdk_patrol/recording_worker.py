from __future__ import annotations

"""Small bounded queue around the dual video recorder."""

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Protocol

import numpy as np


class DualRecorderProtocol(Protocol):
    def write(
        self,
        raw_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        timestamp: float | None = None,
    ) -> object:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class RecordingWorkerStats:
    running: bool
    queued: int
    submitted: int
    skipped: int
    written: int
    dropped: int
    failures: int
    last_write_monotonic: float | None
    last_error: str | None


class AsyncDualStreamRecorder:
    """Write video off the inference thread without unbounded memory growth."""

    def __init__(
        self,
        recorder: DualRecorderProtocol,
        *,
        queue_capacity: int = 3,
        copy_frames: bool = False,
        max_fps: float | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_fps is not None and float(max_fps) <= 0:
            raise ValueError("max_fps must be positive when provided")
        self.recorder = recorder
        self.queue_capacity = max(1, int(queue_capacity))
        self.copy_frames = bool(copy_frames)
        self.max_fps = None if max_fps is None else float(max_fps)
        self._minimum_interval_seconds = (
            0.0 if self.max_fps is None else 1.0 / self.max_fps
        )
        self._monotonic_clock = monotonic_clock
        self._queue: deque[tuple[np.ndarray, np.ndarray, float]] = deque()
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._submitted = 0
        self._skipped = 0
        self._written = 0
        self._dropped = 0
        self._failures = 0
        self._last_write_monotonic: float | None = None
        self._next_submission_monotonic: float | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._run,
                name="rdk-video-recorder",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        raw_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        epoch = time.time() if timestamp is None else float(timestamp)
        if self.copy_frames:
            raw = np.ascontiguousarray(raw_bgr).copy()
            annotated = np.ascontiguousarray(annotated_bgr).copy()
        else:
            raw = np.asarray(raw_bgr)
            annotated = np.asarray(annotated_bgr)
        submitted_at = self._monotonic_clock()
        with self._condition:
            self._submitted += 1
            if (
                self._next_submission_monotonic is not None
                and submitted_at < self._next_submission_monotonic
            ):
                self._skipped += 1
                return
            if self._minimum_interval_seconds:
                if self._next_submission_monotonic is None:
                    self._next_submission_monotonic = (
                        submitted_at + self._minimum_interval_seconds
                    )
                else:
                    overdue = (
                        submitted_at - self._next_submission_monotonic
                    )
                    intervals = max(
                        1,
                        int(overdue // self._minimum_interval_seconds) + 1,
                    )
                    self._next_submission_monotonic += (
                        intervals * self._minimum_interval_seconds
                    )
            while len(self._queue) >= self.queue_capacity:
                self._queue.popleft()
                self._dropped += 1
            self._queue.append((raw, annotated, epoch))
            self._condition.notify()

    def stop(
        self,
        *,
        timeout_seconds: float = 5.0,
        drain: bool = True,
    ) -> None:
        with self._condition:
            self._stop = True
            if not drain:
                self._dropped += len(self._queue)
                self._queue.clear()
            self._condition.notify_all()
            thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(max(0.0, float(timeout_seconds)))
        try:
            self.recorder.close()
        except Exception:
            pass

    def stats(self) -> RecordingWorkerStats:
        with self._condition:
            thread = self._thread
            return RecordingWorkerStats(
                running=bool(thread is not None and thread.is_alive()),
                queued=len(self._queue),
                submitted=self._submitted,
                skipped=self._skipped,
                written=self._written,
                dropped=self._dropped,
                failures=self._failures,
                last_write_monotonic=self._last_write_monotonic,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if not self._queue and self._stop:
                    return
                raw, annotated, epoch = self._queue.popleft()
            try:
                self.recorder.write(raw, annotated, epoch)
            except Exception as exc:
                with self._condition:
                    self._failures += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            else:
                with self._condition:
                    self._written += 1
                    self._last_write_monotonic = time.monotonic()
                    self._last_error = None
