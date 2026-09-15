from __future__ import annotations

"""Paced OpenCV replay source for field recordings and acceptance tests."""

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from .io import FrameHub


@dataclass(frozen=True)
class VideoSourceStats:
    path: str
    fps: float
    frames_read: int
    eof: bool
    last_error: str | None


class OpenCvVideoSource:
    def __init__(
        self,
        path: str | Path,
        *,
        fps: float | None = None,
        pace: bool = True,
        loop: bool = False,
        capture: Any | None = None,
    ) -> None:
        self.path = Path(path)
        self.capture = capture or cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise OSError(f"failed to open replay video: {self.path}")
        detected_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = float(fps) if fps and fps > 0 else (
            detected_fps if detected_fps > 0 else 15.0
        )
        self.pace = bool(pace)
        self.loop = bool(loop)
        self._interval = 1.0 / self.fps
        self._next_due: float | None = None
        self._frames_read = 0
        self._eof = False
        self._last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def eof(self) -> bool:
        with self._lock:
            return self._eof

    def read(
        self,
    ) -> tuple[np.ndarray | None, float | None, dict[str, Any]]:
        if self.pace:
            now = time.monotonic()
            if self._next_due is None:
                self._next_due = now
            delay = self._next_due - now
            if delay > 0:
                threading.Event().wait(delay)
            self._next_due = max(self._next_due + self._interval, time.monotonic())
        ok, frame = self.capture.read()
        if not ok or frame is None:
            if self.loop:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.capture.read()
            if not ok or frame is None:
                with self._lock:
                    self._eof = True
                return None, None, {"replay_eof": True}
        with self._lock:
            self._frames_read += 1
            frame_index = self._frames_read
            self._eof = False
            self._last_error = None
        return (
            np.ascontiguousarray(frame),
            time.time(),
            {"replay_path": str(self.path), "replay_frame": frame_index},
        )

    def close(self) -> None:
        self.capture.release()

    def stats(self) -> VideoSourceStats:
        with self._lock:
            return VideoSourceStats(
                path=str(self.path),
                fps=self.fps,
                frames_read=self._frames_read,
                eof=self._eof,
                last_error=self._last_error,
            )


class VideoFrameHubCamera:
    """Lifecycle adapter that closes both the FrameHub thread and VideoCapture."""

    def __init__(self, frame_hub: FrameHub, source: OpenCvVideoSource) -> None:
        self.frame_hub = frame_hub
        self.source = source

    def start(self) -> None:
        self.frame_hub.start()

    def stop(self) -> None:
        self.frame_hub.stop()
        self.source.close()

    def stats(self) -> VideoSourceStats:
        return self.source.stats()
