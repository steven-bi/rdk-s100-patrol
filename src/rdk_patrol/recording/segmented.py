from __future__ import annotations

import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np


VIDEO_RETENTION_HOURS = 72
VIDEO_RETENTION_SECONDS = VIDEO_RETENTION_HOURS * 60 * 60
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".mov"})


class VideoWriterLike(Protocol):
    def write(self, frame: np.ndarray) -> None:
        ...

    def release(self) -> None:
        ...


WriterFactory = Callable[[Path, float, Tuple[int, int]], VideoWriterLike]


class OpenCVWriterFactory:
    def __init__(self, codec: str = "mp4v") -> None:
        if len(codec) != 4:
            raise ValueError("codec must contain four characters")
        self.codec = codec

    def __call__(
        self,
        path: Path,
        fps: float,
        frame_size: Tuple[int, int],
    ) -> VideoWriterLike:
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)
        if not writer.isOpened():
            writer.release()
            raise OSError("failed to open video writer: {}".format(path))
        return writer


def _safe_stream_name(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value)).strip("_")
    return result[:40] or "video"


class VideoRetentionCleaner:
    """Delete only expired video files below the configured video root."""

    def __init__(
        self,
        video_root: Union[str, Path],
        retention_hours: float = VIDEO_RETENTION_HOURS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        self.video_root = Path(video_root).resolve()
        self.retention_hours = float(retention_hours)
        self.retention_seconds = self.retention_hours * 60.0 * 60.0
        self.clock = clock

    def cleanup(self, now_epoch: Optional[float] = None) -> List[Path]:
        now = self.clock() if now_epoch is None else float(now_epoch)
        cutoff = now - self.retention_seconds
        deleted: List[Path] = []
        if not self.video_root.is_dir():
            return deleted
        for candidate in self.video_root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.video_root)
                if resolved.stat().st_mtime >= cutoff:
                    continue
                resolved.unlink()
                deleted.append(resolved)
            except (OSError, ValueError):
                continue
        return deleted


class SegmentedVideoRecorder:
    """Rotate one video stream into bounded-duration segments."""

    def __init__(
        self,
        video_root: Union[str, Path],
        stream_name: str,
        fps: float = 15.0,
        segment_seconds: float = 300.0,
        writer_factory: Optional[WriterFactory] = None,
        clock: Callable[[], float] = time.time,
        suffix: str = ".mp4",
        retention_hours: float = VIDEO_RETENTION_HOURS,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        suffix = suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError("unsupported video suffix")
        self.video_root = Path(video_root).resolve()
        self.stream_name = _safe_stream_name(stream_name)
        self.output_dir = self.video_root / self.stream_name
        self.fps = float(fps)
        self.segment_seconds = float(segment_seconds)
        self.writer_factory = writer_factory or OpenCVWriterFactory()
        self.clock = clock
        self.suffix = suffix
        self.cleaner = VideoRetentionCleaner(
            self.video_root,
            retention_hours=retention_hours,
            clock=clock,
        )
        self._writer: Optional[VideoWriterLike] = None
        self._segment_started_at: Optional[float] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._current_path: Optional[Path] = None
        self._lock = threading.RLock()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path

    def write(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Path:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("video frame must be a non-empty HxWx3 BGR image")
        now = self.clock() if timestamp is None else float(timestamp)
        size = (int(frame.shape[1]), int(frame.shape[0]))
        with self._lock:
            rotate = (
                self._writer is None
                or self._segment_started_at is None
                or now - self._segment_started_at >= self.segment_seconds
                or self._frame_size != size
            )
            if rotate:
                self._open_segment(now, size)
            assert self._writer is not None
            assert self._current_path is not None
            self._writer.write(frame)
            return self._current_path

    def _open_segment(self, timestamp: float, frame_size: Tuple[int, int]) -> None:
        self._release_unlocked()
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        date_dir = self.output_dir / date.strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        filename = "{}_{}_{}{}".format(
            self.stream_name,
            date.strftime("%H%M%S_%f"),
            uuid.uuid4().hex[:8],
            self.suffix,
        )
        path = date_dir / filename
        writer = self.writer_factory(path, self.fps, frame_size)
        self._writer = writer
        self._segment_started_at = timestamp
        self._frame_size = frame_size
        self._current_path = path
        self.cleaner.cleanup(now_epoch=timestamp)

    def close(self) -> None:
        with self._lock:
            self._release_unlocked()

    def _release_unlocked(self) -> None:
        writer = self._writer
        self._writer = None
        self._segment_started_at = None
        self._frame_size = None
        self._current_path = None
        if writer is not None:
            writer.release()

    def __enter__(self) -> "SegmentedVideoRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class DualStreamRecorder:
    """Record complete raw and annotated streams with identical timestamps."""

    def __init__(
        self,
        video_root: Union[str, Path],
        fps: float = 15.0,
        segment_seconds: float = 300.0,
        writer_factory: Optional[WriterFactory] = None,
        clock: Callable[[], float] = time.time,
        retention_hours: float = VIDEO_RETENTION_HOURS,
    ) -> None:
        self.clock = clock
        self.raw = SegmentedVideoRecorder(
            video_root,
            "raw",
            fps=fps,
            segment_seconds=segment_seconds,
            writer_factory=writer_factory,
            clock=clock,
            retention_hours=retention_hours,
        )
        self.annotated = SegmentedVideoRecorder(
            video_root,
            "annotated",
            fps=fps,
            segment_seconds=segment_seconds,
            writer_factory=writer_factory,
            clock=clock,
            retention_hours=retention_hours,
        )

    def write(
        self,
        raw_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[Path, Path]:
        now = self.clock() if timestamp is None else float(timestamp)
        return self.raw.write(raw_bgr, now), self.annotated.write(annotated_bgr, now)

    def close(self) -> None:
        self.raw.close()
        self.annotated.close()

    def __enter__(self) -> "DualStreamRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
