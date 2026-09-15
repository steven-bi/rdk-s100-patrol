from __future__ import annotations

"""A single-owner, latest-frame camera hub.

The detector must never build a queue of stale patrol images.  ``LatestFrameSlot``
therefore has an effective capacity of one: publishing a newer envelope replaces
the previous value.  Multiple readers may independently ask for a sequence newer
than the one they last consumed.

``FrameHub`` can either be fed explicitly with :meth:`publish` (useful for ROS
callbacks) or own a small reader thread around an OpenCV-like source.  The module
has no ROS or RDK import and is consequently unit-testable on a development PC.
"""

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from ..contracts import FrameEnvelope


_ROTATION_CODES = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "rot180": cv2.ROTATE_180,
}


@dataclass(frozen=True)
class FrameHubStats:
    published: int
    replacements: int
    read_failures: int
    last_sequence: int
    running: bool
    last_error: str | None


class LatestFrameSlot:
    """Thread-safe one-item frame buffer."""

    capacity = 1

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FrameEnvelope | None = None
        self._closed = False
        self._publish_count = 0
        self._replacement_count = 0

    def publish(self, envelope: FrameEnvelope) -> None:
        if not isinstance(envelope, FrameEnvelope):
            raise TypeError("envelope must be a FrameEnvelope")
        with self._condition:
            if self._closed:
                raise RuntimeError("latest-frame slot is closed")
            if self._latest is not None:
                self._replacement_count += 1
            self._latest = envelope
            self._publish_count += 1
            self._condition.notify_all()

    def latest(
        self,
        *,
        after_sequence: int | None = None,
        timeout: float | None = None,
        copy: bool = False,
        include_combined: bool = True,
    ) -> FrameEnvelope | None:
        """Return the newest frame, optionally waiting for a newer sequence.

        ``timeout=0`` performs a non-blocking read.  ``None`` waits without a
        deadline only when ``after_sequence`` is supplied; an ordinary
        ``latest()`` call always returns immediately.
        """

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                candidate = self._latest
                is_new = candidate is not None and (
                    after_sequence is None or candidate.sequence > int(after_sequence)
                )
                if is_new:
                    if not copy:
                        return candidate
                    return candidate.copy_for_worker(include_combined=include_combined)
                if self._closed:
                    return None
                if after_sequence is None:
                    return None
                if timeout is not None:
                    assert deadline is not None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def publish_count(self) -> int:
        with self._condition:
            return self._publish_count

    @property
    def replacement_count(self) -> int:
        with self._condition:
            return self._replacement_count


# A descriptive alias used by some callers and tests.
LatestFrameBuffer = LatestFrameSlot


def split_stereo_frame(
    combined_bgr: np.ndarray,
    *,
    layout: str = "vertical",
    detection_view: str = "bottom",
    auxiliary_view: str = "top",
) -> tuple[np.ndarray, np.ndarray]:
    """Split a GS130W combined frame without guessing physical camera identity."""

    _validate_image(combined_bgr, "combined_bgr")
    layout_name = str(layout).strip().lower().replace("-", "_")
    height, width = combined_bgr.shape[:2]
    if layout_name in {"vertical", "top_bottom", "topbottom"}:
        if height % 2:
            raise ValueError(f"vertical stereo frame height must be even, got {height}")
        half = height // 2
        views = {"top": combined_bgr[:half], "bottom": combined_bgr[half:]}
    elif layout_name in {"horizontal", "left_right", "leftright"}:
        if width % 2:
            raise ValueError(f"horizontal stereo frame width must be even, got {width}")
        half = width // 2
        views = {"left": combined_bgr[:, :half], "right": combined_bgr[:, half:]}
    elif layout_name in {"mono", "full"}:
        if detection_view not in {"full", "mono"}:
            raise ValueError("mono layout requires detection_view='full' or 'mono'")
        return combined_bgr, combined_bgr
    else:
        raise ValueError(f"unsupported stereo layout: {layout!r}")
    if detection_view not in views:
        raise ValueError(
            f"detection_view={detection_view!r} is unavailable; choices={sorted(views)}"
        )
    if auxiliary_view not in views:
        raise ValueError(
            f"auxiliary_view={auxiliary_view!r} is unavailable; choices={sorted(views)}"
        )
    if detection_view == auxiliary_view:
        raise ValueError("detection_view and auxiliary_view must differ")
    return views[detection_view], views[auxiliary_view]


def build_frame_envelope(
    combined_bgr: np.ndarray,
    *,
    sequence: int,
    monotonic_ts: float | None = None,
    source_ts: float | None = None,
    layout: str = "vertical",
    detection_view: str = "bottom",
    auxiliary_view: str = "top",
    rotation: str = "none",
    metadata: Mapping[str, Any] | None = None,
) -> FrameEnvelope:
    _validate_image(combined_bgr, "combined_bgr")
    detection, auxiliary = split_stereo_frame(
        combined_bgr,
        layout=layout,
        detection_view=detection_view,
        auxiliary_view=auxiliary_view,
    )
    rotation_name = str(rotation).strip().lower()
    if rotation_name not in _ROTATION_CODES:
        raise ValueError(f"unsupported rotation: {rotation!r}")
    code = _ROTATION_CODES[rotation_name]
    if code is not None:
        detection = cv2.rotate(detection, code)
        auxiliary = cv2.rotate(auxiliary, code)
    envelope_metadata = dict(metadata or {})
    envelope_metadata.update(
        {
            "combined_layout": layout,
            "detection_view": detection_view,
            "auxiliary_view": auxiliary_view,
            "rotation": rotation_name,
        }
    )
    return FrameEnvelope(
        sequence=int(sequence),
        monotonic_ts=time.monotonic() if monotonic_ts is None else float(monotonic_ts),
        source_ts=None if source_ts is None else float(source_ts),
        combined_bgr=combined_bgr,
        detection_bgr=detection,
        auxiliary_bgr=auxiliary,
        metadata=envelope_metadata,
    )


class FrameHub:
    """Own one camera source and fan out its latest frame to every subsystem."""

    def __init__(
        self,
        source: Any | None = None,
        *,
        layout: str = "vertical",
        detection_view: str = "bottom",
        auxiliary_view: str = "top",
        rotation: str = "none",
        idle_wait_seconds: float = 0.005,
        failure_wait_seconds: float = 0.05,
    ) -> None:
        self.source = source
        self.layout = str(layout)
        self.detection_view = str(detection_view)
        self.auxiliary_view = str(auxiliary_view)
        self.rotation = str(rotation)
        self.idle_wait_seconds = max(0.0, float(idle_wait_seconds))
        self.failure_wait_seconds = max(0.0, float(failure_wait_seconds))
        self.slot = LatestFrameSlot()
        self._state_lock = threading.Lock()
        self._sequence = 0
        self._read_failures = 0
        self._last_error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def publish(
        self,
        combined_bgr: np.ndarray,
        *,
        source_ts: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        monotonic_ts: float | None = None,
    ) -> FrameEnvelope:
        with self._state_lock:
            self._sequence += 1
            sequence = self._sequence
        envelope = build_frame_envelope(
            combined_bgr,
            sequence=sequence,
            monotonic_ts=monotonic_ts,
            source_ts=source_ts,
            layout=self.layout,
            detection_view=self.detection_view,
            auxiliary_view=self.auxiliary_view,
            rotation=self.rotation,
            metadata=metadata,
        )
        self.slot.publish(envelope)
        return envelope

    def latest(
        self,
        *,
        after_sequence: int | None = None,
        timeout: float | None = None,
        copy: bool = False,
        include_combined: bool = True,
    ) -> FrameEnvelope | None:
        return self.slot.latest(
            after_sequence=after_sequence,
            timeout=timeout,
            copy=copy,
            include_combined=include_combined,
        )

    def start(self) -> None:
        if self.source is None:
            raise RuntimeError("FrameHub.start() requires a source")
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._reader_loop,
                name="rdk-frame-hub",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 2.0, close_slot: bool = False) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, float(timeout)))
        if close_slot:
            self.slot.close()

    def stats(self) -> FrameHubStats:
        with self._state_lock:
            thread = self._thread
            return FrameHubStats(
                published=self.slot.publish_count,
                replacements=self.slot.replacement_count,
                read_failures=self._read_failures,
                last_sequence=self._sequence,
                running=bool(thread is not None and thread.is_alive()),
                last_error=self._last_error,
            )

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._read_source()
                frame, source_ts, metadata = _normalize_source_payload(payload)
                if frame is None:
                    self._record_failure("source returned no frame")
                    self._stop_event.wait(self.failure_wait_seconds)
                    continue
                self.publish(frame, source_ts=source_ts, metadata=metadata)
                with self._state_lock:
                    self._last_error = None
                if self.idle_wait_seconds:
                    self._stop_event.wait(self.idle_wait_seconds)
            except Exception as exc:  # a source failure must not kill the process
                self._record_failure(f"{type(exc).__name__}: {exc}")
                self._stop_event.wait(self.failure_wait_seconds)

    def _read_source(self) -> Any:
        if callable(self.source):
            return self.source()
        reader = getattr(self.source, "read", None)
        if not callable(reader):
            raise TypeError("frame source must be callable or expose read()")
        return reader()

    def _record_failure(self, message: str) -> None:
        with self._state_lock:
            self._read_failures += 1
            self._last_error = str(message)


def _normalize_source_payload(
    payload: Any,
) -> tuple[np.ndarray | None, float | None, Mapping[str, Any] | None]:
    if isinstance(payload, np.ndarray):
        return payload, None, None
    if payload is None:
        return None, None, None
    if isinstance(payload, tuple):
        if len(payload) == 2 and isinstance(payload[0], (bool, np.bool_)):
            return (payload[1] if bool(payload[0]) else None), None, None
        if len(payload) == 2:
            return payload[0], payload[1], None
        if len(payload) == 3:
            return payload[0], payload[1], payload[2]
    raise TypeError(
        "source must return an image, (ok, image), "
        "(image, source_ts), or (image, source_ts, metadata)"
    )


def _validate_image(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in {2, 3}:
        raise ValueError(f"{name} must be a non-empty image array")
