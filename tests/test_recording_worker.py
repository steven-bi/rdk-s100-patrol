from __future__ import annotations

import threading
import time

import numpy as np

from rdk_patrol.recording_worker import AsyncDualStreamRecorder


class _Recorder:
    def __init__(self):
        self.rows = []
        self.closed = False

    def write(self, raw, annotated, timestamp=None):
        self.rows.append((raw.shape, annotated.shape, timestamp))

    def close(self):
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_async_recorder_writes_and_closes() -> None:
    recorder = _Recorder()
    worker = AsyncDualStreamRecorder(recorder)
    worker.start()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    worker.submit(frame, frame, timestamp=10.0)
    deadline = time.monotonic() + 1.0
    while worker.stats().written < 1 and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    worker.stop()
    assert recorder.rows == [((4, 6, 3), (4, 6, 3), 10.0)]
    assert recorder.closed


def test_async_recorder_rate_limits_before_queueing() -> None:
    recorder = _Recorder()
    clock = _Clock()
    worker = AsyncDualStreamRecorder(
        recorder,
        queue_capacity=8,
        max_fps=2.0,
        monotonic_clock=clock,
    )
    worker.start()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    for now in (0.0, 0.1, 0.49, 0.5, 0.9, 1.0):
        clock.now = now
        worker.submit(frame, frame, timestamp=10.0 + now)
    worker.stop()

    stats = worker.stats()
    assert stats.submitted == 6
    assert stats.skipped == 3
    assert stats.written == 3
    assert stats.dropped == 0
    assert [row[2] for row in recorder.rows] == [10.0, 10.5, 11.0]


def test_async_recorder_rate_limit_tracks_long_term_target() -> None:
    recorder = _Recorder()
    clock = _Clock()
    worker = AsyncDualStreamRecorder(
        recorder,
        queue_capacity=200,
        max_fps=15.0,
        monotonic_clock=clock,
    )
    worker.start()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    for index in range(301):
        clock.now = index / 30.0
        worker.submit(frame, frame, timestamp=10.0 + clock.now)
    worker.stop()

    stats = worker.stats()
    assert stats.submitted == 301
    assert stats.written in (150, 151)
    assert stats.skipped == stats.submitted - stats.written
    assert stats.dropped == 0
