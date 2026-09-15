from __future__ import annotations

import numpy as np

from rdk_patrol.video_source import OpenCvVideoSource


class _Capture:
    def __init__(self):
        self.calls = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return 20.0

    def read(self):
        self.calls += 1
        if self.calls == 1:
            return True, np.zeros((4, 6, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_video_source_reports_eof() -> None:
    capture = _Capture()
    source = OpenCvVideoSource("dummy.mp4", pace=False, capture=capture)
    frame, _timestamp, metadata = source.read()
    assert frame is not None
    assert metadata["replay_frame"] == 1
    frame, _timestamp, metadata = source.read()
    assert frame is None
    assert metadata["replay_eof"]
    assert source.eof
    source.close()
    assert capture.released
