from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdk_patrol.navigation import NAVIGATION_HOLD_SECONDS, NavigationBridge
from rdk_patrol.recording import (
    VIDEO_RETENTION_SECONDS,
    SegmentedVideoRecorder,
    VideoRetentionCleaner,
)
from rdk_patrol.review import (
    MINIMAX_GARBAGE_PROMPT,
    BestFrameWindow,
    GarbageReviewWorker,
    MiniMaxVisionClient,
    PersistentReviewQueue,
    parse_minimax_decision,
)


class _FakeResponse:
    def __init__(self, value: str) -> None:
        self.value = value

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": self.value}}]},
            ensure_ascii=False,
        ).encode("utf-8")


class _CaptureUrlOpen:
    def __init__(self, reply: str = "已满") -> None:
        self.reply = reply
        self.body: Optional[dict] = None

    def __call__(self, request: Any, timeout: float) -> _FakeResponse:
        self.body = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(self.reply)


def test_best_whole_frame_from_fixed_five_second_window() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue = PersistentReviewQueue(tmp)
        window = BestFrameWindow("trash-1", "垃圾桶一号", started_monotonic=10.0)
        smooth = np.full((80, 120, 3), 120, dtype=np.uint8)
        sharp = smooth.copy()
        sharp[:, ::2] = 0
        assert window.consider(smooth, 10.1, 1000.0)
        assert window.consider(sharp, 14.9, 1001.0)
        assert not window.consider(np.zeros_like(sharp), 15.1, 1002.0)
        assert window.enqueue(queue, 14.99) is None
        review = window.enqueue(queue, 15.0)
        assert review is not None
        selected = queue.read_frame(review)
        assert selected is not None
        assert selected.shape == sharp.shape


def test_minimax_fixed_prompt_and_configured_uncropped_resize() -> None:
    capture = _CaptureUrlOpen()
    client = MiniMaxVisionClient.from_system_config(
        {
            "minimax": {
                "base_url": "https://api.minimaxi.com/v1",
                "model": "MiniMax-M3",
                "api_key_env": "UNUSED",
                "timeout_seconds": 2.0,
                "jpeg_quality": 73,
                "image_detail": "high",
                "max_long_side_pixel": 100,
                "service_tier": "priority",
                # A config prompt must be ignored even if someone adds it.
                "prompt": "错误提示词",
            }
        },
        api_key="test-key",
        urlopen=capture,
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = (0, 255, 0)
    assert client.classify(frame) == "已满"
    assert capture.body is not None
    content = capture.body["messages"][0]["content"]
    assert content[0]["text"] == MINIMAX_GARBAGE_PROMPT
    assert content[1]["image_url"]["detail"] == "high"
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    uploaded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert uploaded.shape[:2] == (50, 100)
    assert abs(uploaded.shape[1] / uploaded.shape[0] - 2.0) < 0.01


def test_strict_reply_and_persistent_cooldown_reuse_without_second_api_call() -> None:
    for invalid in ("", "已经满了", "已满 或 未满", '"已满"', "已满。"):
        try:
            parse_minimax_decision(invalid)
            raise AssertionError("invalid MiniMax reply was accepted")
        except ValueError:
            pass

    class Classifier:
        calls = 0

        def classify(self, frame: np.ndarray) -> str:
            self.calls += 1
            return "已满"

    class Publisher:
        calls = 0

        def try_publish(self, candidate: Any) -> Optional[dict]:
            self.calls += 1
            if self.calls == 1:
                return None
            return {
                "keyframe_image": "images/a.jpg",
                "event_name": "垃圾桶已满",
                "beijing_time": "2026-01-01 00:00:00",
                "point_name": candidate.point_name,
            }

        def remaining_seconds(self, candidate: Any) -> float:
            return 10.0

    with tempfile.TemporaryDirectory() as tmp:
        queue = PersistentReviewQueue(tmp)
        queue.enqueue(
            np.zeros((40, 60, 3), dtype=np.uint8),
            "trash-1",
            "垃圾桶一号",
            occurred_at=100.0,
            quality_score=1.0,
        )
        classifier = Classifier()
        publisher = Publisher()
        worker = GarbageReviewWorker(
            queue,
            classifier,
            publisher,
            retry_delay_seconds=2.0,
        )
        first = worker.process_one(now_epoch=1000.0)
        assert first is not None and first["status"] == "cooldown"
        pending = queue.ready(now_epoch=1009.0)
        assert pending == []
        persisted = queue.ready(now_epoch=1010.0)
        assert len(persisted) == 1 and persisted[0].decision == "已满"
        second = worker.process_one(now_epoch=1010.0)
        assert second is not None and second["status"] == "alarm_saved"
        assert classifier.calls == 1
        assert publisher.calls == 2
        assert queue.pending_count() == 0


class _FakeWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames: List[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def test_segmented_fake_writer_and_video_only_72_hour_cleanup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writers: List[_FakeWriter] = []

        def factory(path: Path, fps: float, size: tuple) -> _FakeWriter:
            assert fps == 15.0
            assert size == (60, 40)
            writer = _FakeWriter(path)
            writers.append(writer)
            return writer

        recorder = SegmentedVideoRecorder(
            tmp,
            "raw",
            fps=15.0,
            segment_seconds=5.0,
            writer_factory=factory,
        )
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        first_path = recorder.write(frame, timestamp=1000.0)
        assert recorder.write(frame, timestamp=1004.9) == first_path
        second_path = recorder.write(frame, timestamp=1005.0)
        assert second_path != first_path
        assert len(writers) == 2
        assert writers[0].released
        recorder.close()
        assert writers[1].released

        video_root = Path(tmp) / "retention"
        video_root.mkdir()
        old_video = video_root / "old.mp4"
        old_alarm_image = video_root / "alarm.jpg"
        old_alarm_json = video_root / "alarm.json"
        recent_video = video_root / "recent.mp4"
        for path in (old_video, old_alarm_image, old_alarm_json, recent_video):
            path.write_bytes(b"x")
        now = 1_000_000.0
        old = now - VIDEO_RETENTION_SECONDS - 1
        os.utime(old_video, (old, old))
        os.utime(old_alarm_image, (old, old))
        os.utime(old_alarm_json, (old, old))
        os.utime(recent_video, (now, now))
        deleted = VideoRetentionCleaner(video_root, retention_hours=72).cleanup(
            now_epoch=now
        )
        assert old_video.resolve() in deleted
        assert not old_video.exists()
        assert old_alarm_image.exists()
        assert old_alarm_json.exists()
        assert recent_video.exists()

        custom_video = video_root / "custom.mp4"
        custom_video.write_bytes(b"x")
        two_hours_old = now - 2 * 60 * 60
        os.utime(custom_video, (two_hours_old, two_hours_old))
        VideoRetentionCleaner(video_root, retention_hours=1).cleanup(now_epoch=now)
        assert not custom_video.exists()


def test_navigation_interface_has_four_actions_and_fixed_hold() -> None:
    events = []
    bridge = NavigationBridge(clock=lambda: 123.0)
    bridge.subscribe(events.append)
    assert bridge.arrive("p1", "点位一").hold_seconds == NAVIGATION_HOLD_SECONDS
    bridge.pause()
    bridge.resume()
    bridge.leave()
    assert [event.action for event in events] == ["arrive", "pause", "resume", "leave"]
    assert all(event.hold_seconds == 5.0 for event in events)
