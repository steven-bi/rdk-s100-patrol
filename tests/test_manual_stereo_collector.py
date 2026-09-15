from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import collect_gs130w_pairs_web as collector


IMAGE_SIZE = (640, 1280)
PATTERN_SIZE = (9, 6)


def board(
    center=(320.0, 640.0),
    spacing=42.0,
    rotation_deg=0.0,
    perspective=0.0,
) -> np.ndarray:
    columns, rows = PATTERN_SIZE
    points = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float64)
    points -= points.mean(axis=0)
    points *= spacing
    angle = math.radians(rotation_deg)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    points = points @ rotation.T
    if perspective:
        normalized_x = points[:, 0] / max(float(np.max(np.abs(points[:, 0]))), 1.0)
        denominator = 1.0 + perspective * normalized_x
        points[:, 0] /= denominator
        points[:, 1] /= denominator
    points += np.asarray(center)
    return points.reshape(-1, 1, 2).astype(np.float32)


def pair_pose(
    center=(320.0, 640.0),
    spacing=42.0,
    rotation_deg=0.0,
    perspective=0.0,
) -> collector.PairPose:
    left = board(center, spacing, rotation_deg, perspective)
    right = board(
        (center[0] - 45.0, center[1]),
        spacing,
        rotation_deg,
        perspective,
    )
    return collector.pair_pose_signature(
        left, right, IMAGE_SIZE, PATTERN_SIZE
    )


def test_pose_gate_rejects_duplicate_and_accepts_clear_changes() -> None:
    first = pair_pose()
    duplicate = pair_pose(center=(322.0, 641.0), rotation_deg=1.0)
    unique, nearest, score = collector.nearest_saved_pose(
        duplicate, [first], collector.PoseThresholds()
    )
    assert not unique
    assert nearest == 1
    assert score < 1.0

    for changed in (
        pair_pose(center=(390.0, 640.0)),
        pair_pose(spacing=50.0),
        pair_pose(rotation_deg=12.0),
        pair_pose(perspective=0.25),
    ):
        unique, nearest, score = collector.nearest_saved_pose(
            changed, [first], collector.PoseThresholds()
        )
        assert unique
        assert nearest == 1
        assert score >= 1.0


def test_pose_gate_compares_with_every_saved_pose() -> None:
    first = pair_pose()
    second = pair_pose(center=(430.0, 640.0))
    returned_to_first = pair_pose(center=(321.0, 640.0))
    unique, nearest, _score = collector.nearest_saved_pose(
        returned_to_first,
        [first, second],
        collector.PoseThresholds(),
    )
    assert not unique
    assert nearest == 1


def test_180_degree_corner_order_flip_is_still_a_duplicate() -> None:
    left = board()
    right = board(center=(275.0, 640.0))
    original = collector.pair_pose_signature(
        left, right, IMAGE_SIZE, PATTERN_SIZE
    )
    flipped = collector.pair_pose_signature(
        left[::-1].copy(),
        right[::-1].copy(),
        IMAGE_SIZE,
        PATTERN_SIZE,
    )
    unique, nearest, score = collector.nearest_saved_pose(
        flipped, [original], collector.PoseThresholds()
    )
    assert not unique
    assert nearest == 1
    assert score < 1e-6


def test_coverage_grid_tracks_all_nine_regions() -> None:
    poses = []
    for row in range(3):
        for column in range(3):
            poses.append(
                pair_pose(
                    center=(
                        (column + 0.5) * IMAGE_SIZE[0] / 3.0 + 22.5,
                        (row + 0.5) * IMAGE_SIZE[1] / 3.0,
                    )
                )
            )
    assert collector.coverage_grid(poses) == [1] * 9


def test_stability_requires_time_and_three_fresh_samples() -> None:
    tracker = collector.StabilityTracker(
        required_seconds=1.0, required_frames=3, max_motion_px=4.0
    )
    left = board()
    right = board(center=(275.0, 640.0))
    first = tracker.update(left, right, 0.0)
    second = tracker.update(left + 0.8, right + 0.8, 0.6)
    third = tracker.update(left + 0.5, right + 0.5, 1.05)
    assert not first[0]
    assert not second[0]
    assert third[0]
    assert third[3] == 3


def test_stability_anchor_prevents_slow_drift() -> None:
    tracker = collector.StabilityTracker(
        required_seconds=1.0, required_frames=3, max_motion_px=4.0
    )
    left = board()
    right = board(center=(275.0, 640.0))
    status = None
    for index in range(7):
        offset = index * 1.5
        status = tracker.update(
            left + offset,
            right + offset,
            index * 0.2,
        )
    assert status is not None
    assert not status[0]
    assert status[1] < 1.0


def test_quality_rejects_too_small_and_outer_border_outside() -> None:
    good = pair_pose(spacing=42.0)
    assert collector._quality_result(good, 0.006, 8.0)[0]

    too_small = pair_pose(spacing=8.0)
    ok, message = collector._quality_result(too_small, 0.006, 8.0)
    assert not ok
    assert "太小" in message

    clipped = pair_pose(center=(75.0, 640.0), spacing=42.0)
    ok, message = collector._quality_result(clipped, 0.006, 8.0)
    assert not ok
    assert "边缘" in message


def test_capture_request_waits_for_next_frame_and_expires() -> None:
    state = collector.CaptureWebState(target=30, capture_timeout_seconds=0.1)
    state.update({"ready": True, "frame_seq": 7})
    accepted, _message = state.request_capture()
    assert accepted
    assert not state.consume_capture_request(7)
    state.note_frame_received(8, received_at=time.monotonic() + 0.01)
    assert state.consume_capture_request(8)
    state.finish_capture_request("saved")
    assert not state.snapshot()[0]["capture_pending"]

    state.update({"ready": True, "frame_seq": 8})
    assert state.request_capture()[0]
    assert state.expire_capture_request(time.monotonic() + 1.0)
    status = state.snapshot()[0]
    assert not status["capture_pending"]
    assert "超时" in status["last_action"]


def test_capture_request_waits_past_frame_already_being_processed() -> None:
    state = collector.CaptureWebState(target=30)
    state.update({"ready": True, "frame_seq": 7})
    state.note_frame_received(8)
    assert state.request_capture()[0]
    assert not state.consume_capture_request(8)
    state.note_frame_received(9, received_at=time.monotonic() + 0.01)
    assert state.consume_capture_request(9)


def test_frame_received_before_click_cannot_be_claimed_after_click() -> None:
    state = collector.CaptureWebState(target=30)
    state.update({"ready": True, "frame_seq": 7})
    received_before_click = time.monotonic()
    assert state.request_capture()[0]
    state.note_frame_received(8, received_at=received_before_click)
    assert not state.consume_capture_request(8)
    state.note_frame_received(9, received_at=time.monotonic() + 0.01)
    assert state.consume_capture_request(9)


def test_timely_new_frame_does_not_expire_during_slow_corner_processing() -> None:
    state = collector.CaptureWebState(target=30, capture_timeout_seconds=0.1)
    state.update({"ready": True, "frame_seq": 2})
    assert state.request_capture()[0]
    state.note_frame_received(3, received_at=time.monotonic() + 0.01)
    assert not state.expire_capture_request(time.monotonic() + 10.0)
    assert state.consume_capture_request(3)


def test_stop_cancels_pending_capture() -> None:
    state = collector.CaptureWebState(target=30)
    state.update({"ready": True, "frame_seq": 4})
    assert state.request_capture()[0]
    state.request_stop()
    assert not state.consume_capture_request(5)
    assert not state.snapshot()[0]["capture_pending"]


def test_web_api_only_accepts_ready_json_capture_and_can_stop() -> None:
    state = collector.CaptureWebState(target=30)
    server = collector.ManualCaptureWebServer(state, "127.0.0.1", 0)
    host, port = server.start()
    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=3.0) as response:
            page = response.read().decode("utf-8")
        assert "采集本组" in page
        assert "安全结束" in page

        request = urllib.request.Request(
            base + "/api/capture",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3.0)
        assert error.value.code == 409

        state.update({"ready": True, "frame_seq": 3})
        with urllib.request.urlopen(request, timeout=3.0) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["accepted"] is True

        stop = urllib.request.Request(
            base + "/api/stop",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(stop, timeout=3.0) as response:
            assert response.status == 202
        assert state.stop_requested()
    finally:
        server.stop()


def test_first_pose_status_is_strict_json_without_infinity() -> None:
    state = collector.CaptureWebState(target=30)
    first_pose_score = collector.nearest_saved_pose(
        pair_pose(), [], collector.PoseThresholds()
    )[2]
    assert math.isinf(first_pose_score)
    state.update(
        {
            "ready": True,
            "left_found": True,
            "right_found": True,
            "quality_ok": True,
            "stable": True,
            "pose_unique": True,
            "pose_difference_score": collector._finite_json_float(first_pose_score),
        }
    )
    server = collector.ManualCaptureWebServer(state, "127.0.0.1", 0)
    host, port = server.start()
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/status", timeout=3.0
        ) as response:
            raw = response.read().decode("utf-8")
        assert "Infinity" not in raw
        assert "NaN" not in raw
        status = json.loads(
            raw,
            parse_constant=lambda value: pytest.fail(
                f"non-standard JSON constant: {value}"
            ),
        )
        assert status["ready"] is True
        assert status["pose_difference_score"] is None
    finally:
        server.stop()


def test_safe_stop_disables_browser_polling_before_request() -> None:
    page = collector.WEB_PAGE
    assert "function stopPolling(removePreview = true)" in page
    assert "clearInterval(statusTimer)" in page
    assert "clearInterval(previewTimer)" in page
    request_stop = page[page.index("async function requestStop()") :]
    assert request_stop.index("stopPolling();") < request_stop.index(
        'fetch("/api/stop"'
    )


def test_completed_status_stops_polling_and_keeps_final_preview() -> None:
    page = collector.WEB_PAGE
    update_status = page[page.index("async function updateStatus()") :]
    assert "if (s.completed) stopPolling(false);" in update_status


def test_output_guard_rejects_unknown_or_partial_content(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "note.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(RuntimeError, match="非本工具"):
        collector._prepare_output(unknown, False, 30, PATTERN_SIZE)

    partial = tmp_path / "partial"
    (partial / "left").mkdir(parents=True)
    (partial / "left" / "0001.png").write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="不完整"):
        collector._prepare_output(partial, True, 30, PATTERN_SIZE)


def test_atomic_pair_save_writes_three_files_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    for name in ("left", "right", "combined"):
        (tmp_path / name).mkdir()
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    collector._save_pair(tmp_path, "0001", image, image, image)
    assert (tmp_path / "left" / "0001.png").is_file()
    assert (tmp_path / "right" / "0001.png").is_file()
    assert (tmp_path / "combined" / "0001.jpg").is_file()
    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        collector._save_pair(tmp_path, "0001", image, image, image)


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_atomic_pair_save_cleans_partial_group_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    for name in ("left", "right", "combined"):
        (tmp_path / name).mkdir()
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    real_save = collector._atomic_save_image
    calls = [0]

    def interrupted(path: Path, frame: np.ndarray) -> None:
        calls[0] += 1
        if calls[0] == failure_call:
            raise KeyboardInterrupt
        real_save(path, frame)

    monkeypatch.setattr(collector, "_atomic_save_image", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collector._save_pair(tmp_path, "0001", image, image, image)
    assert not list((tmp_path / "left").iterdir())
    assert not list((tmp_path / "right").iterdir())
    assert not list((tmp_path / "combined").iterdir())
