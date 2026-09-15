from __future__ import annotations

import numpy as np

from rdk_patrol.contracts import PointContext
from rdk_patrol.garbage_runtime import GarbagePointCoordinator
from rdk_patrol.review import GarbageReviewCollector, PersistentReviewQueue


def test_one_five_second_window_per_visit(tmp_path) -> None:
    queue = PersistentReviewQueue(tmp_path)
    collector = GarbageReviewCollector(queue)
    coordinator = GarbagePointCoordinator(
        collector,
        {"trash_01": {"trash_review"}},
        leave_grace_seconds=1.0,
    )
    context = PointContext(
        "trash_01",
        "垃圾桶点位1",
        "apriltag",
        1.0,
        0.0,
        tag_id=201,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    for value in range(6):
        result = coordinator.process(
            context,
            frame,
            now_monotonic=float(value),
            occurred_at=100.0 + value,
        )
    assert result is not None
    assert queue.pending_count() == 1
    # Remaining on the same point must not immediately start another window.
    coordinator.process(
        context,
        frame,
        now_monotonic=7.0,
        occurred_at=107.0,
    )
    assert coordinator.state().active_point_id is None


def test_incomplete_window_is_aborted_when_point_context_expires(tmp_path) -> None:
    queue = PersistentReviewQueue(tmp_path)
    collector = GarbageReviewCollector(queue)
    coordinator = GarbagePointCoordinator(
        collector,
        {"trash_01": {"trash_review"}},
        leave_grace_seconds=1.0,
    )
    context = PointContext(
        "trash_01",
        "垃圾桶点位1",
        "apriltag",
        1.0,
        0.0,
        tag_id=201,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    coordinator.process(
        context,
        frame,
        now_monotonic=0.0,
        occurred_at=100.0,
    )
    coordinator.process(
        None,
        frame,
        now_monotonic=1.0,
        occurred_at=101.0,
    )

    state = coordinator.state()
    assert state.active_point_id is None
    assert state.windows_aborted == 1
    assert queue.pending_count() == 0
