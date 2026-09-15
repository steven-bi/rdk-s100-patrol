from __future__ import annotations

import threading
import time

import numpy as np

from rdk_patrol.async_workers import AsyncDepthEstimator, AsyncPointResolver
from rdk_patrol.contracts import DepthEstimate, PointContext


class _Resolver:
    def resolve(self, frame, observed_at):
        return PointContext(
            point_id="p1",
            point_name="点位一",
            source="apriltag",
            confidence=1.0,
            observed_at=observed_at,
        )


class _Estimator:
    def estimate(self, left, right, fire_box, computed_at):
        return DepthEstimate(
            valid=True,
            distance_m=2.5,
            reason="ok",
            valid_pixels=50,
            computed_at=computed_at,
        )


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.005)
    return False


def test_async_point_resolver() -> None:
    worker = AsyncPointResolver(_Resolver(), minimum_interval_seconds=0.0)
    worker.start()
    try:
        worker.submit(np.zeros((10, 10, 3), dtype=np.uint8), time.monotonic())
        assert _wait_for(lambda: worker.stats().completed == 1)
        context = worker.latest(max_age_seconds=1.0)
        assert context is not None
        assert context.point_id == "p1"
    finally:
        worker.stop()


def test_async_depth_estimator() -> None:
    worker = AsyncDepthEstimator(_Estimator(), minimum_interval_seconds=0.0)
    worker.start()
    now = time.monotonic()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        worker.submit(image, image, (1, 1, 5, 5), now)
        assert _wait_for(lambda: worker.stats().completed == 1)
        result = worker.latest(
            now=time.monotonic(),
            max_age_seconds=1.0,
            fire_box=(1, 1, 5, 5),
        )
        assert result.valid
        assert result.distance_m == 2.5
        mismatch = worker.latest(
            now=time.monotonic(),
            max_age_seconds=1.0,
            fire_box=(7, 7, 9, 9),
        )
        assert not mismatch.valid
        assert mismatch.reason == "stereo_result_box_mismatch"
    finally:
        worker.stop()
