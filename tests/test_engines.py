from __future__ import annotations

from datetime import time as clock_time
import unittest

import numpy as np

from rdk_patrol.contracts import DepthEstimate, Detection, PointContext
from rdk_patrol.engines import (
    CROWD_EVENT_NAME,
    FLAME_EVENT_NAME,
    PARKING_EVENT_NAME,
    PERSON_EVENT_NAME,
    FlameEngine,
    NightPeopleEngine,
    ParkingEngine,
)


FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def det(
    class_name: str,
    *,
    track_id: int,
    confidence: float = 0.9,
    box: tuple[float, float, float, float] = (30, 20, 70, 90),
) -> Detection:
    class_ids = {"person": 0, "vehicle": 1, "fire": 4, "smoke": 5}
    return Detection(
        class_id=class_ids[class_name],
        class_name=class_name,
        confidence=confidence,
        box=box,
        track_id=track_id,
    )


def point() -> PointContext:
    return PointContext(
        point_id="parking-01",
        point_name="禁停点01",
        source="apriltag",
        confidence=0.99,
        observed_at=0.0,
        tag_id=11,
        rois={"vehicle_parking": [(10, 10), (100, 10), (100, 110), (10, 110)]},
    )


class ParkingEngineTest(unittest.TestCase):
    def test_requires_valid_point_roi_dwell_and_commit_cooldown(self) -> None:
        engine = ParkingEngine(dwell_seconds=2.0, cooldown_seconds=300.0)
        vehicle = det("vehicle", track_id=8)
        self.assertEqual([], engine.process([vehicle], point(), FRAME, now=0.0))
        self.assertEqual([], engine.process([vehicle], point(), FRAME, now=1.0))
        result = engine.process([vehicle], point(), FRAME, now=2.0)
        self.assertEqual(1, len(result))
        self.assertEqual(PARKING_EVENT_NAME, result[0].event_name)
        self.assertEqual("禁停点01", result[0].point_name)
        self.assertEqual("apriltag", result[0].evidence["point_source"])
        self.assertTrue(engine.commit(result[0]))
        self.assertEqual([], engine.process([vehicle], point(), FRAME, now=100.0))
        repeated = engine.process([vehicle], point(), FRAME, now=302.0)
        self.assertEqual(1, len(repeated))

    def test_release_allows_persistence_retry(self) -> None:
        engine = ParkingEngine(dwell_seconds=0.0)
        result = engine.process([det("vehicle", track_id=1)], point(), FRAME, now=0.0)
        self.assertEqual(1, len(result))
        self.assertTrue(engine.release(result[0]))
        retry = engine.process(
            [det("vehicle", track_id=1)], point(), FRAME, now=0.1
        )
        self.assertEqual(1, len(retry))
        self.assertNotEqual(result[0].token, retry[0].token)

    def test_invalid_context_cannot_trigger(self) -> None:
        engine = ParkingEngine(dwell_seconds=0.0)
        invalid = PointContext("", "", "", 0.0, 0.0, rois=point().rois)
        self.assertEqual(
            [],
            engine.process([det("vehicle", track_id=1)], invalid, FRAME, now=0.0),
        )


class NightPeopleEngineTest(unittest.TestCase):
    def test_cross_midnight_person_dwell_and_30_second_rearm(self) -> None:
        engine = NightPeopleEngine(
            person_dwell_seconds=5.0,
            rearm_absence_seconds=30.0,
        )
        person = det("person", track_id=1)
        self.assertEqual(
            [],
            engine.process([person], FRAME, now=0.0, clock_time=clock_time(23, 0)),
        )
        result = engine.process(
            [person], FRAME, now=5.0, clock_time=clock_time(23, 0)
        )
        self.assertEqual(1, len(result))
        self.assertEqual(PERSON_EVENT_NAME, result[0].event_name)
        self.assertIsNone(result[0].point_name)
        self.assertTrue(engine.commit(result[0]))
        engine.process([], FRAME, now=6.0, clock_time=clock_time(23, 1))
        engine.process([], FRAME, now=36.0, clock_time=clock_time(23, 1))
        engine.process([person], FRAME, now=37.0, clock_time=clock_time(23, 2))
        rearmed = engine.process(
            [person], FRAME, now=42.0, clock_time=clock_time(23, 2)
        )
        self.assertEqual(1, len(rearmed))

    def test_crowd_supersedes_person_event(self) -> None:
        engine = NightPeopleEngine(
            person_dwell_seconds=5.0,
            crowd_min_count=3,
            crowd_dwell_seconds=5.0,
        )
        people = [det("person", track_id=index) for index in (1, 2, 3)]
        engine.process(people, FRAME, now=0.0, clock_time="00:30")
        result = engine.process(people, FRAME, now=5.0, clock_time="00:30")
        self.assertEqual(1, len(result))
        self.assertEqual(CROWD_EVENT_NAME, result[0].event_name)
        self.assertEqual(3, result[0].evidence["person_count"])

    def test_daytime_does_not_alarm_and_window_is_adjustable(self) -> None:
        person = det("person", track_id=1)
        default_engine = NightPeopleEngine(person_dwell_seconds=0.0)
        self.assertEqual(
            [],
            default_engine.process(
                [person], FRAME, now=0.0, clock_time=clock_time(12, 0)
            ),
        )
        daytime_test_engine = NightPeopleEngine(
            active_start="00:00",
            active_end="00:00",
            person_dwell_seconds=0.0,
        )
        self.assertEqual(
            1,
            len(
                daytime_test_engine.process(
                    [person], FRAME, now=0.0, clock_time=clock_time(12, 0)
                )
            ),
        )


class FlameEngineTest(unittest.TestCase):
    def test_fire_only_size_confidence_and_depth_failure_does_not_gate(self) -> None:
        engine = FlameEngine(
            confirmation_seconds=2.0,
            min_width_px=10,
            min_height_px=10,
            min_area_px=200,
        )
        fire = det("fire", track_id=4, box=(20, 20, 50, 60))
        smoke = det("smoke", track_id=5, box=(20, 20, 80, 90))
        engine.process([fire, smoke], FRAME, now=0.0)
        result = engine.process(
            [fire, smoke],
            FRAME,
            now=2.0,
            depth_estimate=DepthEstimate.unavailable("calibration_missing"),
        )
        self.assertEqual(1, len(result))
        self.assertEqual(FLAME_EVENT_NAME, result[0].event_name)
        self.assertEqual("不可用", result[0].evidence["distance_text"])
        self.assertFalse(result[0].evidence["depth_valid"])

    def test_commit_enables_ten_minute_repeat_and_valid_distance(self) -> None:
        engine = FlameEngine(
            confirmation_seconds=0.0,
            repeat_alarm_seconds=600.0,
        )
        fire = det("fire", track_id=4, box=(20, 20, 50, 60))
        first = engine.process([fire], FRAME, now=2.0)
        self.assertEqual(1, len(first))
        self.assertTrue(engine.commit(first[0]))
        self.assertEqual([], engine.process([fire], FRAME, now=601.9))
        repeated = engine.process(
            [fire],
            FRAME,
            now=602.0,
            depth_estimate=DepthEstimate(
                valid=True,
                distance_m=3.25,
                reason="ok",
                valid_pixels=120,
            ),
        )
        self.assertEqual(1, len(repeated))
        self.assertEqual("repeat", repeated[0].evidence["reason"])
        self.assertEqual(3.25, repeated[0].evidence["distance_m"])

    def test_release_retries_without_suppressing_alarm(self) -> None:
        engine = FlameEngine(confirmation_seconds=0.0)
        fire = det("fire", track_id=4, box=(20, 20, 50, 60))
        first = engine.process([fire], FRAME, now=0.0)
        self.assertTrue(engine.release(first[0]))
        second = engine.process([fire], FRAME, now=0.1)
        self.assertEqual(1, len(second))
        self.assertNotEqual(first[0].token, second[0].token)


if __name__ == "__main__":
    unittest.main()
