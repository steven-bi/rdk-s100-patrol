from __future__ import annotations

import unittest

import numpy as np

from rdk_patrol.contracts import Detection
from rdk_patrol.inference import (
    ClassAwareTracker,
    FakeBackend,
    UnifiedInference,
    YoloDecoder,
)
from rdk_patrol.inference.preprocess import PreprocessMeta, preprocess_bgr
from rdk_patrol.io import FrameHub, LatestFrameSlot


CLASSES = ("person", "vehicle", "trash_bin", "garbage", "fire", "smoke")


def prediction(*rows: tuple[float, float, float, float, int, float]) -> np.ndarray:
    output = np.zeros((1, 4 + len(CLASSES), len(rows)), dtype=np.float32)
    for index, (cx, cy, width, height, class_id, confidence) in enumerate(rows):
        output[0, :4, index] = (cx, cy, width, height)
        output[0, 4 + class_id, index] = confidence
    return output


class FrameHubTest(unittest.TestCase):
    def test_latest_frame_capacity_one_and_vertical_split(self) -> None:
        hub = FrameHub(layout="vertical", detection_view="bottom", auxiliary_view="top")
        first = np.zeros((8, 6, 3), dtype=np.uint8)
        first[4:] = 10
        second = np.zeros((8, 6, 3), dtype=np.uint8)
        second[4:] = 20
        hub.publish(first, monotonic_ts=1.0)
        hub.publish(second, monotonic_ts=2.0)
        latest = hub.latest()
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(2, latest.sequence)
        self.assertEqual((4, 6, 3), latest.detection_bgr.shape)
        self.assertTrue(np.all(latest.detection_bgr == 20))
        self.assertTrue(np.all(latest.auxiliary_bgr == 0))
        self.assertEqual(1, hub.stats().replacements)

    def test_wait_after_sequence_is_non_stale(self) -> None:
        slot = LatestFrameSlot()
        hub = FrameHub()
        envelope = hub.publish(np.zeros((8, 6, 3), dtype=np.uint8))
        slot.publish(envelope)
        self.assertIsNone(slot.latest(after_sequence=envelope.sequence, timeout=0))


class PreprocessDecodeTest(unittest.TestCase):
    def test_letterbox_and_coordinate_restore(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        prepared = preprocess_bgr(
            image,
            target_height=200,
            target_width=200,
            input_format="rgb_u8",
        )
        self.assertEqual(1.0, prepared.meta.scale)
        self.assertEqual(50, prepared.meta.pad_top)
        raw = prediction((100, 100, 80, 60, 1, 0.9))
        decoded = YoloDecoder(CLASSES, confidence_threshold=0.2).decode(
            raw, prepared.meta
        )
        self.assertEqual(1, len(decoded))
        self.assertEqual("vehicle", decoded[0].class_name)
        self.assertEqual((60.0, 20.0, 140.0, 80.0), decoded[0].box)

    def test_nms_is_class_aware(self) -> None:
        meta = PreprocessMeta(1.0, 0, 0, 100, 100, 100, 100)
        raw = prediction(
            (50, 50, 40, 40, 0, 0.90),
            (50, 50, 40, 40, 0, 0.80),
            (50, 50, 40, 40, 4, 0.85),
        )
        decoded = YoloDecoder(
            CLASSES, confidence_threshold=0.2, iou_threshold=0.5
        ).decode(raw, meta)
        self.assertEqual(["person", "fire"], [item.class_name for item in decoded])


class TrackerPipelineTest(unittest.TestCase):
    def test_tracker_never_crosses_classes(self) -> None:
        tracker = ClassAwareTracker(max_age_seconds=5.0)
        first = tracker.update(
            [
                Detection(0, "person", 0.9, (10, 10, 30, 40)),
                Detection(4, "fire", 0.9, (10, 10, 30, 40)),
            ],
            now=0.0,
        )
        second = tracker.update(
            [
                Detection(0, "person", 0.9, (11, 10, 31, 40)),
                Detection(4, "fire", 0.9, (11, 10, 31, 40)),
            ],
            now=0.1,
        )
        self.assertNotEqual(first[0].track_id, first[1].track_id)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first[1].track_id, second[1].track_id)

    def test_pipeline_calls_resident_backend_once_per_sequence(self) -> None:
        backend = FakeBackend(
            outputs=[prediction((50, 50, 20, 30, 0, 0.9))]
        )
        pipeline = UnifiedInference(
            backend,
            target_height=100,
            target_width=100,
            decoder=YoloDecoder(CLASSES, confidence_threshold=0.2),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        first = pipeline.infer(image, sequence=7, now=0.0)
        second = pipeline.infer(image, sequence=7, now=0.1)
        self.assertEqual(1, backend.call_count)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual("person", first.detections[0].class_name)
        self.assertEqual(first.detections[0].track_id, second.detections[0].track_id)


if __name__ == "__main__":
    unittest.main()
