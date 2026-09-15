from __future__ import annotations

import numpy as np

from rdk_patrol.alarms import AlarmPublisher, AlarmRepository
from rdk_patrol.contracts import FrameEnvelope
from rdk_patrol.engines import FlameEngine, NightPeopleEngine, ParkingEngine
from rdk_patrol.inference.pipeline import InferenceResult
from rdk_patrol.inference.preprocess import PreprocessMeta
from rdk_patrol.processor import PatrolFrameProcessor


class _Inference:
    def __init__(self):
        self.calls = 0

    def infer_envelope(self, envelope, now=None):
        self.calls += 1
        return InferenceResult(
            detections=(),
            sequence=envelope.sequence,
            inference_ms=1.0,
            preprocessing_ms=0.5,
            decoding_ms=0.2,
            meta=PreprocessMeta(
                original_height=10,
                original_width=10,
                target_height=10,
                target_width=10,
                scale=1.0,
                pad_left=0,
                pad_top=0,
            ),
        )


def test_processor_calls_unified_inference_once(tmp_path) -> None:
    inference = _Inference()
    publisher = AlarmPublisher(AlarmRepository(tmp_path))
    processor = PatrolFrameProcessor(
        inference=inference,
        parking_engine=ParkingEngine(),
        night_engine=NightPeopleEngine(),
        flame_engine=FlameEngine(),
        alarm_publisher=publisher,
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        sequence=1,
        monotonic_ts=10.0,
        source_ts=None,
        combined_bgr=frame,
        detection_bgr=frame,
    )
    report = processor.process(envelope, occurred_at=100.0)
    assert inference.calls == 1
    assert report.sequence == 1
    assert report.errors == []
