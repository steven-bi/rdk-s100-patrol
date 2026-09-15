from __future__ import annotations

import numpy as np

from rdk_patrol.contracts import AlarmCandidate, Detection
from rdk_patrol.evidence import annotate_live_frame, enrich_alarm_candidate


def test_enrich_flame_distance_unavailable() -> None:
    detection = Detection(4, "fire", 0.91, (1, 2, 9, 10), 7)
    candidate = AlarmCandidate(
        event_name="明火报警",
        occurred_at=0.0,
        frame_bgr=np.zeros((12, 12, 3), dtype=np.uint8),
        evidence={"distance_m": None, "detection": detection.to_dict()},
    )
    enrich_alarm_candidate(candidate, [detection])
    assert len(candidate.evidence["boxes"]) == 1
    assert "距离：不可用" in candidate.evidence["overlay_lines"]


def test_live_annotation_ignores_smoke() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    smoke = Detection(5, "smoke", 0.99, (1, 1, 18, 18))
    assert np.array_equal(frame, annotate_live_frame(frame, [smoke]))
