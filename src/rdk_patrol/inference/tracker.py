from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

from ..contracts import Box, Detection


@dataclass
class _Track:
    track_id: int
    class_id: int
    class_name: str
    box: Box
    confidence: float
    first_seen: float
    last_seen: float
    missed: int = 0


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    class_name: str
    box: Box
    confidence: float
    first_seen: float
    last_seen: float
    missed: int


class ClassAwareTracker:
    """Greedy CPU tracker that never associates detections across classes."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.10,
        center_distance_ratio_threshold: float = 0.85,
        max_missed: int = 15,
        max_age_seconds: float = 2.0,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.center_distance_ratio_threshold = float(center_distance_ratio_threshold)
        self.max_missed = max(0, int(max_missed))
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self,
        detections: Iterable[Detection],
        *,
        now: float | None = None,
    ) -> list[Detection]:
        current = time.monotonic() if now is None else float(now)
        items = list(detections)
        unmatched = set(self._tracks)
        order = sorted(
            range(len(items)),
            key=lambda index: (-float(items[index].confidence), index),
        )
        for index in order:
            detection = items[index]
            best_id: int | None = None
            best_score = float("-inf")
            for track_id in sorted(unmatched):
                track = self._tracks[track_id]
                if (
                    track.class_id != int(detection.class_id)
                    or track.class_name != detection.class_name
                ):
                    continue
                overlap = box_iou(track.box, detection.box)
                distance_ratio = center_distance_ratio(track.box, detection.box)
                if (
                    overlap < self.iou_threshold
                    and distance_ratio > self.center_distance_ratio_threshold
                ):
                    continue
                score = overlap - min(distance_ratio, 10.0) * 0.05
                if score > best_score:
                    best_id = track_id
                    best_score = score
            if best_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    class_id=int(detection.class_id),
                    class_name=detection.class_name,
                    box=detection.box,
                    confidence=float(detection.confidence),
                    first_seen=current,
                    last_seen=current,
                )
                detection.track_id = track_id
                continue
            track = self._tracks[best_id]
            track.box = detection.box
            track.confidence = float(detection.confidence)
            track.last_seen = current
            track.missed = 0
            detection.track_id = best_id
            unmatched.remove(best_id)

        for track_id in list(unmatched):
            track = self._tracks[track_id]
            track.missed += 1
            if (
                track.missed > self.max_missed
                or current - track.last_seen > self.max_age_seconds
            ):
                del self._tracks[track_id]
        return items

    def snapshots(self) -> list[TrackSnapshot]:
        return [
            TrackSnapshot(
                track_id=track.track_id,
                class_name=track.class_name,
                box=track.box,
                confidence=track.confidence,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                missed=track.missed,
            )
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


def box_iou(first: Box, second: Box) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / max(1e-9, first_area + second_area - intersection)


def center_distance_ratio(first: Box, second: Box) -> float:
    first_center = (
        (float(first[0]) + float(first[2])) * 0.5,
        (float(first[1]) + float(first[3])) * 0.5,
    )
    second_center = (
        (float(second[0]) + float(second[2])) * 0.5,
        (float(second[1]) + float(second[3])) * 0.5,
    )
    distance = math.dist(first_center, second_center)
    first_diag = math.hypot(
        max(1.0, float(first[2]) - float(first[0])),
        max(1.0, float(first[3]) - float(first[1])),
    )
    second_diag = math.hypot(
        max(1.0, float(second[2]) - float(second[0])),
        max(1.0, float(second[3]) - float(second[1])),
    )
    return distance / max(1.0, first_diag, second_diag)
