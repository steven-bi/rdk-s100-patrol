from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


Box = tuple[float, float, float, float]
Point2D = tuple[float, float]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: Box
    track_id: int | None = None

    @property
    def bottom_center(self) -> Point2D:
        x1, _y1, x2, y2 = self.box
        return ((float(x1) + float(x2)) * 0.5, float(y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 6),
            "box_xyxy": [round(float(v), 3) for v in self.box],
            "track_id": self.track_id,
        }


@dataclass
class FrameEnvelope:
    sequence: int
    monotonic_ts: float
    source_ts: float | None
    combined_bgr: np.ndarray
    detection_bgr: np.ndarray
    auxiliary_bgr: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy_for_worker(self, *, include_combined: bool = False) -> "FrameEnvelope":
        return FrameEnvelope(
            sequence=int(self.sequence),
            monotonic_ts=float(self.monotonic_ts),
            source_ts=None if self.source_ts is None else float(self.source_ts),
            combined_bgr=(
                self.combined_bgr.copy()
                if include_combined
                else np.empty((0, 0, 3), dtype=np.uint8)
            ),
            detection_bgr=self.detection_bgr.copy(),
            auxiliary_bgr=(
                None if self.auxiliary_bgr is None else self.auxiliary_bgr.copy()
            ),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class PointContext:
    point_id: str
    point_name: str
    source: str
    confidence: float
    observed_at: float
    tag_id: int | None = None
    rois: Mapping[str, Sequence[Point2D]] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.point_id and self.point_name and self.source)


@dataclass(frozen=True)
class DepthEstimate:
    valid: bool
    distance_m: float | None
    reason: str
    valid_pixels: int = 0
    computed_at: float | None = None
    quality: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(cls, reason: str, *, computed_at: float | None = None) -> "DepthEstimate":
        return cls(
            valid=False,
            distance_m=None,
            reason=str(reason),
            computed_at=computed_at,
        )


@dataclass
class AlarmCandidate:
    event_name: str
    occurred_at: float
    frame_bgr: np.ndarray
    point_name: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NavigationEvent:
    action: str
    occurred_at: float
    point_id: str | None = None
    point_name: str | None = None
    hold_seconds: float = 5.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    VALID_ACTIONS = frozenset({"arrive", "pause", "resume", "leave"})

    def __post_init__(self) -> None:
        if self.action not in self.VALID_ACTIONS:
            raise ValueError(f"unsupported navigation action: {self.action}")
        if self.action == "arrive" and not self.point_id:
            raise ValueError("arrive requires point_id")

