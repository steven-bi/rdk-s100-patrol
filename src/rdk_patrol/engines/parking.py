from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence
from uuid import uuid4

import numpy as np

from ..contracts import AlarmCandidate, Detection, Point2D, PointContext
from .base import EngineResult, ResultOrToken, result_token
from .geometry import point_in_polygon


PARKING_EVENT_NAME = "车辆违停"


@dataclass
class _ParkingState:
    duration_seconds: float
    last_update: float
    last_seen: float
    detection: Detection
    seen_frames: int = 1
    active_previous_frame: bool = True


@dataclass(frozen=True)
class _PendingParking:
    token: str
    scope_key: str
    reserved_at: float


class ParkingEngine:
    """Point-bound parking dwell rule with transactional alarm emission."""

    engine_name = "parking"

    def __init__(
        self,
        *,
        event_name: str = PARKING_EVENT_NAME,
        vehicle_class: str = "vehicle",
        min_confidence: float = 0.20,
        dwell_seconds: float = 2.0,
        cooldown_seconds: float = 300.0,
        track_lost_grace_seconds: float = 0.75,
        roi_names: Sequence[str] | None = None,
    ) -> None:
        self.event_name = str(event_name)
        self.vehicle_class = str(vehicle_class)
        self.min_confidence = float(min_confidence)
        self.dwell_seconds = max(0.0, float(dwell_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.track_lost_grace_seconds = max(0.0, float(track_lost_grace_seconds))
        self.roi_names = (
            None if roi_names is None else frozenset(str(item) for item in roi_names)
        )
        self._states: dict[tuple[str, str, int | str], _ParkingState] = {}
        self._last_alarm: dict[str, float] = {}
        self._pending_by_token: dict[str, _PendingParking] = {}
        self._pending_by_scope: dict[str, str] = {}

    def process(
        self,
        detections: Iterable[Detection],
        point_context: PointContext | None = None,
        frame_bgr: np.ndarray | None = None,
        *,
        now: float | None = None,
        occurred_at: float | None = None,
        frame_valid: bool = True,
    ) -> list[EngineResult]:
        current = time.monotonic() if now is None else float(now)
        if not frame_valid:
            self._pause(current)
            return []
        rois = self._parking_rois(point_context)
        if point_context is None or not point_context.valid or not rois:
            # A parking timer must never survive an untrusted point/ROI context.
            self._states.clear()
            return []

        eligible = [
            detection
            for detection in detections
            if detection.class_name == self.vehicle_class
            and float(detection.confidence) >= self.min_confidence
        ]
        active_keys: set[tuple[str, str, int | str]] = set()
        results: list[EngineResult] = []
        for roi_id, polygon in rois:
            for detection in eligible:
                if not point_in_polygon(detection.bottom_center, polygon):
                    continue
                track_key = self._track_key(detection)
                key = (point_context.point_id, roi_id, track_key)
                active_keys.add(key)
                state = self._states.get(key)
                if state is None:
                    state = _ParkingState(
                        duration_seconds=0.0,
                        last_update=current,
                        last_seen=current,
                        detection=detection,
                    )
                    self._states[key] = state
                else:
                    if (
                        not state.active_previous_frame
                        and current - state.last_seen > self.track_lost_grace_seconds
                    ):
                        state.duration_seconds = 0.0
                        state.seen_frames = 0
                    state.duration_seconds += max(0.0, current - state.last_update)
                    state.last_update = current
                    state.last_seen = current
                    state.detection = detection
                    state.seen_frames += 1
                    state.active_previous_frame = True
                if state.duration_seconds < self.dwell_seconds:
                    continue
                # The formal deduplication policy is point-based: two polygons
                # at one physical patrol point must not create two alarms.
                scope_key = point_context.point_id
                if scope_key in self._pending_by_scope:
                    continue
                last_alarm = self._last_alarm.get(scope_key)
                if (
                    last_alarm is not None
                    and current - last_alarm < self.cooldown_seconds
                ):
                    continue
                token = uuid4().hex
                pending = _PendingParking(token, scope_key, current)
                self._pending_by_token[token] = pending
                self._pending_by_scope[scope_key] = token
                evidence = {
                    "point_id": point_context.point_id,
                    "point_source": point_context.source,
                    "point_confidence": float(point_context.confidence),
                    "tag_id": point_context.tag_id,
                    "roi_id": roi_id,
                    "roi_polygon": [
                        [float(point[0]), float(point[1])] for point in polygon
                    ],
                    "duration_seconds": round(state.duration_seconds, 3),
                    "detection": detection.to_dict(),
                }
                alarm = AlarmCandidate(
                    event_name=self.event_name,
                    occurred_at=(
                        time.time() if occurred_at is None else float(occurred_at)
                    ),
                    frame_bgr=_snapshot(frame_bgr),
                    point_name=point_context.point_name,
                    evidence=evidence,
                )
                results.append(
                    EngineResult(
                        engine_name=self.engine_name,
                        reservation_token=token,
                        candidate=alarm,
                        detections=(detection,),
                        diagnostics={"scope_key": scope_key},
                    )
                )
        self._mark_missing(current, active_keys)
        return results

    evaluate = process

    def commit(self, result_or_token: ResultOrToken) -> bool:
        token = result_token(result_or_token)
        pending = self._pending_by_token.pop(token, None)
        if pending is None or self._pending_by_scope.get(pending.scope_key) != token:
            return False
        self._pending_by_scope.pop(pending.scope_key, None)
        self._last_alarm[pending.scope_key] = pending.reserved_at
        return True

    def release(self, result_or_token: ResultOrToken) -> bool:
        token = result_token(result_or_token)
        pending = self._pending_by_token.pop(token, None)
        if pending is None or self._pending_by_scope.get(pending.scope_key) != token:
            return False
        self._pending_by_scope.pop(pending.scope_key, None)
        return True

    rollback = release

    def reset(self, *, preserve_cooldowns: bool = True) -> None:
        self._states.clear()
        self._pending_by_token.clear()
        self._pending_by_scope.clear()
        if not preserve_cooldowns:
            self._last_alarm.clear()

    def state_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (point_id, roi_id, track_id), state in sorted(
            self._states.items(), key=lambda item: str(item[0])
        ):
            scope = point_id
            rows.append(
                {
                    "point_id": point_id,
                    "roi_id": roi_id,
                    "track_id": track_id,
                    "duration_seconds": round(state.duration_seconds, 3),
                    "seen_frames": state.seen_frames,
                    "pending": scope in self._pending_by_scope,
                    "box_xyxy": [round(float(value), 2) for value in state.detection.box],
                }
            )
        return rows

    def _parking_rois(
        self,
        context: PointContext | None,
    ) -> list[tuple[str, list[Point2D]]]:
        if context is None:
            return []
        result: list[tuple[str, list[Point2D]]] = []
        for name, raw_polygon in context.rois.items():
            if not self._is_parking_roi(str(name)):
                continue
            polygon = [
                (float(point[0]), float(point[1]))
                for point in raw_polygon
                if len(point) >= 2
            ]
            if len(polygon) >= 3:
                result.append((str(name), polygon))
        return result

    def _is_parking_roi(self, name: str) -> bool:
        if self.roi_names is not None:
            return name in self.roi_names
        normalized = name.lower().replace("-", "_")
        return (
            normalized in {"parking", "vehicle_parking", "no_parking"}
            or normalized.startswith("parking_")
            or normalized.startswith("vehicle_parking_")
            or normalized.startswith("no_parking_")
        )

    def _pause(self, now: float) -> None:
        for state in self._states.values():
            state.last_update = now
            state.last_seen = now

    def _mark_missing(
        self,
        now: float,
        active_keys: set[tuple[str, str, int | str]],
    ) -> None:
        for key, state in list(self._states.items()):
            if key in active_keys:
                continue
            state.active_previous_frame = False
            state.last_update = now
            if now - state.last_seen > self.track_lost_grace_seconds:
                del self._states[key]

    @staticmethod
    def _track_key(detection: Detection) -> int | str:
        if detection.track_id is not None:
            return int(detection.track_id)
        x1, y1, x2, y2 = detection.box
        return (
            f"untracked:{round((x1 + x2) / 20.0)}:"
            f"{round((y1 + y2) / 20.0)}"
        )


def _snapshot(frame_bgr: np.ndarray | None) -> np.ndarray:
    if frame_bgr is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return np.asarray(frame_bgr).copy()


VehicleParkingEngine = ParkingEngine
