from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable
from uuid import uuid4

import numpy as np

from ..contracts import AlarmCandidate, DepthEstimate, Detection
from .base import EngineResult, ResultOrToken, result_token
from .geometry import box_center_distance, box_iou


FLAME_EVENT_NAME = "明火报警"


@dataclass
class _FlameIncident:
    incident_id: str
    scope_id: str
    track_id: int | None
    candidate_started_at: float
    first_seen_at: float
    last_seen_at: float
    last_update_at: float
    last_box: tuple[float, float, float, float]
    best_detection: Detection
    last_alarm_at: float | None = None
    alarm_count: int = 0
    pending_token: str | None = None
    active_this_frame: bool = False


@dataclass(frozen=True)
class _PendingFlame:
    token: str
    incident_id: str
    reserved_at: float


class FlameEngine:
    """Strict visual-fire confirmation; depth only enriches the evidence."""

    engine_name = "flame"

    def __init__(
        self,
        *,
        event_name: str = FLAME_EVENT_NAME,
        fire_class: str = "fire",
        min_confidence: float = 0.45,
        min_width_px: float = 8.0,
        min_height_px: float = 8.0,
        min_area_px: float = 100.0,
        confirmation_seconds: float = 2.0,
        repeat_alarm_seconds: float = 600.0,
        clear_absence_seconds: float = 600.0,
        lost_grace_seconds: float = 0.5,
        association_iou_threshold: float = 0.15,
        association_center_distance_px: float = 72.0,
    ) -> None:
        self.event_name = str(event_name)
        self.fire_class = str(fire_class)
        self.min_confidence = float(min_confidence)
        self.min_width_px = max(0.0, float(min_width_px))
        self.min_height_px = max(0.0, float(min_height_px))
        self.min_area_px = max(0.0, float(min_area_px))
        self.confirmation_seconds = max(0.0, float(confirmation_seconds))
        self.repeat_alarm_seconds = max(0.0, float(repeat_alarm_seconds))
        self.clear_absence_seconds = max(0.0, float(clear_absence_seconds))
        self.lost_grace_seconds = max(0.0, float(lost_grace_seconds))
        self.association_iou_threshold = min(
            1.0, max(0.0, float(association_iou_threshold))
        )
        self.association_center_distance_px = max(
            0.0, float(association_center_distance_px)
        )
        self._incidents: dict[str, _FlameIncident] = {}
        self._pending: dict[str, _PendingFlame] = {}

    def process(
        self,
        detections: Iterable[Detection],
        frame_bgr: np.ndarray | None = None,
        *,
        now: float | None = None,
        occurred_at: float | None = None,
        depth_estimate: DepthEstimate | None = None,
        scope_id: str = "full_frame",
        frame_valid: bool = True,
    ) -> list[EngineResult]:
        current = time.monotonic() if now is None else float(now)
        if not frame_valid:
            self._pause(current)
            return []
        wall_epoch = time.time() if occurred_at is None else float(occurred_at)
        scope = str(scope_id or "full_frame")
        previously_active = {
            incident.incident_id
            for incident in self._incidents.values()
            if incident.active_this_frame
        }
        for incident in self._incidents.values():
            incident.active_this_frame = False
            incident.last_update_at = current
        eligible = [
            detection for detection in detections if self._eligible(detection)
        ]
        matched: set[str] = set()
        active: list[_FlameIncident] = []
        for detection in eligible:
            incident = self._best_match(detection, scope, matched)
            if incident is None:
                incident = self._new_incident(detection, scope, current)
            else:
                self._observe(
                    incident,
                    detection,
                    current,
                    continuous=incident.incident_id in previously_active,
                )
            matched.add(incident.incident_id)
            active.append(incident)

        self._prune(current)
        results: list[EngineResult] = []
        for incident in active:
            result = self._maybe_reserve(
                incident,
                current,
                wall_epoch,
                frame_bgr,
                depth_estimate,
            )
            if result is not None:
                results.append(result)
        return results

    evaluate = process

    def commit(self, result_or_token: ResultOrToken) -> bool:
        token = result_token(result_or_token)
        pending = self._pending.pop(token, None)
        if pending is None:
            return False
        incident = self._incidents.get(pending.incident_id)
        if incident is None or incident.pending_token != token:
            return False
        incident.pending_token = None
        incident.last_alarm_at = pending.reserved_at
        incident.alarm_count += 1
        return True

    def release(self, result_or_token: ResultOrToken) -> bool:
        token = result_token(result_or_token)
        pending = self._pending.pop(token, None)
        if pending is None:
            return False
        incident = self._incidents.get(pending.incident_id)
        if incident is None or incident.pending_token != token:
            return False
        incident.pending_token = None
        return True

    rollback = release

    def reset(self) -> None:
        self._incidents.clear()
        self._pending.clear()

    def state_rows(self, *, now: float | None = None) -> list[dict[str, object]]:
        current = time.monotonic() if now is None else float(now)
        rows: list[dict[str, object]] = []
        for incident in sorted(
            self._incidents.values(), key=lambda item: item.incident_id
        ):
            if incident.pending_token:
                phase = "PENDING_PERSIST"
            elif incident.alarm_count:
                phase = "ALARMED"
            else:
                phase = "CONFIRMING"
            rows.append(
                {
                    "incident_id": incident.incident_id,
                    "scope_id": incident.scope_id,
                    "phase": phase,
                    "active": incident.active_this_frame,
                    "duration_seconds": round(
                        max(0.0, current - incident.candidate_started_at), 3
                    ),
                    "alarm_count": incident.alarm_count,
                    "confidence": round(
                        float(incident.best_detection.confidence), 4
                    ),
                    "box_xyxy": [
                        round(float(value), 2) for value in incident.last_box
                    ],
                }
            )
        return rows

    def _eligible(self, detection: Detection) -> bool:
        if detection.class_name != self.fire_class:
            return False
        if float(detection.confidence) < self.min_confidence:
            return False
        x1, y1, x2, y2 = detection.box
        width = max(0.0, float(x2) - float(x1))
        height = max(0.0, float(y2) - float(y1))
        return (
            width >= self.min_width_px
            and height >= self.min_height_px
            and width * height >= self.min_area_px
        )

    def _new_incident(
        self,
        detection: Detection,
        scope_id: str,
        now: float,
    ) -> _FlameIncident:
        incident_id = f"flame_{uuid4().hex[:12]}"
        incident = _FlameIncident(
            incident_id=incident_id,
            scope_id=scope_id,
            track_id=detection.track_id,
            candidate_started_at=now,
            first_seen_at=now,
            last_seen_at=now,
            last_update_at=now,
            last_box=detection.box,
            best_detection=detection,
            active_this_frame=True,
        )
        self._incidents[incident_id] = incident
        return incident

    def _observe(
        self,
        incident: _FlameIncident,
        detection: Detection,
        now: float,
        *,
        continuous: bool,
    ) -> None:
        if (
            incident.alarm_count == 0
            and not continuous
            and now - incident.last_seen_at > self.lost_grace_seconds
        ):
            incident.candidate_started_at = now
            incident.first_seen_at = now
        incident.last_seen_at = now
        incident.last_update_at = now
        incident.last_box = detection.box
        incident.track_id = detection.track_id
        if detection.confidence >= incident.best_detection.confidence:
            incident.best_detection = detection
        else:
            incident.best_detection = Detection(
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=incident.best_detection.confidence,
                box=detection.box,
                track_id=detection.track_id,
            )
        incident.active_this_frame = True

    def _best_match(
        self,
        detection: Detection,
        scope_id: str,
        matched: set[str],
    ) -> _FlameIncident | None:
        if detection.track_id is not None:
            for incident in self._incidents.values():
                if (
                    incident.incident_id not in matched
                    and incident.scope_id == scope_id
                    and incident.track_id == detection.track_id
                ):
                    return incident
        candidates: list[tuple[float, float, str, _FlameIncident]] = []
        for incident in self._incidents.values():
            if incident.incident_id in matched or incident.scope_id != scope_id:
                continue
            overlap = box_iou(detection.box, incident.last_box)
            distance = box_center_distance(detection.box, incident.last_box)
            if (
                overlap >= self.association_iou_threshold
                or distance <= self.association_center_distance_px
            ):
                candidates.append(
                    (-overlap, distance, incident.incident_id, incident)
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    def _maybe_reserve(
        self,
        incident: _FlameIncident,
        now: float,
        occurred_at: float,
        frame_bgr: np.ndarray | None,
        depth: DepthEstimate | None,
    ) -> EngineResult | None:
        if incident.pending_token is not None:
            return None
        duration = max(0.0, now - incident.candidate_started_at)
        if incident.alarm_count == 0:
            if duration < self.confirmation_seconds:
                return None
            reason = "initial"
        else:
            if (
                incident.last_alarm_at is None
                or now - incident.last_alarm_at < self.repeat_alarm_seconds
            ):
                return None
            reason = "repeat"
        token = uuid4().hex
        incident.pending_token = token
        self._pending[token] = _PendingFlame(token, incident.incident_id, now)
        depth_value = (
            depth
            if depth is not None
            else DepthEstimate.unavailable("not_computed", computed_at=now)
        )
        evidence = {
            "incident_id": incident.incident_id,
            "scope_id": incident.scope_id,
            "duration_seconds": round(duration, 3),
            "reason": reason,
            "alarm_number": incident.alarm_count + 1,
            "detection": incident.best_detection.to_dict(),
            "distance_m": (
                round(float(depth_value.distance_m), 3)
                if depth_value.valid and depth_value.distance_m is not None
                else None
            ),
            "distance_text": (
                f"{float(depth_value.distance_m):.2f} m"
                if depth_value.valid and depth_value.distance_m is not None
                else "不可用"
            ),
            "depth_valid": bool(depth_value.valid),
            "depth_reason": depth_value.reason,
            "depth_valid_pixels": int(depth_value.valid_pixels),
        }
        return EngineResult(
            engine_name=self.engine_name,
            reservation_token=token,
            candidate=AlarmCandidate(
                event_name=self.event_name,
                occurred_at=float(occurred_at),
                frame_bgr=_snapshot(frame_bgr),
                point_name=None,
                evidence=evidence,
            ),
            detections=(incident.best_detection,),
            diagnostics={
                "incident_id": incident.incident_id,
                "distance_does_not_gate_alarm": True,
            },
        )

    def _prune(self, now: float) -> None:
        for incident_id, incident in list(self._incidents.items()):
            if incident.pending_token is not None:
                continue
            absence = max(0.0, now - incident.last_seen_at)
            if incident.alarm_count == 0 and absence > self.lost_grace_seconds:
                del self._incidents[incident_id]
            elif incident.alarm_count > 0 and absence >= self.clear_absence_seconds:
                del self._incidents[incident_id]

    def _pause(self, now: float) -> None:
        for incident in self._incidents.values():
            pause = max(0.0, now - incident.last_update_at)
            incident.candidate_started_at += pause
            if incident.last_alarm_at is not None:
                incident.last_alarm_at += pause
            incident.last_seen_at = now
            incident.last_update_at = now


def _snapshot(frame_bgr: np.ndarray | None) -> np.ndarray:
    if frame_bgr is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return np.asarray(frame_bgr).copy()


FlameDetectionEngine = FlameEngine
