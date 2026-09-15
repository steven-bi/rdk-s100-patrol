from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as clock_type, timedelta, timezone
import time
from typing import Iterable
from uuid import uuid4

import numpy as np

from ..contracts import AlarmCandidate, Detection
from .base import EngineResult, ResultOrToken, result_token


PERSON_EVENT_NAME = "夜间人员逗留"
CROWD_EVENT_NAME = "夜间人群聚集"
_BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


@dataclass
class _PersonTimer:
    key: int | str
    detection: Detection
    duration_seconds: float
    last_update: float
    missing_since: float | None = None


@dataclass
class _NightState:
    visit_active: bool = False
    visit_index: int = 0
    alarmed_in_visit: bool = False
    pending_token: str | None = None
    absent_since: float | None = None
    people: dict[int | str, _PersonTimer] = field(default_factory=dict)
    crowd_active: bool = False
    crowd_duration_seconds: float = 0.0
    crowd_last_update: float | None = None
    last_count: int = 0
    last_mode: str = "idle"


@dataclass(frozen=True)
class _PendingNight:
    token: str
    visit_index: int
    reserved_at: float


class NightPeopleEngine:
    """Full-frame night person/crowd rule with a shared 30-second rearm."""

    engine_name = "night_people"

    def __init__(
        self,
        *,
        person_event_name: str = PERSON_EVENT_NAME,
        crowd_event_name: str = CROWD_EVENT_NAME,
        person_class: str = "person",
        min_confidence: float = 0.45,
        active_start: str | clock_type = "22:00",
        active_end: str | clock_type = "06:00",
        person_dwell_seconds: float = 5.0,
        crowd_min_count: int = 3,
        crowd_dwell_seconds: float = 5.0,
        rearm_absence_seconds: float = 30.0,
        track_lost_grace_seconds: float = 0.75,
    ) -> None:
        self.person_event_name = str(person_event_name)
        self.crowd_event_name = str(crowd_event_name)
        self.person_class = str(person_class)
        self.min_confidence = float(min_confidence)
        self.active_start = _parse_clock(active_start)
        self.active_end = _parse_clock(active_end)
        self.person_dwell_seconds = max(0.0, float(person_dwell_seconds))
        self.crowd_min_count = max(2, int(crowd_min_count))
        self.crowd_dwell_seconds = max(0.0, float(crowd_dwell_seconds))
        self.rearm_absence_seconds = max(0.0, float(rearm_absence_seconds))
        self.track_lost_grace_seconds = max(0.0, float(track_lost_grace_seconds))
        self._state = _NightState()
        self._pending: dict[str, _PendingNight] = {}

    def process(
        self,
        detections: Iterable[Detection],
        frame_bgr: np.ndarray | None = None,
        *,
        now: float | None = None,
        occurred_at: float | None = None,
        clock_time: str | clock_type | datetime | None = None,
        frame_valid: bool = True,
    ) -> list[EngineResult]:
        current = time.monotonic() if now is None else float(now)
        if not frame_valid:
            self._pause(current)
            self._state.last_mode = "frame_paused"
            return []
        wall_epoch = time.time() if occurred_at is None else float(occurred_at)
        clock = _resolve_clock(clock_time, wall_epoch)
        eligible = [
            detection
            for detection in detections
            if detection.class_name == self.person_class
            and float(detection.confidence) >= self.min_confidence
        ]
        visible, deltas = self._sync_people(eligible, current)
        count = len(visible)
        self._state.last_count = count
        self._update_visit(count, current)

        if not self.time_active(clock):
            self._reset_candidates()
            self._state.last_mode = "inactive_time"
            return []
        if count >= self.crowd_min_count:
            # Crowd always wins.  Individual dwell is reset so it cannot emit
            # on the same frame or immediately after a crowd dips to two.
            for timer in self._state.people.values():
                timer.duration_seconds = 0.0
            if (
                self._state.crowd_active
                and self._state.crowd_last_update is not None
            ):
                self._state.crowd_duration_seconds += max(
                    0.0, current - self._state.crowd_last_update
                )
            else:
                self._state.crowd_active = True
                self._state.crowd_duration_seconds = 0.0
            self._state.crowd_last_update = current
            self._state.last_mode = "crowd_candidate"
        elif 1 <= count < self.crowd_min_count:
            self._reset_crowd()
            for key, timer in visible.items():
                timer.duration_seconds += deltas.get(key, 0.0)
            self._state.last_mode = "person_candidate"
        else:
            self._reset_crowd()
            self._state.last_mode = "idle"

        if (
            self._state.alarmed_in_visit
            or self._state.pending_token is not None
            or not self._state.visit_active
        ):
            if self._state.alarmed_in_visit:
                self._state.last_mode = "alarmed"
            return []

        if (
            count >= self.crowd_min_count
            and self._state.crowd_duration_seconds >= self.crowd_dwell_seconds
        ):
            return [
                self._reserve(
                    event_name=self.crowd_event_name,
                    event_code="night_crowd_gathering",
                    detections=tuple(timer.detection for timer in visible.values()),
                    count=count,
                    duration=self._state.crowd_duration_seconds,
                    frame_bgr=frame_bgr,
                    now=current,
                    occurred_at=wall_epoch,
                )
            ]

        qualifying = [
            timer
            for timer in visible.values()
            if timer.duration_seconds >= self.person_dwell_seconds
        ]
        if 1 <= count < self.crowd_min_count and qualifying:
            longest = max(qualifying, key=lambda timer: timer.duration_seconds)
            return [
                self._reserve(
                    event_name=self.person_event_name,
                    event_code="night_person_loitering",
                    detections=tuple(timer.detection for timer in visible.values()),
                    count=count,
                    duration=longest.duration_seconds,
                    frame_bgr=frame_bgr,
                    now=current,
                    occurred_at=wall_epoch,
                )
            ]
        return []

    evaluate = process

    def commit(self, result_or_token: ResultOrToken) -> bool:
        token = result_token(result_or_token)
        pending = self._pending.pop(token, None)
        if (
            pending is None
            or self._state.pending_token != token
            or self._state.visit_index != pending.visit_index
            or not self._state.visit_active
        ):
            return False
        self._state.pending_token = None
        self._state.alarmed_in_visit = True
        self._state.last_mode = "alarmed"
        return True

    def release(self, result_or_token: ResultOrToken) -> bool:
        token = result_token(result_or_token)
        pending = self._pending.pop(token, None)
        if (
            pending is None
            or self._state.pending_token != token
            or self._state.visit_index != pending.visit_index
        ):
            return False
        self._state.pending_token = None
        self._state.last_mode = "retry_ready"
        return True

    rollback = release

    def time_active(self, value: clock_type) -> bool:
        if self.active_start == self.active_end:
            return True
        if self.active_start < self.active_end:
            return self.active_start <= value < self.active_end
        return value >= self.active_start or value < self.active_end

    def reset(self) -> None:
        self._state = _NightState()
        self._pending.clear()

    def state_rows(self) -> list[dict[str, object]]:
        return [
            {
                "visit_active": self._state.visit_active,
                "visit_index": self._state.visit_index,
                "alarmed_in_visit": self._state.alarmed_in_visit,
                "pending": self._state.pending_token is not None,
                "person_count": self._state.last_count,
                "mode": self._state.last_mode,
                "crowd_duration_seconds": round(
                    self._state.crowd_duration_seconds, 3
                ),
                "person_timers": [
                    {
                        "track_id": timer.key,
                        "duration_seconds": round(timer.duration_seconds, 3),
                        "missing": timer.missing_since is not None,
                    }
                    for timer in self._state.people.values()
                ],
            }
        ]

    def _sync_people(
        self,
        detections: list[Detection],
        now: float,
    ) -> tuple[dict[int | str, _PersonTimer], dict[int | str, float]]:
        current_by_key: dict[int | str, Detection] = {}
        for detection in detections:
            key = self._track_key(detection)
            previous = current_by_key.get(key)
            if previous is None or detection.confidence > previous.confidence:
                current_by_key[key] = detection
        visible: dict[int | str, _PersonTimer] = {}
        deltas: dict[int | str, float] = {}
        for key, detection in current_by_key.items():
            timer = self._state.people.get(key)
            if timer is None:
                timer = _PersonTimer(
                    key=key,
                    detection=detection,
                    duration_seconds=0.0,
                    last_update=now,
                )
                self._state.people[key] = timer
                deltas[key] = 0.0
            else:
                if (
                    timer.missing_since is not None
                    and now - timer.missing_since > self.track_lost_grace_seconds
                ):
                    timer.duration_seconds = 0.0
                    delta = 0.0
                else:
                    delta = max(0.0, now - timer.last_update)
                timer.detection = detection
                timer.last_update = now
                timer.missing_since = None
                deltas[key] = delta
            visible[key] = timer
        for key, timer in list(self._state.people.items()):
            if key in visible:
                continue
            if timer.missing_since is None:
                timer.missing_since = now
            timer.last_update = now
            if now - timer.missing_since > self.track_lost_grace_seconds:
                del self._state.people[key]
        return visible, deltas

    def _update_visit(self, count: int, now: float) -> None:
        if count > 0:
            self._state.absent_since = None
            if not self._state.visit_active:
                self._state.visit_active = True
                self._state.visit_index += 1
                self._state.alarmed_in_visit = False
                self._state.pending_token = None
            return
        if not self._state.visit_active:
            return
        if self._state.absent_since is None:
            self._state.absent_since = now
        if now - self._state.absent_since >= self.rearm_absence_seconds:
            self._finish_visit()

    def _finish_visit(self) -> None:
        if self._state.pending_token is not None:
            self._pending.pop(self._state.pending_token, None)
        self._state.visit_active = False
        self._state.alarmed_in_visit = False
        self._state.pending_token = None
        self._state.absent_since = None
        self._state.people.clear()
        self._reset_candidates()
        self._state.last_mode = "rearmed"

    def _reserve(
        self,
        *,
        event_name: str,
        event_code: str,
        detections: tuple[Detection, ...],
        count: int,
        duration: float,
        frame_bgr: np.ndarray | None,
        now: float,
        occurred_at: float,
    ) -> EngineResult:
        token = uuid4().hex
        self._state.pending_token = token
        self._pending[token] = _PendingNight(token, self._state.visit_index, now)
        evidence = {
            "event_code": event_code,
            "person_count": int(count),
            "duration_seconds": round(float(duration), 3),
            "full_frame": True,
            "detections": [detection.to_dict() for detection in detections],
            "visit_index": self._state.visit_index,
        }
        self._state.last_mode = "pending_persist"
        return EngineResult(
            engine_name=self.engine_name,
            reservation_token=token,
            candidate=AlarmCandidate(
                event_name=event_name,
                occurred_at=float(occurred_at),
                frame_bgr=_snapshot(frame_bgr),
                point_name=None,
                evidence=evidence,
            ),
            detections=detections,
            diagnostics={"visit_index": self._state.visit_index},
        )

    def _pause(self, now: float) -> None:
        for timer in self._state.people.values():
            timer.last_update = now
            if timer.missing_since is not None:
                timer.missing_since = now
        if self._state.crowd_active:
            self._state.crowd_last_update = now
        if self._state.absent_since is not None:
            self._state.absent_since = now

    def _reset_candidates(self) -> None:
        for timer in self._state.people.values():
            timer.duration_seconds = 0.0
            timer.missing_since = None
        self._reset_crowd()

    def _reset_crowd(self) -> None:
        self._state.crowd_active = False
        self._state.crowd_duration_seconds = 0.0
        self._state.crowd_last_update = None

    @staticmethod
    def _track_key(detection: Detection) -> int | str:
        if detection.track_id is not None:
            return int(detection.track_id)
        x1, y1, x2, y2 = detection.box
        return (
            f"untracked:{round((x1 + x2) / 20.0)}:"
            f"{round((y1 + y2) / 20.0)}"
        )


def _parse_clock(value: str | clock_type) -> clock_type:
    if isinstance(value, clock_type):
        return value.replace(tzinfo=None)
    pieces = str(value).strip().split(":")
    if len(pieces) not in {2, 3}:
        raise ValueError(f"invalid clock value: {value!r}")
    return clock_type(
        int(pieces[0]),
        int(pieces[1]),
        int(pieces[2]) if len(pieces) == 3 else 0,
    )


def _resolve_clock(
    value: str | clock_type | datetime | None,
    wall_epoch: float,
) -> clock_type:
    if value is None:
        return datetime.fromtimestamp(wall_epoch, tz=_BEIJING_TZ).time().replace(
            tzinfo=None
        )
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(_BEIJING_TZ)
        return value.time().replace(tzinfo=None)
    return _parse_clock(value)


def _snapshot(frame_bgr: np.ndarray | None) -> np.ndarray:
    if frame_bgr is None:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return np.asarray(frame_bgr).copy()
