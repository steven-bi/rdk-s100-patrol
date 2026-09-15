from __future__ import annotations

import io
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from rdk_patrol.atomic_io import atomic_write_bytes, atomic_write_json, atomic_write_text
from rdk_patrol.contracts import AlarmCandidate
from rdk_patrol.time_utils import beijing_datetime, format_beijing_time

from .overlay import PillowEvidenceRenderer


ALARM_EVENT_NAMES = frozenset(
    {
        "车辆违停",
        "垃圾桶已满",
        "夜间人员逗留",
        "夜间人群聚集",
        "明火报警",
    }
)
POINT_EVENT_NAMES = frozenset({"车辆违停", "垃圾桶已满"})
DEFAULT_COOLDOWN_SECONDS = {
    "车辆违停": 300.0,
    "垃圾桶已满": 300.0,
    "明火报警": 600.0,
    "夜间人员逗留": 0.0,
    "夜间人群聚集": 0.0,
}


def _strict_public_record(
    keyframe_image: str,
    event_name: str,
    beijing_time: str,
    point_name: Optional[str],
) -> "OrderedDict[str, str]":
    # Insertion order is an external contract used by JSONL, latest.json,
    # sidecars, the LAN API, and the offline handover file.
    record: "OrderedDict[str, str]" = OrderedDict()
    record["keyframe_image"] = str(keyframe_image)
    record["event_name"] = str(event_name)
    record["beijing_time"] = str(beijing_time)
    if point_name is not None:
        record["point_name"] = str(point_name)
    return record


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return cleaned[:40] or "alarm"


class AlarmRepository:
    """Store alarm keyframes and records with atomic per-file replacement."""

    def __init__(
        self,
        root: Union[str, Path],
        renderer: Optional[PillowEvidenceRenderer] = None,
        jpeg_quality: int = 94,
    ) -> None:
        self.root = Path(root).resolve()
        self.images_dir = self.root / "images"
        self.sidecars_dir = self.root / "sidecars"
        self.records_path = self.root / "records.jsonl"
        self.latest_path = self.root / "latest.json"
        self.renderer = renderer or PillowEvidenceRenderer()
        self.jpeg_quality = min(100, max(70, int(jpeg_quality)))
        self._lock = threading.RLock()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sidecars_dir.mkdir(parents=True, exist_ok=True)

    def save(self, candidate: AlarmCandidate) -> "OrderedDict[str, str]":
        self._validate_candidate(candidate)
        beijing_time = format_beijing_time(candidate.occurred_at)
        date_dir = beijing_datetime(candidate.occurred_at).strftime("%Y%m%d")
        token = "{}_{}_{}".format(
            beijing_datetime(candidate.occurred_at).strftime("%H%M%S_%f"),
            _safe_component(candidate.event_name),
            uuid.uuid4().hex[:10],
        )
        relative_image = (Path("images") / date_dir / (token + ".jpg")).as_posix()
        image_path = self.root / Path(relative_image)
        relative_sidecar = Path("sidecars") / date_dir / (token + ".json")
        sidecar_path = self.root / relative_sidecar
        public = _strict_public_record(
            relative_image,
            candidate.event_name,
            beijing_time,
            candidate.point_name if candidate.event_name in POINT_EVENT_NAMES else None,
        )

        rendered = self.renderer.render(candidate, beijing_time)
        buffer = io.BytesIO()
        rendered.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        image_payload = buffer.getvalue()

        with self._lock:
            existing = b""
            if self.records_path.is_file():
                existing = self.records_path.read_bytes()
                if existing and not existing.endswith(b"\n"):
                    existing += b"\n"
            row = json.dumps(public, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            # Each artifact is committed via temp-file + fsync + os.replace.
            # Image and sidecar are written before indexes, so a reader never
            # sees a record whose image has not reached durable storage.
            atomic_write_bytes(image_path, image_payload)
            atomic_write_json(sidecar_path, public)
            atomic_write_bytes(self.records_path, existing + row + b"\n")
            atomic_write_json(self.latest_path, public)
        return public

    @staticmethod
    def _validate_candidate(candidate: AlarmCandidate) -> None:
        if candidate.event_name not in ALARM_EVENT_NAMES:
            raise ValueError("unsupported alarm event_name: {}".format(candidate.event_name))
        if candidate.event_name in POINT_EVENT_NAMES and not candidate.point_name:
            raise ValueError("{} requires point_name".format(candidate.event_name))
        if candidate.event_name not in POINT_EVENT_NAMES and candidate.point_name is not None:
            raise ValueError("{} must not contain point_name".format(candidate.event_name))

    def latest(self) -> Optional["OrderedDict[str, str]"]:
        with self._lock:
            if not self.latest_path.is_file():
                return None
            try:
                value = json.loads(
                    self.latest_path.read_text(encoding="utf-8"),
                    object_pairs_hook=OrderedDict,
                )
            except (OSError, ValueError):
                return None
        return self._validate_loaded_record(value)

    def list_records(
        self,
        limit: Optional[int] = None,
        event_name: Optional[str] = None,
        point_name: Optional[str] = None,
        newest_first: bool = True,
    ) -> List["OrderedDict[str, str]"]:
        records: List["OrderedDict[str, str]"] = []
        with self._lock:
            if self.records_path.is_file():
                try:
                    lines = self.records_path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    lines = []
            else:
                lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=OrderedDict)
            except ValueError:
                continue
            record = self._validate_loaded_record(value)
            if record is None:
                continue
            if event_name is not None and record["event_name"] != event_name:
                continue
            if point_name is not None and record.get("point_name") != point_name:
                continue
            records.append(record)
        if newest_first:
            records.reverse()
        if limit is not None:
            records = records[: max(0, int(limit))]
        return records

    @staticmethod
    def _validate_loaded_record(value: Any) -> Optional["OrderedDict[str, str]"]:
        if not isinstance(value, Mapping):
            return None
        event_name = value.get("event_name")
        point_name = value.get("point_name")
        expected = ["keyframe_image", "event_name", "beijing_time"]
        if point_name is not None:
            expected.append("point_name")
        if list(value.keys()) != expected:
            return None
        if event_name not in ALARM_EVENT_NAMES:
            return None
        if event_name in POINT_EVENT_NAMES and not point_name:
            return None
        if event_name not in POINT_EVENT_NAMES and point_name is not None:
            return None
        return _strict_public_record(
            str(value["keyframe_image"]),
            str(event_name),
            str(value["beijing_time"]),
            None if point_name is None else str(point_name),
        )

    def resolve_image(self, relative_path: str) -> Optional[Path]:
        try:
            candidate = (self.root / Path(relative_path)).resolve()
            candidate.relative_to(self.images_dir.resolve())
        except (OSError, ValueError):
            return None
        if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            return None
        return candidate if candidate.is_file() else None

    def records_bytes(self) -> bytes:
        with self._lock:
            if not self.records_path.is_file():
                return b""
            return self.records_path.read_bytes()


class AlarmPublisher:
    """Publish through persistent cooldown state.

    State is updated only after AlarmRepository.save succeeds. Callers can
    therefore treat a truthy return as both "cooldown accepted" and "alarm is
    durable", which is important for acknowledging persistent review jobs.
    """

    def __init__(
        self,
        repository: AlarmRepository,
        state_path: Optional[Union[str, Path]] = None,
        cooldown_seconds: Optional[Mapping[str, float]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.repository = repository
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else repository.root / "cooldown_state.json"
        )
        self.cooldown_seconds: Dict[str, float] = dict(DEFAULT_COOLDOWN_SECONDS)
        if cooldown_seconds:
            for key, value in cooldown_seconds.items():
                self.cooldown_seconds[str(key)] = max(0.0, float(value))
        self.clock = clock
        self._lock = threading.RLock()

    def try_publish(
        self,
        candidate: AlarmCandidate,
    ) -> Optional["OrderedDict[str, str]"]:
        AlarmRepository._validate_candidate(candidate)
        key = self._key(candidate)
        now = float(self.clock())
        with self._lock:
            state = self._load_state()
            last = state.get(key)
            cooldown = self.cooldown_seconds.get(candidate.event_name, 0.0)
            if last is not None and now - float(last) < cooldown:
                return None
            record = self.repository.save(candidate)
            state[key] = now
            atomic_write_json(self.state_path, state)
            return record

    def remaining_seconds(self, candidate: AlarmCandidate) -> float:
        key = self._key(candidate)
        now = float(self.clock())
        with self._lock:
            state = self._load_state()
        last = state.get(key)
        if last is None:
            return 0.0
        cooldown = self.cooldown_seconds.get(candidate.event_name, 0.0)
        return max(0.0, cooldown - (now - float(last)))

    @staticmethod
    def _key(candidate: AlarmCandidate) -> str:
        point = candidate.point_name if candidate.event_name in POINT_EVENT_NAMES else ""
        return "{}\u001f{}".format(candidate.event_name, point or "")

    def _load_state(self) -> Dict[str, float]:
        if not self.state_path.is_file():
            return {}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, Mapping):
            return {}
        result: Dict[str, float] = {}
        for key, value in raw.items():
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return result
