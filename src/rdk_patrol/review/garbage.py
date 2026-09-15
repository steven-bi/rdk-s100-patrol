from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Union

import cv2
import numpy as np

from rdk_patrol.atomic_io import atomic_write_bytes, atomic_write_json
from rdk_patrol.contracts import AlarmCandidate


GARBAGE_REVIEW_WINDOW_SECONDS = 5.0
MINIMAX_GARBAGE_PROMPT = (
    "图片中垃圾桶满了吗？如果图片中有多个垃圾桶或者一个垃圾桶有多个桶口，"
    "只要其中一个垃圾桶或者其中一个桶口满了就判断为已满，只输出一个词：已满 或 未满。"
)
DECISION_FULL = "已满"
DECISION_NOT_FULL = "未满"


def parse_minimax_decision(value: Any) -> str:
    """Accept exactly one of the two required words and no surrounding prose."""

    text = str(value if value is not None else "").strip()
    if text in (DECISION_FULL, DECISION_NOT_FULL):
        return text
    raise ValueError("MiniMax reply must be exactly 已满 or 未满")


def score_frame_quality(frame_bgr: np.ndarray) -> float:
    """Score sharp, well-exposed full frames without cropping or resizing."""

    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty HxWx3 BGR image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    mean = float(gray.mean())
    dark_fraction = float(np.mean(gray < 8))
    bright_fraction = float(np.mean(gray > 247))
    exposure_penalty = abs(mean - 128.0) * 0.18
    clipping_penalty = (dark_fraction + bright_fraction) * 80.0
    return float(np.log1p(max(0.0, sharpness)) * 18.0 + contrast - exposure_penalty - clipping_penalty)


@dataclass
class _SelectedFrame:
    frame_bgr: np.ndarray
    quality_score: float
    captured_epoch: float


class BestFrameWindow:
    """Select one quality-best, whole patrol frame from a fixed five seconds."""

    def __init__(
        self,
        point_id: str,
        point_name: str,
        started_monotonic: float,
    ) -> None:
        if not point_id or not point_name:
            raise ValueError("point_id and point_name are required")
        self.point_id = str(point_id)
        self.point_name = str(point_name)
        self.started_monotonic = float(started_monotonic)
        self.deadline_monotonic = self.started_monotonic + GARBAGE_REVIEW_WINDOW_SECONDS
        self._selected: Optional[_SelectedFrame] = None

    def consider(
        self,
        frame_bgr: np.ndarray,
        captured_monotonic: float,
        captured_epoch: Optional[float] = None,
    ) -> bool:
        captured = float(captured_monotonic)
        if captured < self.started_monotonic or captured > self.deadline_monotonic:
            return False
        score = score_frame_quality(frame_bgr)
        if self._selected is None or score > self._selected.quality_score:
            # A whole-frame copy protects the queued keyframe from a camera
            # buffer being reused by the capture thread.
            self._selected = _SelectedFrame(
                frame_bgr=np.ascontiguousarray(frame_bgr).copy(),
                quality_score=score,
                captured_epoch=float(time.time() if captured_epoch is None else captured_epoch),
            )
            return True
        return False

    def is_complete(self, now_monotonic: float) -> bool:
        return float(now_monotonic) >= self.deadline_monotonic

    def enqueue(
        self,
        queue: "PersistentReviewQueue",
        now_monotonic: float,
    ) -> Optional["PendingReview"]:
        if not self.is_complete(now_monotonic):
            return None
        if self._selected is None:
            raise RuntimeError("review window completed without a valid frame")
        return queue.enqueue(
            frame_bgr=self._selected.frame_bgr,
            point_id=self.point_id,
            point_name=self.point_name,
            occurred_at=self._selected.captured_epoch,
            quality_score=self._selected.quality_score,
        )

    @property
    def selected_frame(self) -> Optional[np.ndarray]:
        if self._selected is None:
            return None
        return self._selected.frame_bgr.copy()

    @property
    def selected_quality(self) -> Optional[float]:
        return None if self._selected is None else self._selected.quality_score


class GarbageReviewCollector:
    """Manage at most one active five-second window per point."""

    def __init__(
        self,
        queue: "PersistentReviewQueue",
        monotonic_clock: Callable[[], float] = time.monotonic,
        epoch_clock: Callable[[], float] = time.time,
    ) -> None:
        self.queue = queue
        self.monotonic_clock = monotonic_clock
        self.epoch_clock = epoch_clock
        self._windows: Dict[str, BestFrameWindow] = {}
        self._lock = threading.RLock()

    def start(
        self,
        point_id: str,
        point_name: str,
        now_monotonic: Optional[float] = None,
    ) -> BestFrameWindow:
        now = self.monotonic_clock() if now_monotonic is None else float(now_monotonic)
        window = BestFrameWindow(point_id, point_name, now)
        with self._lock:
            self._windows[str(point_id)] = window
        return window

    def offer(
        self,
        point_id: str,
        full_frame_bgr: np.ndarray,
        captured_monotonic: Optional[float] = None,
        captured_epoch: Optional[float] = None,
    ) -> bool:
        monotonic_value = (
            self.monotonic_clock() if captured_monotonic is None else float(captured_monotonic)
        )
        epoch_value = self.epoch_clock() if captured_epoch is None else float(captured_epoch)
        with self._lock:
            window = self._windows.get(str(point_id))
            if window is None:
                return False
            return window.consider(full_frame_bgr, monotonic_value, epoch_value)

    def finalize(
        self,
        point_id: str,
        now_monotonic: Optional[float] = None,
    ) -> Optional["PendingReview"]:
        now = self.monotonic_clock() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            window = self._windows.get(str(point_id))
            if window is None or not window.is_complete(now):
                return None
            review = window.enqueue(self.queue, now)
            del self._windows[str(point_id)]
            return review

    def cancel(self, point_id: str) -> bool:
        """Discard an incomplete window when the verified point is lost."""

        with self._lock:
            return self._windows.pop(str(point_id), None) is not None


@dataclass(frozen=True)
class PendingReview:
    review_id: str
    point_id: str
    point_name: str
    image_path: str
    occurred_at: float
    quality_score: float
    attempts: int
    next_attempt_at: float
    last_error: str
    decision: str
    metadata_path: Path


class PersistentReviewQueue:
    """Filesystem-backed queue; pending work survives process and network loss."""

    def __init__(self, root: Union[str, Path], jpeg_quality: int = 95) -> None:
        self.root = Path(root).resolve()
        self.pending_dir = self.root / "pending"
        self.completed_dir = self.root / "completed"
        self.images_dir = self.root / "images"
        self.jpeg_quality = min(100, max(80, int(jpeg_quality)))
        self._lock = threading.RLock()
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        frame_bgr: np.ndarray,
        point_id: str,
        point_name: str,
        occurred_at: float,
        quality_score: float,
    ) -> PendingReview:
        if not point_id or not point_name:
            raise ValueError("point_id and point_name are required")
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty HxWx3 image")
        review_id = "{}_{}".format(int(float(occurred_at) * 1000), uuid.uuid4().hex[:12])
        image_path = self.images_dir / (review_id + ".jpg")
        metadata_path = self.pending_dir / (review_id + ".json")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise OSError("failed to encode pending review image")
        relative_image = image_path.relative_to(self.root).as_posix()
        payload: Dict[str, Any] = {
            "review_id": review_id,
            "status": "pending",
            "point_id": str(point_id),
            "point_name": str(point_name),
            "image_path": relative_image,
            "occurred_at": float(occurred_at),
            "quality_score": float(quality_score),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "attempts": 0,
            "next_attempt_at": 0.0,
            "last_error": "",
            "decision": "",
        }
        with self._lock:
            atomic_write_bytes(image_path, encoded.tobytes())
            atomic_write_json(metadata_path, payload)
        return self._from_payload(metadata_path, payload)

    def ready(
        self,
        now_epoch: Optional[float] = None,
        limit: int = 1,
    ) -> List[PendingReview]:
        now = time.time() if now_epoch is None else float(now_epoch)
        result: List[PendingReview] = []
        with self._lock:
            paths = sorted(self.pending_dir.glob("*.json"))
            for path in paths:
                payload = self._read_payload(path)
                if payload is None or payload.get("status") != "pending":
                    continue
                try:
                    next_attempt = float(payload.get("next_attempt_at", 0.0))
                except (TypeError, ValueError):
                    next_attempt = 0.0
                if next_attempt > now:
                    continue
                result.append(self._from_payload(path, payload))
                if len(result) >= max(0, int(limit)):
                    break
        return result

    def pending_count(self) -> int:
        return len(self.ready(now_epoch=float("inf"), limit=2**31 - 1))

    def read_frame(self, review: PendingReview) -> Optional[np.ndarray]:
        try:
            path = (self.root / Path(review.image_path)).resolve()
            path.relative_to(self.images_dir.resolve())
            payload = path.read_bytes()
        except (OSError, ValueError):
            return None
        array = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def retry(
        self,
        review: PendingReview,
        error: str,
        now_epoch: Optional[float] = None,
        delay_seconds: Optional[float] = None,
    ) -> PendingReview:
        now = time.time() if now_epoch is None else float(now_epoch)
        with self._lock:
            payload = self._read_payload(review.metadata_path)
            if payload is None:
                raise FileNotFoundError(str(review.metadata_path))
            attempts = int(payload.get("attempts", 0)) + 1
            delay = (
                min(300.0, max(2.0, float(2 ** min(attempts, 8))))
                if delay_seconds is None
                else max(0.1, float(delay_seconds))
            )
            payload.update(
                {
                    "attempts": attempts,
                    "next_attempt_at": now + delay,
                    "last_error": str(error)[:500],
                }
            )
            atomic_write_json(review.metadata_path, payload)
            return self._from_payload(review.metadata_path, payload)

    def record_decision(
        self,
        review: PendingReview,
        decision: str,
        decided_at: Optional[float] = None,
    ) -> PendingReview:
        """Persist a paid MiniMax result before attempting the cooldown gate."""

        strict_decision = parse_minimax_decision(decision)
        with self._lock:
            payload = self._read_payload(review.metadata_path)
            if payload is None:
                raise FileNotFoundError(str(review.metadata_path))
            payload.update(
                {
                    "decision": strict_decision,
                    "decision_confirmed_at": float(
                        time.time() if decided_at is None else decided_at
                    ),
                    "last_error": "",
                }
            )
            atomic_write_json(review.metadata_path, payload)
            return self._from_payload(review.metadata_path, payload)

    def acknowledge(
        self,
        review: PendingReview,
        final_status: str,
        decision: str,
        alarm_record: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if final_status not in {"not_full", "alarm_saved"}:
            raise ValueError("unsupported final_status")
        with self._lock:
            payload = self._read_payload(review.metadata_path)
            if payload is None:
                return
            payload.update(
                {
                    "status": final_status,
                    "decision": parse_minimax_decision(decision),
                    "completed_at": time.time(),
                    "alarm": None if alarm_record is None else dict(alarm_record),
                }
            )
            completed_path = self.completed_dir / review.metadata_path.name
            atomic_write_json(completed_path, payload)
            try:
                review.metadata_path.unlink()
            except FileNotFoundError:
                pass
            try:
                (self.root / Path(review.image_path)).unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _read_payload(path: Path) -> Optional[Dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _from_payload(path: Path, payload: Mapping[str, Any]) -> PendingReview:
        return PendingReview(
            review_id=str(payload.get("review_id", path.stem)),
            point_id=str(payload.get("point_id", "")),
            point_name=str(payload.get("point_name", "")),
            image_path=str(payload.get("image_path", "")),
            occurred_at=float(payload.get("occurred_at", 0.0)),
            quality_score=float(payload.get("quality_score", 0.0)),
            attempts=int(payload.get("attempts", 0)),
            next_attempt_at=float(payload.get("next_attempt_at", 0.0)),
            last_error=str(payload.get("last_error", "")),
            decision=str(payload.get("decision", "")),
            metadata_path=path,
        )


class MiniMaxVisionClient:
    """Minimal OpenAI-compatible MiniMax vision client.

    The endpoint/model may differ between deployments, but the decision prompt
    is the code constant above and intentionally has no constructor override.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_env: str = "MINIMAX_API_KEY",
        base_url: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M3",
        timeout_seconds: float = 5.0,
        jpeg_quality: int = 85,
        image_detail: str = "low",
        max_long_side_pixel: int = 960,
        service_tier: str = "priority",
        urlopen: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.api_key = api_key
        self.api_key_env = str(api_key_env)
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self.jpeg_quality = min(95, max(40, int(jpeg_quality)))
        self.image_detail = str(image_detail)
        if self.image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be low, high, or auto")
        self.max_long_side_pixel = max(1, int(max_long_side_pixel))
        self.service_tier = str(service_tier)
        self.urlopen = urlopen or urllib.request.urlopen

    @classmethod
    def from_system_config(
        cls,
        document: Mapping[str, Any],
        api_key: Optional[str] = None,
        urlopen: Optional[Callable[..., Any]] = None,
    ) -> "MiniMaxVisionClient":
        """Build from system.yaml/load_config output without a prompt override."""

        section_value = document.get("minimax", document)
        if not isinstance(section_value, Mapping):
            raise ValueError("minimax configuration must be a mapping")
        section = section_value
        return cls(
            api_key=api_key,
            api_key_env=str(section.get("api_key_env", "MINIMAX_API_KEY")),
            base_url=str(section.get("base_url", "https://api.minimaxi.com/v1")),
            model=str(section.get("model", "MiniMax-M3")),
            timeout_seconds=float(section.get("timeout_seconds", 5.0)),
            jpeg_quality=int(section.get("jpeg_quality", 85)),
            image_detail=str(section.get("image_detail", "low")),
            max_long_side_pixel=int(section.get("max_long_side_pixel", 960)),
            service_tier=str(section.get("service_tier", "priority")),
            urlopen=urlopen,
        )

    def classify(self, full_frame_bgr: np.ndarray) -> str:
        api_key = self.api_key or os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError("{} is empty".format(self.api_key_env))
        frame = np.asarray(full_frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("full frame must be a non-empty HxWx3 BGR image")
        upload_frame = self._resize_whole_frame(frame)
        ok, encoded = cv2.imencode(
            ".jpg",
            upload_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise OSError("failed to encode MiniMax upload frame")
        image_url = "data:image/jpeg;base64," + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": MINIMAX_GARBAGE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": self.image_detail,
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 8,
            "thinking": {"type": "disabled"},
            "service_tier": self.service_tier,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
            method="POST",
        )
        with self.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        reply = self._extract_reply(payload)
        return parse_minimax_decision(reply)

    def _resize_whole_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale the complete view proportionally; never crop an ROI."""

        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest <= self.max_long_side_pixel:
            return frame
        scale = float(self.max_long_side_pixel) / float(longest)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _extract_reply(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], Mapping) else {}
            message = first.get("message") if isinstance(first, Mapping) else {}
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, Mapping)
                    )
        output_text = payload.get("output_text")
        return "" if output_text is None else str(output_text)


class _Classifier(Protocol):
    def classify(self, full_frame_bgr: np.ndarray) -> str:
        ...


class _Publisher(Protocol):
    def try_publish(self, candidate: AlarmCandidate) -> Optional[Mapping[str, Any]]:
        ...

    def remaining_seconds(self, candidate: AlarmCandidate) -> float:
        ...


class GarbageReviewWorker:
    """Process one durable review, failing closed and retrying every anomaly."""

    def __init__(
        self,
        queue: PersistentReviewQueue,
        classifier: _Classifier,
        publisher: _Publisher,
        retry_delay_seconds: float = 10.0,
        epoch_clock: Callable[[], float] = time.time,
    ) -> None:
        self.queue = queue
        self.classifier = classifier
        self.publisher = publisher
        self.retry_delay_seconds = max(0.1, float(retry_delay_seconds))
        self.epoch_clock = epoch_clock

    def process_one(self, now_epoch: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = self.epoch_clock() if now_epoch is None else float(now_epoch)
        ready = self.queue.ready(now, limit=1)
        if not ready:
            return None
        review = ready[0]
        frame = self.queue.read_frame(review)
        if frame is None:
            self.queue.retry(
                review,
                "pending full-frame image is unreadable",
                now_epoch=now,
                delay_seconds=self.retry_delay_seconds,
            )
            return {"status": "retry", "review_id": review.review_id}
        if review.decision:
            # A strict decision is durable, so cooldown retries never incur a
            # second MiniMax request for the same keyframe.
            try:
                decision = parse_minimax_decision(review.decision)
            except ValueError:
                decision = ""
        else:
            decision = ""
        if not decision:
            try:
                decision = parse_minimax_decision(self.classifier.classify(frame))
                review = self.queue.record_decision(review, decision, decided_at=now)
            except Exception as exc:
                self.queue.retry(
                    review,
                    "{}: {}".format(type(exc).__name__, exc),
                    now_epoch=now,
                    delay_seconds=self.retry_delay_seconds,
                )
                return {"status": "retry", "review_id": review.review_id}

        if decision == DECISION_NOT_FULL:
            self.queue.acknowledge(review, "not_full", decision)
            return {
                "status": "not_full",
                "review_id": review.review_id,
                "decision": decision,
            }

        # Only a strict 已满 result is allowed to become an AlarmCandidate.
        candidate = AlarmCandidate(
            event_name="垃圾桶已满",
            occurred_at=review.occurred_at,
            frame_bgr=frame,
            point_name=review.point_name,
            evidence={
                "review_result": DECISION_FULL,
                "point_id": review.point_id,
                "quality_score": review.quality_score,
            },
        )
        try:
            alarm_record = self.publisher.try_publish(candidate)
        except Exception as exc:
            self.queue.retry(
                review,
                "alarm publish failed: {}: {}".format(type(exc).__name__, exc),
                now_epoch=now,
                delay_seconds=self.retry_delay_seconds,
            )
            return {"status": "retry", "review_id": review.review_id}
        if not alarm_record:
            remaining = 0.0
            try:
                remaining = float(self.publisher.remaining_seconds(candidate))
            except Exception:
                remaining = 0.0
            self.queue.retry(
                review,
                "alarm cooldown not accepted",
                now_epoch=now,
                delay_seconds=max(self.retry_delay_seconds, remaining),
            )
            return {"status": "cooldown", "review_id": review.review_id}

        # Consumption happens only after the cooldown gate accepted the event
        # and the alarm repository returned a durable public record.
        self.queue.acknowledge(
            review,
            "alarm_saved",
            decision,
            alarm_record=alarm_record,
        )
        return {
            "status": "alarm_saved",
            "review_id": review.review_id,
            "decision": decision,
            "alarm": dict(alarm_record),
        }
