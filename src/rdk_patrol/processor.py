from __future__ import annotations

"""One-frame coordinator: one inference result, four concurrent functions."""

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .async_workers import AsyncDepthEstimator, AsyncPointResolver
from .contracts import DepthEstimate, Detection, FrameEnvelope, PointContext
from .engines import FlameEngine, NightPeopleEngine, ParkingEngine
from .engines.base import EngineResult, ReservingEngine
from .evidence import annotate_live_frame, enrich_alarm_candidate
from .garbage_runtime import GarbagePointCoordinator
from .health import HealthRegistry, RollingRate
from .recording_worker import AsyncDualStreamRecorder


@dataclass
class FrameProcessingReport:
    sequence: int
    occurred_at: float
    detections: tuple[Detection, ...]
    point_context: PointContext | None
    inference_ms: float | None
    alarm_records: list[Mapping[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    annotated_bgr: np.ndarray | None = None


class PatrolFrameProcessor:
    """Fan one frame/inference result into every rule without duplicate HBM work."""

    def __init__(
        self,
        *,
        inference: Any,
        parking_engine: ParkingEngine,
        night_engine: NightPeopleEngine,
        flame_engine: FlameEngine,
        alarm_publisher: Any,
        delivery_outbox: Any | None = None,
        point_worker: AsyncPointResolver | None = None,
        depth_worker: AsyncDepthEstimator | None = None,
        garbage_coordinator: GarbagePointCoordinator | None = None,
        recorder_worker: AsyncDualStreamRecorder | None = None,
        health: HealthRegistry | None = None,
        target_fps: float = 15.0,
        point_context_max_age_seconds: float = 2.0,
        depth_max_age_seconds: float = 1.0,
        physical_left_view: str = "",
        physical_right_view: str = "",
    ) -> None:
        self.inference = inference
        self.parking_engine = parking_engine
        self.night_engine = night_engine
        self.flame_engine = flame_engine
        self.alarm_publisher = alarm_publisher
        self.delivery_outbox = delivery_outbox
        self.point_worker = point_worker
        self.depth_worker = depth_worker
        self.garbage_coordinator = garbage_coordinator
        self.recorder_worker = recorder_worker
        self.health = health
        self.target_fps = float(target_fps)
        self.point_context_max_age_seconds = float(
            point_context_max_age_seconds
        )
        self.depth_max_age_seconds = float(depth_max_age_seconds)
        self.physical_left_view = str(physical_left_view)
        self.physical_right_view = str(physical_right_view)
        self.loop_rate = RollingRate(window_seconds=5.0)
        self.inference_rate = RollingRate(window_seconds=5.0)
        self._consecutive_inference_failures = 0

    @property
    def consecutive_inference_failures(self) -> int:
        return self._consecutive_inference_failures

    def process(
        self,
        envelope: FrameEnvelope,
        *,
        occurred_at: float | None = None,
    ) -> FrameProcessingReport:
        wall_epoch = (
            _frame_epoch(envelope) if occurred_at is None else float(occurred_at)
        )
        current = float(envelope.monotonic_ts)
        errors: list[str] = []
        records: list[Mapping[str, str]] = []

        point_context = self._point_context(envelope, current, errors)
        self._process_garbage(
            point_context,
            envelope,
            current,
            wall_epoch,
            errors,
        )

        detections: tuple[Detection, ...] = ()
        inference_ms: float | None = None
        frame_valid = True
        try:
            inference_result = self.inference.infer_envelope(
                envelope, now=current
            )
            detections = tuple(inference_result.detections)
            inference_ms = float(inference_result.inference_ms)
            self._consecutive_inference_failures = 0
            self.inference_rate.mark(current)
            self._heartbeat(
                "inference",
                {
                    "fps_5s": round(self.inference_rate.rate(current), 3),
                    "inference_ms": round(inference_ms, 3),
                    "preprocessing_ms": round(
                        float(inference_result.preprocessing_ms), 3
                    ),
                    "decoding_ms": round(
                        float(inference_result.decoding_ms), 3
                    ),
                    "detections": len(detections),
                },
            )
        except Exception as exc:
            frame_valid = False
            self._consecutive_inference_failures += 1
            message = f"inference: {type(exc).__name__}: {exc}"
            errors.append(message)
            self._failure("inference", message)

        self._submit_depth(envelope, detections, current, errors)
        engine_calls: Sequence[tuple[ReservingEngine, Any]] = (
            (
                self.parking_engine,
                lambda: self.parking_engine.process(
                    detections,
                    point_context,
                    envelope.detection_bgr,
                    now=current,
                    occurred_at=wall_epoch,
                    frame_valid=frame_valid,
                ),
            ),
            (
                self.night_engine,
                lambda: self.night_engine.process(
                    detections,
                    envelope.detection_bgr,
                    now=current,
                    occurred_at=wall_epoch,
                    frame_valid=frame_valid,
                ),
            ),
            (
                self.flame_engine,
                lambda: self.flame_engine.process(
                    detections,
                    envelope.detection_bgr,
                    now=current,
                    occurred_at=wall_epoch,
                    depth_estimate=None,
                    scope_id="full_frame",
                    frame_valid=frame_valid,
                ),
            ),
        )
        for engine, call in engine_calls:
            try:
                results = list(call())
                self._heartbeat(
                    f"engine_{engine.engine_name}",
                    {"pending_results": len(results)},
                )
            except Exception as exc:
                message = (
                    f"engine_{engine.engine_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(message)
                self._failure(f"engine_{engine.engine_name}", message)
                continue
            for result in results:
                if engine is self.flame_engine:
                    self._apply_depth_to_flame_result(
                        result, current
                    )
                saved = self._persist_result(engine, result, errors)
                if saved is not None:
                    records.append(saved)

        self.loop_rate.mark(current)
        annotated = annotate_live_frame(
            envelope.detection_bgr,
            detections,
            point_context=point_context,
            loop_fps=self.loop_rate.rate(current),
        )
        if self.recorder_worker is not None:
            try:
                self.recorder_worker.submit(
                    envelope.combined_bgr,
                    annotated,
                    timestamp=wall_epoch,
                )
            except Exception as exc:
                message = f"recording_submit: {type(exc).__name__}: {exc}"
                errors.append(message)
                self._failure("recording", message)
        self._heartbeat(
            "pipeline",
            {
                "fps_5s": round(self.loop_rate.rate(current), 3),
                "sequence": int(envelope.sequence),
                "alarms_this_frame": len(records),
            },
        )
        return FrameProcessingReport(
            sequence=int(envelope.sequence),
            occurred_at=wall_epoch,
            detections=detections,
            point_context=point_context,
            inference_ms=inference_ms,
            alarm_records=records,
            errors=errors,
            annotated_bgr=annotated,
        )

    def _point_context(
        self,
        envelope: FrameEnvelope,
        now: float,
        errors: list[str],
    ) -> PointContext | None:
        if self.point_worker is None:
            return None
        try:
            self.point_worker.submit(envelope.detection_bgr, now)
            context = self.point_worker.latest(
                now=now,
                max_age_seconds=self.point_context_max_age_seconds,
            )
            stats = self.point_worker.stats()
            state = "ok" if stats.last_error is None else "degraded"
            if self.health is not None:
                self.health.update(
                    "localization",
                    state=state,
                    message=stats.last_error or "",
                    metrics={
                        "submitted": stats.submitted,
                        "completed": stats.completed,
                        "replaced": stats.replaced,
                        "point_id": (
                            None if context is None else context.point_id
                        ),
                        "source": (
                            None if context is None else context.source
                        ),
                    },
                )
            return context
        except Exception as exc:
            message = f"localization: {type(exc).__name__}: {exc}"
            errors.append(message)
            self._failure("localization", message)
            return None

    def _process_garbage(
        self,
        context: PointContext | None,
        envelope: FrameEnvelope,
        now: float,
        occurred_at: float,
        errors: list[str],
    ) -> None:
        if self.garbage_coordinator is None:
            return
        try:
            review = self.garbage_coordinator.process(
                context,
                envelope.detection_bgr,
                now_monotonic=now,
                occurred_at=occurred_at,
            )
            state = self.garbage_coordinator.state()
            self._heartbeat(
                "garbage_capture",
                {
                    "active_point_id": state.active_point_id,
                    "windows_started": state.windows_started,
                    "reviews_enqueued": state.reviews_enqueued,
                    "windows_aborted": state.windows_aborted,
                    "enqueued_this_frame": review is not None,
                },
            )
        except Exception as exc:
            message = f"garbage_capture: {type(exc).__name__}: {exc}"
            errors.append(message)
            self._failure("garbage_capture", message)

    def _submit_depth(
        self,
        envelope: FrameEnvelope,
        detections: Sequence[Detection],
        now: float,
        errors: list[str],
    ) -> None:
        if self.depth_worker is None:
            return
        fires = [
            detection
            for detection in detections
            if detection.class_name == "fire"
        ]
        if not fires:
            return
        pair = _stereo_pair(
            envelope,
            self.physical_left_view,
            self.physical_right_view,
        )
        if pair is None:
            return
        best = max(fires, key=lambda item: float(item.confidence))
        try:
            self.depth_worker.submit(pair[0], pair[1], best.box, now)
            stats = self.depth_worker.stats()
            self._heartbeat(
                "stereo",
                {
                    "submitted": stats.submitted,
                    "completed": stats.completed,
                    "failures": stats.failures,
                },
            )
        except Exception as exc:
            message = f"stereo_submit: {type(exc).__name__}: {exc}"
            errors.append(message)
            self._failure("stereo", message)

    def _apply_depth_to_flame_result(
        self,
        result: EngineResult,
        now: float,
    ) -> None:
        estimate = DepthEstimate.unavailable(
            "stereo_not_configured", computed_at=now
        )
        if self.depth_worker is not None and result.detections:
            estimate = self.depth_worker.latest(
                now=now,
                max_age_seconds=self.depth_max_age_seconds,
                fire_box=result.detections[0].box,
            )
        evidence = result.candidate.evidence
        evidence["distance_m"] = (
            round(float(estimate.distance_m), 3)
            if estimate.valid and estimate.distance_m is not None
            else None
        )
        evidence["distance_text"] = (
            f"{float(estimate.distance_m):.2f} m"
            if estimate.valid and estimate.distance_m is not None
            else "不可用"
        )
        evidence["depth_valid"] = bool(estimate.valid)
        evidence["depth_reason"] = estimate.reason
        evidence["depth_valid_pixels"] = int(estimate.valid_pixels)

    def _persist_result(
        self,
        engine: ReservingEngine,
        result: EngineResult,
        errors: list[str],
    ) -> Mapping[str, str] | None:
        enrich_alarm_candidate(result.candidate, result.detections)
        try:
            record = self.alarm_publisher.try_publish(result.candidate)
        except Exception as exc:
            engine.release(result)
            message = (
                f"alarm_persist_{engine.engine_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            errors.append(message)
            self._failure("alarms", message)
            return None
        # A falsey result means a durable cross-restart cooldown already
        # covers this incident. It is not a storage failure, so the in-memory
        # engine reservation is committed to avoid a hot retry loop.
        engine.commit(result)
        if record and self.delivery_outbox is not None:
            try:
                delivery = self.delivery_outbox.enqueue(record)
                self._heartbeat(
                    "ding_talk_outbox",
                    {
                        "enabled": bool(
                            getattr(self.delivery_outbox, "enabled", False)
                        ),
                        "status": getattr(delivery, "status", "unknown"),
                    },
                )
            except Exception as exc:
                # Delivery is outside the first-release durability contract;
                # the already-saved local alarm must never be rolled back.
                self._failure(
                    "ding_talk_outbox",
                    f"{type(exc).__name__}: {exc}",
                )
        self._heartbeat(
            "alarms",
            {"last_event": result.event_name, "saved": bool(record)},
        )
        return record

    def _heartbeat(self, name: str, metrics: Mapping[str, Any]) -> None:
        if self.health is not None:
            self.health.heartbeat(name, metrics=metrics)

    def _failure(self, name: str, message: str) -> None:
        if self.health is not None:
            self.health.failure(name, message)


def _frame_epoch(envelope: FrameEnvelope) -> float:
    source = envelope.source_ts
    now = time.time()
    if source is None:
        return now
    # Reject monotonic timestamps and unset camera clocks as wall time.
    if abs(float(source) - now) > 24 * 60 * 60:
        return now
    return float(source)


def _stereo_pair(
    envelope: FrameEnvelope,
    physical_left_view: str,
    physical_right_view: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if envelope.auxiliary_bgr is None:
        return None
    detection_name = str(envelope.metadata.get("detection_view") or "")
    auxiliary_name = str(envelope.metadata.get("auxiliary_view") or "")
    views = {
        detection_name: envelope.detection_bgr,
        auxiliary_name: envelope.auxiliary_bgr,
    }
    left = views.get(str(physical_left_view))
    right = views.get(str(physical_right_view))
    if left is None or right is None or left is right:
        return None
    return left, right
