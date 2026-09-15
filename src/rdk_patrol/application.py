from __future__ import annotations

"""Lifecycle and watchdog for the unified board process."""

from dataclasses import asdict, dataclass, is_dataclass
import logging
import threading
import time
from typing import Any

from .health import HealthFileReporter, HealthRegistry
from .io import FrameHub
from .processor import PatrolFrameProcessor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    processed_frames: int
    alarms_saved: int
    started_at: float
    stopped_at: float
    reason: str

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.stopped_at - self.started_at)


class RuntimeApplication:
    def __init__(
        self,
        *,
        frame_hub: FrameHub,
        camera_source: Any,
        processor: PatrolFrameProcessor,
        health: HealthRegistry,
        health_reporter: HealthFileReporter,
        point_worker: Any | None = None,
        depth_worker: Any | None = None,
        recorder_worker: Any | None = None,
        review_worker: Any | None = None,
        review_queue: Any | None = None,
        web_server: Any | None = None,
        camera_stale_seconds: float = 1.0,
        target_fps: float = 15.0,
        max_consecutive_inference_failures: int = 10,
        source_exhausted: Any | None = None,
        navigation_bridge: Any | None = None,
    ) -> None:
        self.frame_hub = frame_hub
        self.camera_source = camera_source
        self.processor = processor
        self.health = health
        self.health_reporter = health_reporter
        self.point_worker = point_worker
        self.depth_worker = depth_worker
        self.recorder_worker = recorder_worker
        self.review_worker = review_worker
        self.review_queue = review_queue
        self.web_server = web_server
        self.camera_stale_seconds = max(0.1, float(camera_stale_seconds))
        self.target_fps = float(target_fps)
        self.max_consecutive_inference_failures = max(
            1, int(max_consecutive_inference_failures)
        )
        self.source_exhausted = source_exhausted
        self.navigation_bridge = navigation_bridge
        self._started = False
        self._start_monotonic: float | None = None

    def start(self) -> None:
        if self._started:
            return
        self._register_health_components()
        self.health.update("runtime", state="starting", message="正在启动")
        started: list[Any] = []
        try:
            for worker in (
                self.point_worker,
                self.depth_worker,
                self.recorder_worker,
                self.review_worker,
            ):
                if worker is not None:
                    worker.start()
                    started.append(worker)
            self.health_reporter.start()
            started.append(self.health_reporter)
            if self.web_server is not None:
                address = self.web_server.start_background()
                self.health.heartbeat(
                    "web",
                    metrics={"bind": address[0], "port": address[1]},
                )
                started.append(self.web_server)
            self.camera_source.start()
            started.append(self.camera_source)
        except Exception:
            for component in reversed(started):
                self._safe_stop(component)
            self.health.failure("runtime", "startup failed", failed=True)
            raise
        self._started = True
        self._start_monotonic = time.monotonic()
        self.health.heartbeat("runtime", message="运行中")

    def stop(self) -> None:
        if not self._started:
            return
        self.health.update("runtime", state="stopped", message="正在停止")
        self._safe_stop(self.camera_source)
        self._safe_stop(self.review_worker)
        self._safe_stop(self.depth_worker)
        self._safe_stop(self.point_worker)
        self._safe_stop(self.recorder_worker, drain=True)
        self._safe_stop(self.web_server)
        self.health.update("runtime", state="stopped", message="已停止")
        self._safe_stop(self.health_reporter)
        self._started = False

    def run(
        self,
        *,
        stop_event: threading.Event | None = None,
        max_frames: int = 0,
        duration_seconds: float = 0.0,
    ) -> RunSummary:
        external_stop = stop_event or threading.Event()
        self.start()
        started_wall = time.time()
        started_monotonic = time.monotonic()
        last_sequence = 0
        last_frame_seen_at = started_monotonic
        processed = 0
        alarms = 0
        reason = "stopped"
        try:
            while not external_stop.is_set():
                if max_frames > 0 and processed >= int(max_frames):
                    reason = "max_frames"
                    break
                if (
                    duration_seconds > 0
                    and time.monotonic() - started_monotonic
                    >= float(duration_seconds)
                ):
                    reason = "duration"
                    break
                envelope = self.frame_hub.latest(
                    after_sequence=last_sequence,
                    timeout=0.20,
                    copy=False,
                )
                now = time.monotonic()
                if envelope is None:
                    if (
                        callable(self.source_exhausted)
                        and self.source_exhausted()
                    ):
                        reason = "source_exhausted"
                        break
                    stale = max(0.0, now - last_frame_seen_at)
                    state = (
                        "failed"
                        if stale >= self.camera_stale_seconds
                        else "degraded"
                    )
                    self.health.update(
                        "camera",
                        state=state,
                        message=f"no new frame for {stale:.2f}s",
                        metrics={"stale_seconds": round(stale, 3)},
                    )
                    self._update_background_health(now)
                    continue

                last_sequence = int(envelope.sequence)
                last_frame_seen_at = now
                self.health.heartbeat(
                    "camera",
                    metrics={
                        "sequence": last_sequence,
                        "frame_age_seconds": round(
                            max(0.0, now - envelope.monotonic_ts), 4
                        ),
                        "shape": list(envelope.combined_bgr.shape),
                    },
                )
                report = self.processor.process(envelope)
                processed += 1
                alarms += len(report.alarm_records)
                if report.errors:
                    LOGGER.warning(
                        "frame=%s errors=%s",
                        report.sequence,
                        " | ".join(report.errors),
                    )
                if (
                    self.processor.consecutive_inference_failures
                    >= self.max_consecutive_inference_failures
                ):
                    reason = "inference_failure_watchdog"
                    raise RuntimeError(
                        "inference failed "
                        f"{self.processor.consecutive_inference_failures} "
                        "consecutive frames"
                    )
                self._update_background_health(now)
                self._update_performance_health(now, started_monotonic)
        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
        finally:
            stopped_wall = time.time()
            self.stop()
        return RunSummary(
            processed_frames=processed,
            alarms_saved=alarms,
            started_at=started_wall,
            stopped_at=stopped_wall,
            reason=reason,
        )

    def _register_health_components(self) -> None:
        required = {
            "runtime": None,
            "camera": max(2.0, self.camera_stale_seconds * 2.0),
            "pipeline": 2.0,
            "inference": 2.0,
            "engine_parking": 2.0,
            "engine_night_people": 2.0,
            "engine_flame": 2.0,
            "alarms": None,
        }
        optional = {
            "localization": 3.0,
            "stereo": None,
            "garbage_capture": 3.0,
            "garbage_review": None,
            "recording": 3.0,
            "web": None,
            "ding_talk_outbox": None,
        }
        for name, stale in required.items():
            self.health.register(
                name, required=True, stale_after_seconds=stale
            )
        for name, stale in optional.items():
            self.health.register(
                name, required=False, stale_after_seconds=stale
            )
        self.health.heartbeat("alarms", message="本地持久化仓库已就绪")
        self.health.update(
            "ding_talk_outbox",
            state="stopped",
            message="首版已按要求关闭，仅保留接口",
            metrics={"enabled": False},
        )

    def _update_background_health(self, now: float) -> None:
        if self.recorder_worker is not None:
            stats = self.recorder_worker.stats()
            state = "degraded" if stats.failures or stats.dropped else "ok"
            self.health.update(
                "recording",
                state=state,
                message=stats.last_error or (
                    f"dropped={stats.dropped}" if stats.dropped else ""
                ),
                metrics=_stats_dict(stats),
            )
        if self.review_worker is not None:
            stats = self.review_worker.stats()
            pending = (
                self.review_queue.pending_count()
                if self.review_queue is not None
                else None
            )
            state = "degraded" if stats.last_error else "ok"
            metrics = _stats_dict(stats)
            metrics["pending_queue"] = pending
            self.health.update(
                "garbage_review",
                state=state,
                message=stats.last_error or "",
                metrics=metrics,
            )

    def _update_performance_health(
        self,
        now: float,
        started_monotonic: float,
    ) -> None:
        if now - started_monotonic < 10.0:
            return
        rate = self.processor.loop_rate.rate(now)
        self.health.set_global_metrics(
            target_fps=self.target_fps,
            measured_pipeline_fps_5s=round(rate, 3),
            target_fps_met=bool(rate >= self.target_fps),
        )
        if rate < self.target_fps:
            self.health.update(
                "pipeline",
                state="degraded",
                message=(
                    f"measured {rate:.2f} FPS below required "
                    f"{self.target_fps:.2f} FPS"
                ),
                metrics={"fps_5s": round(rate, 3)},
            )

    @staticmethod
    def _safe_stop(component: Any | None, **kwargs: Any) -> None:
        if component is None:
            return
        stopper = getattr(component, "stop", None)
        if not callable(stopper):
            stopper = getattr(component, "close", None)
        if not callable(stopper):
            return
        try:
            stopper(**kwargs)
        except TypeError:
            try:
                stopper()
            except Exception:
                pass
        except Exception:
            pass


def _stats_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raw = dict(vars(value))
    for key, item in list(raw.items()):
        if isinstance(item, (str, int, float, bool)) or item is None:
            continue
        if isinstance(item, dict):
            raw[key] = item
        else:
            raw[key] = str(item)
    return raw
