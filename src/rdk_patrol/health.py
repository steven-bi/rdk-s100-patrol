from __future__ import annotations

"""Thread-safe health and performance reporting."""

from collections import deque
from dataclasses import dataclass, field
import platform
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .atomic_io import atomic_write_json
from .time_utils import format_beijing_time


VALID_STATES = frozenset({"starting", "ok", "degraded", "failed", "stopped"})
_STATE_RANK = {
    "ok": 0,
    "starting": 1,
    "stopped": 1,
    "degraded": 2,
    "failed": 3,
}


@dataclass
class _Component:
    name: str
    required: bool
    stale_after_seconds: float | None
    state: str = "starting"
    message: str = ""
    updated_monotonic: float = field(default_factory=time.monotonic)
    metrics: dict[str, Any] = field(default_factory=dict)


class RollingRate:
    """A bounded timestamp window suitable for frame/FPS counters."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = max(0.25, float(window_seconds))
        self._values: deque[float] = deque()
        self._lock = threading.Lock()

    def mark(self, timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._values.append(now)
            self._prune(now)

    def rate(self, timestamp: float | None = None) -> float:
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._prune(now)
            if len(self._values) < 2:
                return 0.0
            duration = min(
                self.window_seconds,
                max(1e-6, now - self._values[0]),
            )
            return float(len(self._values) - 1) / duration

    def count(self, timestamp: float | None = None) -> int:
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._prune(now)
            return len(self._values)

    def _prune(self, now: float) -> None:
        threshold = now - self.window_seconds
        while self._values and self._values[0] < threshold:
            self._values.popleft()


class HealthRegistry:
    """Collect component state for the LAN status page and status file."""

    def __init__(self, *, version: str = "1.0.0") -> None:
        self.version = str(version)
        self.started_wall = time.time()
        self.started_monotonic = time.monotonic()
        self._lock = threading.RLock()
        self._components: dict[str, _Component] = {}
        self._global_metrics: dict[str, Any] = {}

    def register(
        self,
        name: str,
        *,
        required: bool = True,
        stale_after_seconds: float | None = None,
    ) -> None:
        component_name = str(name)
        stale = (
            None
            if stale_after_seconds is None
            else max(0.1, float(stale_after_seconds))
        )
        with self._lock:
            existing = self._components.get(component_name)
            if existing is None:
                self._components[component_name] = _Component(
                    name=component_name,
                    required=bool(required),
                    stale_after_seconds=stale,
                )
            else:
                existing.required = bool(required)
                existing.stale_after_seconds = stale

    def update(
        self,
        name: str,
        *,
        state: str = "ok",
        message: str = "",
        metrics: Mapping[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        state_name = str(state)
        if state_name not in VALID_STATES:
            raise ValueError(f"unsupported health state: {state_name}")
        with self._lock:
            if name not in self._components:
                self.register(name)
            component = self._components[name]
            component.state = state_name
            component.message = str(message)
            component.updated_monotonic = (
                time.monotonic() if timestamp is None else float(timestamp)
            )
            if metrics:
                component.metrics.update(dict(metrics))

    def heartbeat(
        self,
        name: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        message: str = "",
    ) -> None:
        self.update(name, state="ok", message=message, metrics=metrics)

    def failure(
        self,
        name: str,
        error: BaseException | str,
        *,
        failed: bool = False,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(error, BaseException):
            message = f"{type(error).__name__}: {error}"
        else:
            message = str(error)
        self.update(
            name,
            state="failed" if failed else "degraded",
            message=message,
            metrics=metrics,
        )

    def set_global_metrics(self, **metrics: Any) -> None:
        with self._lock:
            self._global_metrics.update(metrics)

    def snapshot(
        self,
        *,
        now_monotonic: float | None = None,
        now_wall: float | None = None,
    ) -> dict[str, Any]:
        monotonic_now = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        wall_now = time.time() if now_wall is None else float(now_wall)
        with self._lock:
            rows: dict[str, dict[str, Any]] = {}
            overall_rank = 0
            for name in sorted(self._components):
                component = self._components[name]
                age = max(0.0, monotonic_now - component.updated_monotonic)
                state = component.state
                message = component.message
                stale = (
                    component.stale_after_seconds is not None
                    and age > component.stale_after_seconds
                )
                if stale and state not in {"failed", "stopped"}:
                    state = "failed" if component.required else "degraded"
                    message = (
                        f"heartbeat stale for {age:.2f}s "
                        f"(limit {component.stale_after_seconds:.2f}s)"
                    )
                if component.required:
                    overall_rank = max(overall_rank, _STATE_RANK[state])
                elif state == "failed":
                    overall_rank = max(overall_rank, _STATE_RANK["degraded"])
                rows[name] = {
                    "state": state,
                    "message": message,
                    "required": component.required,
                    "heartbeat_age_seconds": round(age, 3),
                    "metrics": dict(component.metrics),
                }

            overall = (
                "failed"
                if overall_rank >= _STATE_RANK["failed"]
                else "degraded"
                if overall_rank >= _STATE_RANK["degraded"]
                else "starting"
                if overall_rank >= _STATE_RANK["starting"]
                else "ok"
            )
            return {
                "schema_version": "rdk-patrol-health/v1",
                "status": overall,
                "version": self.version,
                "beijing_time": format_beijing_time(wall_now),
                "uptime_seconds": round(
                    max(0.0, monotonic_now - self.started_monotonic), 3
                ),
                "host": platform.node(),
                "components": rows,
                "metrics": dict(self._global_metrics),
            }


class HealthFileReporter:
    """Periodically atomically publish ``health.json``."""

    def __init__(
        self,
        registry: HealthRegistry,
        destination: str | Path,
        *,
        interval_seconds: float = 1.0,
        enrich: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.destination = Path(destination)
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.enrich = enrich
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="rdk-health-reporter",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, float(timeout_seconds)))
        self.write_once()

    def write_once(self) -> None:
        payload = self.registry.snapshot()
        if self.enrich is not None:
            try:
                extra = self.enrich()
            except Exception as exc:
                extra = {"reporter_enrich_error": f"{type(exc).__name__}: {exc}"}
            payload.update(dict(extra))
        atomic_write_json(self.destination, payload)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.write_once()
            except Exception:
                # A temporary disk failure should not terminate event detection.
                pass
            self._stop.wait(self.interval_seconds)
