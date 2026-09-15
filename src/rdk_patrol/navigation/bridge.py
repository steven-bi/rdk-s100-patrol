from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

from rdk_patrol.contracts import NavigationEvent


NAVIGATION_HOLD_SECONDS = 5.0


class NavigationListener(Protocol):
    def __call__(self, event: NavigationEvent) -> None:
        ...


class NavigationBridge:
    """Synchronous, transport-neutral hooks for later navigation integration.

    It does not drive the robot. A future MapPilot/LingTu adapter only needs to
    translate its lifecycle into arrive/pause/resume/leave calls.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._listeners: List[NavigationListener] = []
        self._active_point_id: Optional[str] = None
        self._active_point_name: Optional[str] = None
        self._paused = False
        self._lock = threading.RLock()

    @property
    def active_point_id(self) -> Optional[str]:
        return self._active_point_id

    @property
    def paused(self) -> bool:
        return self._paused

    def subscribe(self, listener: NavigationListener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

        return unsubscribe

    def arrive(
        self,
        point_id: str,
        point_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Optional[float] = None,
    ) -> NavigationEvent:
        if not point_id:
            raise ValueError("arrive requires point_id")
        with self._lock:
            self._active_point_id = str(point_id)
            self._active_point_name = None if point_name is None else str(point_name)
            self._paused = False
        return self._emit(
            "arrive",
            point_id=str(point_id),
            point_name=point_name,
            metadata=metadata,
            occurred_at=occurred_at,
        )

    def pause(
        self,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Optional[float] = None,
    ) -> NavigationEvent:
        with self._lock:
            self._paused = True
            point_id = self._active_point_id
            point_name = self._active_point_name
        return self._emit(
            "pause",
            point_id=point_id,
            point_name=point_name,
            metadata=metadata,
            occurred_at=occurred_at,
        )

    def resume(
        self,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Optional[float] = None,
    ) -> NavigationEvent:
        with self._lock:
            self._paused = False
            point_id = self._active_point_id
            point_name = self._active_point_name
        return self._emit(
            "resume",
            point_id=point_id,
            point_name=point_name,
            metadata=metadata,
            occurred_at=occurred_at,
        )

    def leave(
        self,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Optional[float] = None,
    ) -> NavigationEvent:
        with self._lock:
            point_id = self._active_point_id
            point_name = self._active_point_name
            self._active_point_id = None
            self._active_point_name = None
            self._paused = False
        return self._emit(
            "leave",
            point_id=point_id,
            point_name=point_name,
            metadata=metadata,
            occurred_at=occurred_at,
        )

    def _emit(
        self,
        action: str,
        point_id: Optional[str],
        point_name: Optional[str],
        metadata: Optional[Mapping[str, Any]],
        occurred_at: Optional[float],
    ) -> NavigationEvent:
        event = NavigationEvent(
            action=action,
            occurred_at=float(self.clock() if occurred_at is None else occurred_at),
            point_id=point_id,
            point_name=point_name,
            hold_seconds=NAVIGATION_HOLD_SECONDS,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener(event)
        return event
