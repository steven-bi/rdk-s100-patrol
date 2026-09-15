"""Transactional event-rule engines for the unified patrol runtime."""

from .base import EngineResult, ReservingEngine, ResultOrToken
from .flame import FLAME_EVENT_NAME, FlameDetectionEngine, FlameEngine
from .night_people import (
    CROWD_EVENT_NAME,
    PERSON_EVENT_NAME,
    NightPeopleEngine,
)
from .parking import PARKING_EVENT_NAME, ParkingEngine, VehicleParkingEngine

__all__ = [
    "CROWD_EVENT_NAME",
    "EngineResult",
    "FLAME_EVENT_NAME",
    "FlameDetectionEngine",
    "FlameEngine",
    "NightPeopleEngine",
    "PARKING_EVENT_NAME",
    "PERSON_EVENT_NAME",
    "ParkingEngine",
    "ReservingEngine",
    "ResultOrToken",
    "VehicleParkingEngine",
]
