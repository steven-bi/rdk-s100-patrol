from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

from ..contracts import AlarmCandidate, Detection


@dataclass(frozen=True)
class EngineResult:
    """A reserved alarm candidate awaiting durable persistence.

    A caller must invoke the originating engine's :meth:`commit` after the
    image and record are durable.  If persistence fails it must invoke
    :meth:`release`, allowing the same still-valid incident to retry.
    """

    engine_name: str
    reservation_token: str
    candidate: AlarmCandidate
    detections: tuple[Detection, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine_name:
            raise ValueError("engine_name cannot be empty")
        if not self.reservation_token:
            raise ValueError("reservation_token cannot be empty")
    @property
    def token(self) -> str:
        return self.reservation_token

    @property
    def event_name(self) -> str:
        return self.candidate.event_name

    @property
    def occurred_at(self) -> float:
        return float(self.candidate.occurred_at)

    @property
    def point_name(self) -> str | None:
        return self.candidate.point_name

    @property
    def evidence(self) -> dict[str, Any]:
        return self.candidate.evidence


ResultOrToken = Union[EngineResult, str]


@runtime_checkable
class ReservingEngine(Protocol):
    engine_name: str

    def process(self, *args: Any, **kwargs: Any) -> Sequence[EngineResult]:
        ...

    def commit(self, result_or_token: ResultOrToken) -> bool:
        ...

    def release(self, result_or_token: ResultOrToken) -> bool:
        ...


def result_token(result_or_token: ResultOrToken) -> str:
    if isinstance(result_or_token, EngineResult):
        return result_or_token.reservation_token
    return str(result_or_token)
