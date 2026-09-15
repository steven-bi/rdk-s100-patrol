from __future__ import annotations

"""Disabled first-release delivery boundary for future DingTalk transport."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DeliveryResult:
    accepted: bool
    status: str


class DisabledDingTalkOutbox:
    """Explicit no-network adapter.

    The first release persists and serves alarms locally.  A future transport
    may implement the same ``enqueue`` method, but this adapter intentionally
    stores no secret and performs no HTTP request.
    """

    enabled = False

    def enqueue(self, record: Mapping[str, str]) -> DeliveryResult:
        # Touch the mapping to catch programming mistakes without retaining it.
        if "event_name" not in record or "keyframe_image" not in record:
            raise ValueError("invalid alarm record for delivery outbox")
        return DeliveryResult(False, "disabled_first_release")
