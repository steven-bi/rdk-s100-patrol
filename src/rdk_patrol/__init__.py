"""Unified patrol detection runtime for RDK S100."""

from .contracts import (
    AlarmCandidate,
    DepthEstimate,
    Detection,
    FrameEnvelope,
    NavigationEvent,
    PointContext,
)

__all__ = [
    "AlarmCandidate",
    "DepthEstimate",
    "Detection",
    "FrameEnvelope",
    "NavigationEvent",
    "PointContext",
]

__version__ = "1.0.0"

