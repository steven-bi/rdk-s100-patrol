"""Camera input primitives shared by every patrol detector."""

from .frame_hub import (
    FrameHub,
    FrameHubStats,
    LatestFrameBuffer,
    LatestFrameSlot,
    build_frame_envelope,
    split_stereo_frame,
)

__all__ = [
    "FrameHub",
    "FrameHubStats",
    "LatestFrameBuffer",
    "LatestFrameSlot",
    "build_frame_envelope",
    "split_stereo_frame",
]
