from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoPair:
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    physical_left_view: str
    physical_right_view: str


def split_gs130w_vertical(
    combined_bgr: np.ndarray,
    *,
    physical_left_view: str = "bottom",
    physical_right_view: str = "top",
    rotate: str = "none",
) -> StereoPair:
    """Split GS130W's vertically combined image using explicit lens mapping.

    The driver labels halves by layout, not physical lens.  Confirm the mapping
    once with an occlusion test and persist it; silently guessing left/right can
    invert disparity and makes all distances invalid.
    """

    if combined_bgr is None or combined_bgr.size == 0 or combined_bgr.ndim not in (2, 3):
        raise ValueError("combined GS130W frame is empty or malformed")
    height, _width = combined_bgr.shape[:2]
    if height < 2 or height % 2:
        raise ValueError("GS130W vertical combined frame must have an even height")
    allowed = {"top", "bottom"}
    if (
        physical_left_view not in allowed
        or physical_right_view not in allowed
        or physical_left_view == physical_right_view
    ):
        raise ValueError("physical left/right must be an explicit top/bottom permutation")
    top, bottom = np.split(combined_bgr, 2, axis=0)
    views = {"top": top, "bottom": bottom}
    left = _rotate(views[physical_left_view], rotate)
    right = _rotate(views[physical_right_view], rotate)
    return StereoPair(
        left_bgr=left,
        right_bgr=right,
        physical_left_view=physical_left_view,
        physical_right_view=physical_right_view,
    )


def _rotate(image: np.ndarray, rotation: str) -> np.ndarray:
    normalized = str(rotation).lower().replace("_", "")
    if normalized in {"", "none", "0"}:
        return image
    if normalized in {"ccw90", "90ccw"}:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if normalized in {"cw90", "90cw"}:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized in {"180", "rotate180"}:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"unsupported stereo view rotation: {rotation}")
