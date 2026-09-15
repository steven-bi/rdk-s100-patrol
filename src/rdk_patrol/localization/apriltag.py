from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    area_px: float
    confidence: float
    quality_reason: str = "accepted"


class AprilTagDetector:
    """OpenCV AprilTag detector with conservative geometric quality gates.

    ``opencv-contrib`` is optional at runtime.  When its aruco backend (or the
    AprilTag 36h11 dictionary) is absent, ``available`` is false and ``detect``
    returns an empty list; callers can then use visual-fingerprint fallback.
    """

    def __init__(
        self,
        accepted_ids: Iterable[int],
        *,
        tag_family: str = "tag36h11",
        min_area_px: float = 225.0,
        min_side_px: float = 10.0,
        max_side_ratio: float = 3.0,
        border_margin_px: float = 2.0,
    ) -> None:
        self.accepted_ids = frozenset(int(value) for value in accepted_ids)
        self.tag_family = str(tag_family).lower().replace("_", "")
        self.min_area_px = max(1.0, float(min_area_px))
        self.min_side_px = max(1.0, float(min_side_px))
        self.max_side_ratio = max(1.0, float(max_side_ratio))
        self.border_margin_px = max(0.0, float(border_margin_px))
        self.available = False
        self.unavailable_reason = ""
        self._detector = None
        self._dictionary = None
        self._prepare_backend()

    def _prepare_backend(self) -> None:
        if self.tag_family not in {"tag36h11", "apriltag36h11"}:
            self.unavailable_reason = "unsupported_tag_family"
            return
        aruco = getattr(cv2, "aruco", None)
        dictionary_id = getattr(aruco, "DICT_APRILTAG_36h11", None) if aruco is not None else None
        if aruco is None or dictionary_id is None:
            self.unavailable_reason = "opencv_aruco_apriltag_backend_unavailable"
            return
        try:
            self._dictionary = aruco.getPredefinedDictionary(dictionary_id)
            if hasattr(aruco, "ArucoDetector"):
                parameters = aruco.DetectorParameters()
                if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
                    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
                self._detector = aruco.ArucoDetector(self._dictionary, parameters)
            self.available = True
        except (AttributeError, cv2.error):
            self.unavailable_reason = "opencv_aruco_apriltag_backend_initialization_failed"

    def detect(self, frame_bgr: np.ndarray) -> list[TagObservation]:
        if not self.available or frame_bgr is None or frame_bgr.size == 0:
            return []
        gray = (
            frame_bgr
            if frame_bgr.ndim == 2
            else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        )
        try:
            if self._detector is not None:
                corners, ids, _rejected = self._detector.detectMarkers(gray)
            else:
                corners, ids, _rejected = cv2.aruco.detectMarkers(gray, self._dictionary)
        except (AttributeError, cv2.error):
            self.available = False
            self.unavailable_reason = "opencv_aruco_apriltag_detection_failed"
            return []
        if ids is None:
            return []

        height, width = gray.shape[:2]
        observations: list[TagObservation] = []
        for raw_corners, raw_id in zip(corners, ids.reshape(-1).tolist()):
            tag_id = int(raw_id)
            if tag_id not in self.accepted_ids:
                continue
            points = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
            accepted, reason, area, side_ratio = self._quality(points, width, height)
            if not accepted:
                continue
            center = np.mean(points, axis=0)
            area_score = min(1.0, area / (self.min_area_px * 4.0))
            shape_score = 1.0 / max(1.0, side_ratio)
            confidence = float(np.clip(0.65 * area_score + 0.35 * shape_score, 0.0, 1.0))
            observations.append(
                TagObservation(
                    tag_id=tag_id,
                    corners=tuple((float(x), float(y)) for x, y in points),
                    center=(float(center[0]), float(center[1])),
                    area_px=float(area),
                    confidence=confidence,
                    quality_reason=reason,
                )
            )
        observations.sort(key=lambda item: (item.confidence, item.area_px), reverse=True)
        return observations

    def _quality(
        self, points: np.ndarray, width: int, height: int
    ) -> tuple[bool, str, float, float]:
        if points.shape != (4, 2) or not np.isfinite(points).all():
            return False, "invalid_corners", 0.0, float("inf")
        contour = points.reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        min_side = float(np.min(sides))
        max_side = float(np.max(sides))
        side_ratio = max_side / max(min_side, 1e-6)
        margin = self.border_margin_px
        inside = bool(
            np.all(points[:, 0] >= margin)
            and np.all(points[:, 0] < width - margin)
            and np.all(points[:, 1] >= margin)
            and np.all(points[:, 1] < height - margin)
        )
        if area < self.min_area_px:
            return False, "tag_area_too_small", area, side_ratio
        if min_side < self.min_side_px:
            return False, "tag_side_too_short", area, side_ratio
        if side_ratio > self.max_side_ratio:
            return False, "tag_shape_too_skewed", area, side_ratio
        if not bool(cv2.isContourConvex(contour.astype(np.float32))):
            return False, "tag_not_convex", area, side_ratio
        if not inside:
            return False, "tag_touches_frame_border", area, side_ratio
        return True, "accepted", area, side_ratio
