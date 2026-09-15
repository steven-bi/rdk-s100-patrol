from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from rdk_patrol.contracts import DepthEstimate

from .calibration import StereoCalibration, load_stereo_config


class StereoDepthEstimator:
    def __init__(
        self,
        calibration: StereoCalibration,
        *,
        min_valid_pixels: int = 24,
        min_disparity: float = 0.75,
        min_distance_m: float = 0.15,
        max_distance_m: float = 30.0,
        max_relative_mad: float = 0.40,
        matcher_min_disparity: int = 0,
        num_disparities: int = 128,
        block_size: int = 5,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 80,
        speckle_range: int = 2,
    ) -> None:
        self.calibration = calibration
        self.min_valid_pixels = max(3, int(min_valid_pixels))
        self.min_disparity = max(0.0, float(min_disparity))
        self.min_distance_m = max(0.0, float(min_distance_m))
        self.max_distance_m = max(self.min_distance_m, float(max_distance_m))
        self.max_relative_mad = max(0.01, float(max_relative_mad))
        self.matcher_min_disparity = int(matcher_min_disparity)
        self.num_disparities = max(
            16, min(512, ((int(num_disparities) + 15) // 16) * 16)
        )
        self.block_size = max(3, min(21, int(block_size) | 1))
        self.uniqueness_ratio = max(0, min(100, int(uniqueness_ratio)))
        self.speckle_window_size = max(0, int(speckle_window_size))
        self.speckle_range = max(0, int(speckle_range))
        self._maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._matcher_factory: Callable[[int], object] = self._create_matcher

    @classmethod
    def from_file(cls, calibration_path: str | Path) -> "StereoDepthEstimator":
        # Invalid/missing calibration is represented as an unavailable result,
        # never as a startup exception.
        calibration, matcher = load_stereo_config(calibration_path)
        return cls(
            calibration,
            min_valid_pixels=_safe_int(matcher, "minimum_valid_pixels", 24),
            min_disparity=_safe_float(
                matcher,
                "valid_disparity_min",
                _safe_float(matcher, "min_disparity", 0.0) + 0.75,
            ),
            min_distance_m=_safe_float(matcher, "min_distance_m", 0.15),
            max_distance_m=_safe_float(matcher, "max_distance_m", 30.0),
            max_relative_mad=_safe_float(matcher, "max_relative_mad", 0.40),
            matcher_min_disparity=_safe_int(matcher, "min_disparity", 0),
            num_disparities=_safe_int(matcher, "num_disparities", 128),
            block_size=_safe_int(matcher, "block_size", 5),
            uniqueness_ratio=_safe_int(matcher, "uniqueness_ratio", 10),
            speckle_window_size=_safe_int(
                matcher, "speckle_window_size", 80
            ),
            speckle_range=_safe_int(matcher, "speckle_range", 2),
        )

    @property
    def physical_left_view(self) -> str:
        return self.calibration.physical_left_view

    @property
    def physical_right_view(self) -> str:
        return self.calibration.physical_right_view

    def validate_view_mapping(
        self, detection_view: str, auxiliary_view: str
    ) -> tuple[bool, str]:
        """Ensure a detection box indexes the physical-left disparity image."""

        if not self.calibration.valid:
            return False, self.calibration.reason
        if not self.physical_left_view or not self.physical_right_view:
            return False, "physical_view_mapping_missing"
        if str(detection_view) != self.physical_left_view:
            return False, "detection_view_is_not_physical_left"
        if str(auxiliary_view) != self.physical_right_view:
            return False, "auxiliary_view_is_not_physical_right"
        return True, "ok"

    def estimate(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fire_box: tuple[float, float, float, float],
        computed_at: float,
    ) -> DepthEstimate:
        calibration = self.calibration
        if not calibration.valid:
            return DepthEstimate.unavailable(
                calibration.reason, computed_at=float(computed_at)
            )
        if (
            left_bgr is None
            or right_bgr is None
            or left_bgr.size == 0
            or right_bgr.size == 0
        ):
            return DepthEstimate.unavailable(
                "stereo_frame_empty", computed_at=float(computed_at)
            )
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            return DepthEstimate.unavailable(
                "stereo_frame_shape_mismatch", computed_at=float(computed_at)
            )
        height, width = left_bgr.shape[:2]
        if calibration.image_size != (width, height):
            return DepthEstimate.unavailable(
                "stereo_frame_calibration_size_mismatch", computed_at=float(computed_at)
            )
        try:
            left_rectified, right_rectified = self._rectify(left_bgr, right_bgr)
            polygon = self._fire_polygon_rectified(fire_box)
            if polygon is None:
                return DepthEstimate.unavailable(
                    "fire_box_invalid", computed_at=float(computed_at)
                )
            gray_left = (
                left_rectified
                if left_rectified.ndim == 2
                else cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            )
            gray_right = (
                right_rectified
                if right_rectified.ndim == 2
                else cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            )
            matcher = self._matcher_factory(width)
            disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
            xyz = cv2.reprojectImageTo3D(
                disparity,
                np.asarray(calibration.q_matrix, dtype=np.float64),
                handleMissingValues=False,
            )
        except (cv2.error, ValueError, TypeError, AttributeError):
            return DepthEstimate.unavailable(
                "stereo_computation_failed", computed_at=float(computed_at)
            )

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
        # Avoid the uncertain flame-box boundary and stereo occlusion edges.
        box_width = max(1.0, float(np.ptp(polygon[:, 0])))
        box_height = max(1.0, float(np.ptp(polygon[:, 1])))
        erosion = max(1, int(round(min(box_width, box_height) * 0.08)))
        if erosion > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion * 2 + 1,) * 2)
            mask = cv2.erode(mask, kernel)
        distances = np.linalg.norm(xyz, axis=2)
        valid = (
            (mask > 0)
            & np.isfinite(distances)
            & np.isfinite(disparity)
            & (disparity > self.min_disparity)
            & (distances >= self.min_distance_m)
            & (distances <= self.max_distance_m)
            & (xyz[:, :, 2] > 0)
        )
        values = distances[valid].astype(np.float64)
        disparities = disparity[valid].astype(np.float64)
        if values.size < self.min_valid_pixels:
            return DepthEstimate(
                valid=False,
                distance_m=None,
                reason="insufficient_valid_stereo_pixels",
                valid_pixels=int(values.size),
                computed_at=float(computed_at),
                quality={"required_valid_pixels": self.min_valid_pixels},
            )

        low, high = np.percentile(values, [10.0, 90.0])
        trimmed = values[(values >= low) & (values <= high)]
        if trimmed.size < self.min_valid_pixels:
            trimmed = values
        median = float(np.median(trimmed))
        mad = float(np.median(np.abs(trimmed - median)))
        relative_mad = mad / max(median, 1e-6)
        if relative_mad > self.max_relative_mad:
            return DepthEstimate(
                valid=False,
                distance_m=None,
                reason="stereo_depth_inconsistent",
                valid_pixels=int(values.size),
                computed_at=float(computed_at),
                quality={"median_m": median, "relative_mad": relative_mad},
            )
        return DepthEstimate(
            valid=True,
            distance_m=median,
            reason="ok",
            valid_pixels=int(values.size),
            computed_at=float(computed_at),
            quality={
                "median_disparity_px": float(np.median(disparities)),
                "relative_mad": relative_mad,
                "trimmed_pixels": int(trimmed.size),
                "distance_definition": "camera_to_fire_region_euclidean_m",
            },
        )

    def _rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.calibration
        if calibration.images_are_rectified:
            return left, right
        if self._maps is None:
            assert calibration.image_size is not None
            self._maps = (
                *cv2.initUndistortRectifyMap(
                    calibration.left_camera_matrix,
                    calibration.left_distortion,
                    calibration.left_rectification,
                    calibration.left_projection,
                    calibration.image_size,
                    cv2.CV_32FC1,
                ),
                *cv2.initUndistortRectifyMap(
                    calibration.right_camera_matrix,
                    calibration.right_distortion,
                    calibration.right_rectification,
                    calibration.right_projection,
                    calibration.image_size,
                    cv2.CV_32FC1,
                ),
            )
        left_x, left_y, right_x, right_y = self._maps
        return (
            cv2.remap(left, left_x, left_y, cv2.INTER_LINEAR),
            cv2.remap(right, right_x, right_y, cv2.INTER_LINEAR),
        )

    def _fire_polygon_rectified(
        self, fire_box: tuple[float, float, float, float]
    ) -> np.ndarray | None:
        try:
            x1, y1, x2, y2 = (float(value) for value in fire_box)
        except (TypeError, ValueError):
            return None
        width, height = self.calibration.image_size or (0, 0)
        x1, x2 = sorted((np.clip(x1, 0, width - 1), np.clip(x2, 0, width - 1)))
        y1, y2 = sorted((np.clip(y1, 0, height - 1), np.clip(y2, 0, height - 1)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        points = np.asarray(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        )
        if self.calibration.images_are_rectified:
            return points
        transformed = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            self.calibration.left_camera_matrix,
            self.calibration.left_distortion,
            R=self.calibration.left_rectification,
            P=self.calibration.left_projection[:, :3],
        ).reshape(-1, 2)
        return transformed if np.isfinite(transformed).all() else None

    def _create_matcher(self, width: int):
        width_limit = max(16, ((max(16, width - 16)) // 16) * 16)
        num_disparities = min(self.num_disparities, width_limit)
        block_size = self.block_size
        return cv2.StereoSGBM_create(
            minDisparity=self.matcher_min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )


def _safe_int(values: dict, key: str, default: int) -> int:
    try:
        return int(values.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(values: dict, key: str, default: float) -> float:
    try:
        value = float(values.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if np.isfinite(value) else float(default)
