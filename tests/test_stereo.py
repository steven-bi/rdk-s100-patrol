from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from rdk_patrol.stereo import (
    StereoDepthEstimator,
    calibration_from_mapping,
    split_gs130w_vertical,
)


class _FixedMatcher:
    def __init__(self, shape: tuple[int, int], disparity: float) -> None:
        self.value = np.full(shape, round(disparity * 16), dtype=np.int16)

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        assert left.shape == self.value.shape
        assert right.shape == self.value.shape
        return self.value.copy()


def _calibration(width: int = 160, height: int = 120):
    fx = 200.0
    baseline = 0.1
    cx, cy = width / 2.0, height / 2.0
    camera = [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]
    p1 = [[fx, 0.0, cx, 0.0], [0.0, fx, cy, 0.0], [0.0, 0.0, 1.0, 0.0]]
    p2 = [row[:] for row in p1]
    p2[0][3] = -fx * baseline
    q = [
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, fx],
        [0.0, 0.0, 1.0 / baseline, 0.0],
    ]
    return calibration_from_mapping(
        {
            "valid": True,
            "images_are_rectified": True,
            "physical_left_view": "top",
            "physical_right_view": "bottom",
            "runtime_rotation": "ccw90",
            "image_size": [width, height],
            "baseline_m": baseline,
            "left": {
                "camera_matrix": camera,
                "distortion_coefficients": [],
                "projection_matrix": p1,
            },
            "right": {
                "camera_matrix": camera,
                "distortion_coefficients": [],
                "projection_matrix": p2,
            },
            "rotation": np.eye(3).tolist(),
            "translation_m": [-baseline, 0.0, 0.0],
            "rectification": {
                "left_matrix": np.eye(3).tolist(),
                "right_matrix": np.eye(3).tolist(),
                "left_projection": p1,
                "right_projection": p2,
                "q_matrix": q,
            },
        }
    )


def test_gs130w_vertical_split_uses_explicit_physical_mapping() -> None:
    top = np.full((20, 30, 3), 11, dtype=np.uint8)
    bottom = np.full((20, 30, 3), 22, dtype=np.uint8)
    pair = split_gs130w_vertical(
        np.vstack([top, bottom]),
        physical_left_view="bottom",
        physical_right_view="top",
    )
    assert int(pair.left_bgr[0, 0, 0]) == 22
    assert int(pair.right_bgr[0, 0, 0]) == 11


def test_invalid_calibration_returns_unavailable_without_raising() -> None:
    estimator = StereoDepthEstimator.from_file("does-not-exist.yaml")
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    result = estimator.estimate(image, image, (2, 2, 10, 10), 123.0)
    assert not result.valid
    assert result.distance_m is None
    assert result.reason == "calibration_file_not_found"


def test_malformed_calibration_returns_unavailable_without_raising() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "broken.yaml"
        path.write_text("calibration: [unterminated", encoding="utf-8")
        estimator = StereoDepthEstimator.from_file(path)
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    result = estimator.estimate(image, image, (2, 2, 10, 10), 123.0)
    assert not result.valid
    assert result.reason.startswith("calibration_load_failed:")


def test_fixed_disparity_estimates_camera_to_fire_euclidean_distance() -> None:
    calibration = _calibration()
    assert calibration.valid
    estimator = StereoDepthEstimator(calibration, min_valid_pixels=20)
    estimator._matcher_factory = lambda _width: _FixedMatcher((120, 160), 8.0)
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    result = estimator.estimate(image, image, (60, 40, 100, 80), 7.5)

    assert result.valid, result
    assert result.reason == "ok"
    assert result.distance_m is not None
    # Z = fx * baseline / disparity = 200 * 0.1 / 8 = 2.5m;
    # the requested contract is Euclidean range, not merely optical-axis Z.
    assert abs(result.distance_m - 2.5) < 0.04
    assert result.quality["distance_definition"] == "camera_to_fire_region_euclidean_m"


def test_too_few_valid_pixels_returns_unavailable() -> None:
    estimator = StereoDepthEstimator(_calibration(), min_valid_pixels=100)
    disparity = np.full((120, 160), -1.0, dtype=np.float32)
    disparity[55:60, 75:80] = 8.0

    class SparseMatcher:
        def compute(self, _left: np.ndarray, _right: np.ndarray) -> np.ndarray:
            return np.rint(disparity * 16.0).astype(np.int16)

    estimator._matcher_factory = lambda _width: SparseMatcher()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = estimator.estimate(image, image, (50, 30, 110, 90), 9.0)

    assert not result.valid
    assert result.reason == "insufficient_valid_stereo_pixels"


def test_from_file_loads_complete_json_compatible_calibration() -> None:
    calibration = _calibration()
    payload = {
        "physical_left_view": "top",
        "physical_right_view": "bottom",
        "runtime_rotation": "ccw90",
        "matcher": {
            "min_disparity": 2,
            "num_disparities": 96,
            "block_size": 7,
            "uniqueness_ratio": 17,
            "speckle_window_size": 44,
            "speckle_range": 3,
            "minimum_valid_pixels": 37,
            "min_distance_m": 0.8,
            "max_distance_m": 12.5,
            "max_relative_mad": 0.22,
        },
        "calibration": {
            "valid": True,
            "images_are_rectified": True,
            "image_size": list(calibration.image_size),
            "baseline_m": calibration.baseline_m,
            "left": {
                "camera_matrix": calibration.left_camera_matrix.tolist(),
                "distortion_coefficients": [],
                "projection_matrix": calibration.left_projection.tolist(),
            },
            "right": {
                "camera_matrix": calibration.right_camera_matrix.tolist(),
                "distortion_coefficients": [],
                "projection_matrix": calibration.right_projection.tolist(),
            },
            "rotation": calibration.rotation.tolist(),
            "translation_m": calibration.translation_m.tolist(),
            "rectification": {
                "left_matrix": calibration.left_rectification.tolist(),
                "right_matrix": calibration.right_rectification.tolist(),
                "left_projection": calibration.left_projection.tolist(),
                "right_projection": calibration.right_projection.tolist(),
                "q_matrix": calibration.q_matrix.tolist(),
            },
        }
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stereo.yaml"
        path.write_text(json.dumps(payload), encoding="utf-8")
        estimator = StereoDepthEstimator.from_file(path)
    assert estimator.calibration.valid
    assert estimator.min_valid_pixels == 37
    assert estimator.min_distance_m == 0.8
    assert estimator.max_distance_m == 12.5
    assert estimator.matcher_min_disparity == 2
    assert estimator.num_disparities == 96
    assert estimator.block_size == 7
    assert estimator.uniqueness_ratio == 17
    assert estimator.speckle_window_size == 44
    assert estimator.speckle_range == 3
    assert estimator.validate_view_mapping("top", "bottom") == (True, "ok")
    assert estimator.validate_view_mapping("bottom", "top") == (
        False,
        "detection_view_is_not_physical_left",
    )
