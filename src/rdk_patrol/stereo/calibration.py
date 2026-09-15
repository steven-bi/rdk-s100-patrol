from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    valid: bool
    reason: str
    image_size: tuple[int, int] | None = None
    left_camera_matrix: np.ndarray | None = None
    left_distortion: np.ndarray | None = None
    right_camera_matrix: np.ndarray | None = None
    right_distortion: np.ndarray | None = None
    rotation: np.ndarray | None = None
    translation_m: np.ndarray | None = None
    left_rectification: np.ndarray | None = None
    right_rectification: np.ndarray | None = None
    left_projection: np.ndarray | None = None
    right_projection: np.ndarray | None = None
    q_matrix: np.ndarray | None = None
    baseline_m: float | None = None
    images_are_rectified: bool = False
    quality: Mapping[str, Any] = field(default_factory=dict)
    physical_left_view: str = ""
    physical_right_view: str = ""
    runtime_rotation: str = "none"

    @classmethod
    def invalid(cls, reason: str) -> "StereoCalibration":
        return cls(valid=False, reason=str(reason))


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    calibration, _matcher = load_stereo_config(path)
    return calibration


def load_stereo_config(
    path: str | Path,
) -> tuple[StereoCalibration, dict[str, Any]]:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return StereoCalibration.invalid("calibration_file_not_found"), {}
    try:
        text = candidate.read_text(encoding="utf-8-sig")
        try:
            import yaml

            raw = yaml.safe_load(text)
        except ImportError:
            raw = json.loads(text)
        if not isinstance(raw, Mapping):
            return StereoCalibration.invalid("calibration_root_not_mapping"), {}
        nested = raw.get("calibration")
        payload = dict(nested) if isinstance(nested, Mapping) else dict(raw)
        # Tools may wrap matrices in ``calibration`` while operational config
        # keeps view mapping and matcher settings at the document root.
        for key in (
            "physical_left_view",
            "physical_right_view",
            "runtime_rotation",
            "view_mapping",
        ):
            if key not in payload and key in raw:
                payload[key] = raw[key]
        if not isinstance(payload, Mapping):
            return StereoCalibration.invalid("calibration_section_missing"), {}
        matcher = raw.get("matcher", payload.get("matcher", {}))
        return (
            calibration_from_mapping(payload),
            dict(matcher) if isinstance(matcher, Mapping) else {},
        )
    except Exception as exc:
        return (
            StereoCalibration.invalid(
                f"calibration_load_failed:{type(exc).__name__}"
            ),
            {},
        )


def calibration_from_mapping(payload: Mapping[str, Any]) -> StereoCalibration:
    if not bool(payload.get("valid", payload.get("calibration_valid", False))):
        return StereoCalibration.invalid("calibration_not_validated")
    try:
        size_values = payload["image_size"]
        image_size = (int(size_values[0]), int(size_values[1]))
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("invalid image size")
        left = payload["left"]
        right = payload["right"]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError("camera sections must be mappings")
        left_k = _matrix(left.get("camera_matrix"), (3, 3))
        right_k = _matrix(right.get("camera_matrix"), (3, 3))
        left_d = _distortion(left.get("distortion_coefficients", []))
        right_d = _distortion(right.get("distortion_coefficients", []))
        rotation = _matrix(
            payload.get("rotation", payload.get("rotation_matrix")), (3, 3)
        )
        translation = np.asarray(
            payload.get("translation_m", payload.get("translation")), dtype=np.float64
        ).reshape(-1)
        if translation.size != 3 or not np.isfinite(translation).all():
            raise ValueError("invalid stereo translation")
        baseline = float(payload.get("baseline_m") or np.linalg.norm(translation))
        if not np.isfinite(baseline) or baseline <= 1e-4 or baseline >= 2.0:
            raise ValueError("implausible stereo baseline")
        if abs(np.linalg.norm(translation) - baseline) > max(0.005, baseline * 0.1):
            raise ValueError("baseline does not match translation")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 0.1:
            raise ValueError("rotation matrix is not orthonormal")

        rectification = payload.get("rectification", {})
        if not isinstance(rectification, Mapping):
            rectification = {}
        r1 = _optional_matrix(
            rectification.get("left_matrix", left.get("rectification_matrix")), (3, 3)
        )
        r2 = _optional_matrix(
            rectification.get("right_matrix", right.get("rectification_matrix")), (3, 3)
        )
        p1 = _optional_matrix(
            rectification.get("left_projection", left.get("projection_matrix")), (3, 4)
        )
        p2 = _optional_matrix(
            rectification.get("right_projection", right.get("projection_matrix")), (3, 4)
        )
        q = _optional_matrix(
            rectification.get(
                "q_matrix",
                rectification.get("q", payload.get("q_matrix", payload.get("q"))),
            ),
            (4, 4),
        )
        images_are_rectified = bool(payload.get("images_are_rectified", False))
        if images_are_rectified and (p1 is None or p2 is None):
            raise ValueError("rectified stream requires projection matrices")
        if r1 is None or r2 is None or p1 is None or p2 is None or q is None:
            r1, r2, p1, p2, q, _roi1, _roi2 = cv2.stereoRectify(
                left_k,
                left_d,
                right_k,
                right_d,
                image_size,
                rotation,
                translation.reshape(3, 1),
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )
        quality = (
            dict(payload.get("quality", {}))
            if isinstance(payload.get("quality"), Mapping)
            else {}
        )
        view_mapping = payload.get("view_mapping", {})
        if not isinstance(view_mapping, Mapping):
            view_mapping = {}
        physical_left = str(
            payload.get("physical_left_view")
            or view_mapping.get("physical_left_view")
            or ""
        )
        physical_right = str(
            payload.get("physical_right_view")
            or view_mapping.get("physical_right_view")
            or ""
        )
        if physical_left not in {"top", "bottom"}:
            physical_left = ""
        if physical_right not in {"top", "bottom"} or physical_right == physical_left:
            physical_right = ""
        return StereoCalibration(
            valid=True,
            reason="ok",
            image_size=image_size,
            left_camera_matrix=left_k,
            left_distortion=left_d,
            right_camera_matrix=right_k,
            right_distortion=right_d,
            rotation=rotation,
            translation_m=translation,
            left_rectification=r1,
            right_rectification=r2,
            left_projection=p1,
            right_projection=p2,
            q_matrix=q,
            baseline_m=baseline,
            images_are_rectified=images_are_rectified,
            quality=quality,
            physical_left_view=physical_left,
            physical_right_view=physical_right,
            runtime_rotation=str(payload.get("runtime_rotation") or "none"),
        )
    except (KeyError, TypeError, ValueError, cv2.error):
        return StereoCalibration.invalid("calibration_parameters_invalid")


def _matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"expected finite matrix {shape}")
    return matrix


def _optional_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    return _matrix(value, shape)


def _distortion(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(result).all():
        raise ValueError("invalid distortion coefficients")
    return result
