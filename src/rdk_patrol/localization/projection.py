from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .apriltag import TagObservation


@dataclass(frozen=True)
class RoiProjection:
    rois: Mapping[str, tuple[tuple[float, float], ...]]
    applied: bool
    method: str
    reason: str


def normalized_rois(point: Mapping[str, Any]) -> dict[str, tuple[tuple[float, float], ...]]:
    raw = point.get("rois", {})
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = (
            (
                str(item.get("id") or item.get("name") or f"roi_{index + 1}"),
                item.get("polygon", []),
            )
            for index, item in enumerate(raw)
            if isinstance(item, Mapping)
        )
    else:
        return result
    for name, polygon in items:
        if isinstance(polygon, Mapping):
            polygon = polygon.get("polygon", [])
        try:
            points = tuple((float(value[0]), float(value[1])) for value in polygon)
        except (TypeError, ValueError, IndexError):
            continue
        if len(points) >= 3:
            result[str(name)] = points
    return result


def project_rois_with_homography(
    rois: Mapping[str, Sequence[Sequence[float]]],
    homography: Sequence[Sequence[float]],
) -> RoiProjection:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return RoiProjection({}, False, "reference_homography", "invalid_homography")
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, polygon in rois.items():
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        if len(transformed) < 3 or not np.isfinite(transformed).all():
            return RoiProjection({}, False, "reference_homography", "invalid_projected_roi")
        result[str(name)] = tuple((float(x), float(y)) for x, y in transformed)
    return RoiProjection(result, True, "reference_homography", "ok")


def project_point_rois_from_tag(
    point: Mapping[str, Any],
    observation: TagObservation,
    *,
    camera_matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
) -> RoiProjection:
    """Project point ROIs using 3-D tag coordinates or a planar reference.

    A homography is only valid when the ROI vertices and reference tag lie on
    the same physical plane.  Ground ROIs with a wall-mounted tag are
    non-coplanar and must use ``roi_points_tag_m`` plus calibrated camera
    intrinsics; the function deliberately refuses the planar shortcut unless
    ``rois_coplanar_with_tag`` is explicitly true.
    """

    world_rois = point.get("roi_points_tag_m")
    tag_size = float(point.get("tag_size_m") or 0.0)
    if isinstance(world_rois, Mapping) and world_rois:
        if camera_matrix is None or tag_size <= 0:
            return RoiProjection({}, False, "tag_3d", "tag_3d_requires_intrinsics_and_tag_size")
        return _project_3d(
            world_rois,
            observation,
            tag_size,
            np.asarray(camera_matrix, dtype=np.float64),
            np.zeros(5, dtype=np.float64)
            if distortion is None
            else np.asarray(distortion, dtype=np.float64),
        )

    reference = point.get("reference_tag_corners")
    rois = normalized_rois(point)
    if reference is None or not rois:
        return RoiProjection(rois, False, "static", "projection_configuration_missing")
    if not bool(point.get("rois_coplanar_with_tag", False)):
        return RoiProjection(
            {},
            False,
            "reference_homography",
            "non_coplanar_roi_requires_tag_3d_coordinates",
        )
    source = np.asarray(reference, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(observation.corners, dtype=np.float32).reshape(-1, 2)
    if source.shape != (4, 2):
        return RoiProjection({}, False, "reference_homography", "invalid_reference_tag_corners")
    matrix = cv2.getPerspectiveTransform(source, target)
    return project_rois_with_homography(rois, matrix)


def _project_3d(
    world_rois: Mapping[str, Any],
    observation: TagObservation,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> RoiProjection:
    if camera_matrix.shape != (3, 3):
        return RoiProjection({}, False, "tag_3d", "invalid_camera_matrix")
    half = tag_size_m * 0.5
    tag_points = np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    image_points = np.asarray(observation.corners, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        tag_points,
        image_points,
        camera_matrix,
        distortion,
        # Tag frame: +X right, +Y up and +Z out of the printed front face.
        # ITERATIVE honours the detector's TL/TR/BR/BL correspondence; some
        # OpenCV IPPE_SQUARE versions silently choose the mirrored ordering.
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if (
        not ok
        or not np.isfinite(rvec).all()
        or not np.isfinite(tvec).all()
        or float(np.asarray(tvec).reshape(-1)[2]) <= 0
    ):
        return RoiProjection({}, False, "tag_3d", "tag_pose_estimation_failed")
    tag_reprojected, _jacobian = cv2.projectPoints(
        tag_points, rvec, tvec, camera_matrix, distortion
    )
    tag_error = np.linalg.norm(
        tag_reprojected.reshape(-1, 2) - image_points, axis=1
    )
    if float(np.median(tag_error)) > 4.0:
        return RoiProjection({}, False, "tag_3d", "tag_pose_reprojection_error")
    rotation, _jacobian = cv2.Rodrigues(rvec)
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, vertices in world_rois.items():
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
            return RoiProjection({}, False, "tag_3d", "invalid_roi_points_tag_m")
        camera_points = (rotation @ points.T + np.asarray(tvec).reshape(3, 1)).T
        if np.any(camera_points[:, 2] <= 0.05):
            return RoiProjection({}, False, "tag_3d", "roi_behind_camera")
        image, _jacobian = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
        image = image.reshape(-1, 2)
        if (
            not np.isfinite(image).all()
            or abs(float(cv2.contourArea(image.astype(np.float32)))) < 10.0
        ):
            return RoiProjection({}, False, "tag_3d", "invalid_projected_roi")
        result[str(name)] = tuple((float(x), float(y)) for x, y in image)
    return RoiProjection(result, True, "tag_3d", "ok")
