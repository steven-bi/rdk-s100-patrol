from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FingerprintMatch:
    point_id: str
    point_name: str
    score: float
    margin: float
    homography: tuple[tuple[float, float, float], ...] | None
    geometry_verified: bool
    diagnostics: Mapping[str, Any]


def frame_fingerprint(image: np.ndarray, max_orb_descriptors: int = 192) -> dict[str, Any]:
    if image is None or image.size == 0:
        raise ValueError("cannot fingerprint an empty image")
    height, width = image.shape[:2]
    scale = min(1.0, 480.0 / max(height, width))
    resized = (
        image.copy()
        if scale >= 0.999
        else cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).reshape(-1)
    hist = hist / max(float(hist.sum()), 1.0)
    orb = cv2.ORB_create(
        nfeatures=max(32, int(max_orb_descriptors)),
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=9,
    )
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
        keypoints = []
    return {
        "schema_version": "rdk-patrol-visual-fingerprint/v1",
        "method": "phash_hsv_orb_ransac",
        "frame_width": int(width),
        "frame_height": int(height),
        "descriptor_scale": float(scale),
        "phash": _phash(gray),
        "hsv_hist": [round(float(value), 7) for value in hist.tolist()],
        "orb": {
            "count": int(len(descriptors)),
            "descriptors": [bytes(row.tolist()).hex() for row in descriptors],
            "points": [
                [
                    round(float(point.pt[0]) / max(1, resized.shape[1]), 7),
                    round(float(point.pt[1]) / max(1, resized.shape[0]), 7),
                ]
                for point in keypoints
            ],
        },
    }


class VisualFingerprintMatcher:
    def __init__(
        self,
        points: Sequence[Mapping[str, Any]],
        *,
        min_score: float = 0.58,
        min_margin: float = 0.05,
        geometry_required: bool = True,
        min_matches: int = 8,
        min_inliers: int = 6,
        min_inlier_ratio: float = 0.35,
        max_reprojection_error: float = 8.0,
    ) -> None:
        self.points = list(points)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.geometry_required = bool(geometry_required)
        self.min_matches = max(4, int(min_matches))
        self.min_inliers = max(4, int(min_inliers))
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_error = float(max_reprojection_error)

    @property
    def has_anchors(self) -> bool:
        return any(_anchors_for_point(point) for point in self.points)

    def match(self, image: np.ndarray) -> FingerprintMatch | None:
        if not self.has_anchors or not _image_quality_ok(image):
            return None
        current = frame_fingerprint(image)
        candidates: list[tuple[float, Mapping[str, Any], Mapping[str, Any], dict[str, Any]]] = []
        for point in self.points:
            for anchor in _anchors_for_point(point):
                score, geometry = self._score_and_geometry(current, anchor)
                candidates.append((score, point, anchor, geometry))
        if not candidates:
            return None
        candidates.sort(key=lambda value: value[0], reverse=True)
        score, point, anchor, geometry = candidates[0]
        point_id = _point_id(point)
        other_scores = [
            item[0] for item in candidates if _point_id(item[1]) != point_id
        ]
        margin = score - max([0.0, *other_scores])
        required_score = float(point.get("fingerprint_min_score", self.min_score))
        required_margin = float(point.get("fingerprint_min_margin", self.min_margin))
        require_geometry = bool(
            point.get("fingerprint_geometry_required", self.geometry_required)
        )
        if score < required_score or margin < required_margin:
            return None
        if require_geometry and not bool(geometry["verified"]):
            return None
        homography = geometry.get("homography")
        return FingerprintMatch(
            point_id=point_id,
            point_name=_point_name(point),
            score=float(score),
            margin=float(margin),
            homography=(
                None
                if homography is None
                else tuple(tuple(float(value) for value in row) for row in homography)
            ),
            geometry_verified=bool(geometry["verified"]),
            diagnostics={
                "anchor_image": str(anchor.get("keyframe_image", "")),
                "geometry_matches": int(geometry["matches"]),
                "geometry_inliers": int(geometry["inliers"]),
                "geometry_inlier_ratio": float(geometry["inlier_ratio"]),
                "geometry_reprojection_error": geometry["reprojection_error"],
            },
        )

    def _score_and_geometry(
        self, current: Mapping[str, Any], anchor: Mapping[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        phash = _phash_score(str(current.get("phash", "")), str(anchor.get("phash", "")))
        hist = _hist_score(current.get("hsv_hist"), anchor.get("hsv_hist"))
        geometry = _orb_geometry(
            anchor,
            current,
            min_matches=self.min_matches,
            min_inliers=self.min_inliers,
            min_inlier_ratio=self.min_inlier_ratio,
            max_reprojection_error=self.max_reprojection_error,
        )
        orb_similarity = min(1.0, float(geometry["matches"]) / max(12.0, self.min_matches * 2.0))
        score = 0.45 * orb_similarity + 0.30 * phash + 0.25 * hist
        if int(geometry["matches"]) == 0:
            score = 0.55 * phash + 0.45 * hist
        return float(score), geometry


def _anchors_for_point(point: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for key in ("visual_fingerprints", "fingerprints", "anchors", "scene_anchors"):
        value = point.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result.extend(item for item in value if isinstance(item, Mapping))
    single = point.get("scene_anchor")
    if isinstance(single, Mapping):
        result.append(single)
    rois = point.get("rois")
    if isinstance(rois, Sequence) and not isinstance(rois, (str, bytes)):
        for roi in rois:
            if not isinstance(roi, Mapping):
                continue
            anchor = roi.get("scene_anchor")
            if isinstance(anchor, Mapping):
                result.append(anchor)
            extra = roi.get("scene_anchors")
            if isinstance(extra, Sequence) and not isinstance(extra, (str, bytes)):
                result.extend(item for item in extra if isinstance(item, Mapping))
    return result


def _point_id(point: Mapping[str, Any]) -> str:
    return str(point.get("point_id") or point.get("id") or "")


def _point_name(point: Mapping[str, Any]) -> str:
    return str(
        point.get("point_name")
        or point.get("name")
        or point.get("location")
        or _point_id(point)
    )


def _image_quality_ok(image: np.ndarray) -> bool:
    if image is None or image.size == 0:
        return False
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 3) | (gray >= 252)))
    return 8.0 <= mean <= 247.0 and blur >= 8.0 and clipped <= 0.92


def _phash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    value = 0
    for bit in (low > threshold).tolist():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _phash_score(left: str, right: str) -> float:
    if len(left) != 16 or len(right) != 16:
        return 0.0
    try:
        distance = (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 0.0
    except AttributeError:
        distance = bin(int(left, 16) ^ int(right, 16)).count("1")
    return max(0.0, 1.0 - distance / 64.0)


def _hist_score(left: Any, right: Any) -> float:
    try:
        a = np.asarray(left, dtype=np.float32).reshape(-1)
        b = np.asarray(right, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if a.size != 64 or b.size != 64 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    return float(np.clip(cv2.compareHist(a, b, cv2.HISTCMP_CORREL), 0.0, 1.0))


def _decode_orb(payload: Mapping[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    raw = payload.get("orb")
    if not isinstance(raw, Mapping):
        return None, None
    try:
        points = np.asarray(raw.get("points", []), dtype=np.float32)
    except (TypeError, ValueError):
        return None, None
    raw_descriptors = raw.get("descriptors", [])
    descriptors: np.ndarray
    if (
        isinstance(raw_descriptors, Sequence)
        and raw_descriptors
        and isinstance(raw_descriptors[0], str)
    ):
        rows = []
        for value in raw_descriptors:
            try:
                row = np.frombuffer(bytes.fromhex(str(value)), dtype=np.uint8)
            except ValueError:
                continue
            if row.size == 32:
                rows.append(row)
        descriptors = (
            np.stack(rows).astype(np.uint8)
            if rows
            else np.empty((0, 32), dtype=np.uint8)
        )
    else:
        try:
            descriptors = np.asarray(raw_descriptors, dtype=np.uint8)
        except (TypeError, ValueError):
            return None, None
    if descriptors.ndim != 2 or descriptors.shape[1:] != (32,):
        return None, None
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) != len(descriptors):
        return None, None
    # Legacy vehicle/trash anchors serialize keypoints normalized to [0, 1].
    # New registrations use the same representation.  Numeric absolute points
    # from early prototypes remain accepted for backward compatibility.
    if points.size and float(np.max(np.abs(points))) <= 1.5:
        width = float(payload.get("frame_width") or 0.0)
        height = float(payload.get("frame_height") or 0.0)
        if width <= 0 or height <= 0:
            return descriptors, None
        points = points * np.asarray([width, height], dtype=np.float32)
    return descriptors, points


def _orb_geometry(
    anchor: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    min_matches: int,
    min_inliers: int,
    min_inlier_ratio: float,
    max_reprojection_error: float,
) -> dict[str, Any]:
    anchor_desc, anchor_points = _decode_orb(anchor)
    current_desc, current_points = _decode_orb(current)
    failed = {
        "verified": False,
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reprojection_error": None,
        "homography": None,
    }
    if anchor_desc is None or current_desc is None or len(anchor_desc) < 2 or len(current_desc) < 2:
        return failed
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(anchor_desc, current_desc, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance <= 64 and pair[0].distance < 0.78 * pair[1].distance
    ]
    failed["matches"] = len(good)
    if len(good) < min_matches:
        return failed
    source = np.float32([anchor_points[item.queryIdx] for item in good]).reshape(-1, 1, 2)
    target = np.float32([current_points[item.trainIdx] for item in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return failed
    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(inlier_mask.sum())
    ratio = inliers / max(1, len(good))
    projected = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - target.reshape(-1, 2), axis=1)
    reprojection = float(np.median(errors[inlier_mask])) if inliers else float("inf")
    verified = (
        inliers >= min_inliers
        and ratio >= min_inlier_ratio
        and reprojection <= max_reprojection_error
        and _stable_homography(matrix)
    )
    return {
        "verified": verified,
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(ratio),
        "reprojection_error": reprojection,
        "homography": matrix.tolist() if verified else None,
    }


def _stable_homography(matrix: np.ndarray) -> bool:
    normalized = matrix / max(abs(float(matrix[2, 2])), 1e-9)
    affine_det = float(np.linalg.det(normalized[:2, :2]))
    perspective = float(np.linalg.norm(normalized[2, :2]))
    return 0.08 <= abs(affine_det) <= 12.0 and perspective <= 0.03
