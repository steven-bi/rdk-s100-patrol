from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from rdk_patrol.contracts import PointContext

from .apriltag import AprilTagDetector, TagObservation
from .fingerprint import FingerprintMatch, VisualFingerprintMatcher
from .projection import (
    normalized_rois,
    project_point_rois_from_tag,
    project_rois_with_homography,
)


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml

        value = yaml.safe_load(text)
    except ImportError:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


class PointResolver:
    def __init__(
        self,
        points: Sequence[Mapping[str, Any]],
        *,
        tag_family: str = "tag36h11",
        required_stable_frames: int = 3,
        fingerprint_options: Mapping[str, Any] | None = None,
        camera_matrix: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
    ) -> None:
        normalized: list[dict[str, Any]] = []
        for raw in points:
            if not isinstance(raw, Mapping) or not bool(raw.get("enabled", True)):
                continue
            point = dict(raw)
            point_id = str(point.get("point_id") or point.get("id") or "")
            point_name = str(
                point.get("point_name")
                or point.get("name")
                or point.get("location")
                or point_id
            )
            if not point_id or not point_name:
                continue
            point["point_id"] = point_id
            point["point_name"] = point_name
            normalized.append(point)
        self.points = normalized
        self.by_id = {str(point["point_id"]): point for point in self.points}
        self.by_tag = {
            int(point["tag_id"]): point
            for point in self.points
            if point.get("tag_id") is not None
        }
        self.required_stable_frames = max(1, int(required_stable_frames))
        self.detector = AprilTagDetector(self.by_tag, tag_family=tag_family)
        self.fingerprint_matcher = VisualFingerprintMatcher(
            self.points, **dict(fingerprint_options or {})
        )
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self._tag_candidate: int | None = None
        self._tag_count = 0
        self._fingerprint_candidate: str | None = None
        self._fingerprint_count = 0
        self.last_diagnostics: dict[str, Any] = {}

    @classmethod
    def from_file(
        cls,
        points_path: str | Path,
        tag_family: str = "tag36h11",
        required_stable_frames: int = 3,
        camera_matrix: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
    ) -> "PointResolver":
        path = Path(points_path).expanduser().resolve()
        raw = _load_structured(path)
        points = []
        for item in raw.get("points", []):
            if not isinstance(item, Mapping):
                continue
            point = dict(item)
            _normalize_point_schema(point)
            points.append(point)
        _merge_per_point_legacy(points, path.parent)
        _merge_legacy_points(points, raw, path.parent)
        fingerprint_options = (
            dict(raw.get("fingerprint", {}))
            if isinstance(raw.get("fingerprint"), Mapping)
            else {}
        )
        camera = raw.get("camera")
        if isinstance(camera, Mapping):
            try:
                if camera_matrix is None:
                    camera_matrix = np.asarray(
                        camera.get("camera_matrix"), dtype=np.float64
                    )
                    if camera_matrix.shape != (3, 3):
                        camera_matrix = None
                if distortion is None:
                    distortion = np.asarray(
                        camera.get("distortion_coefficients", []), dtype=np.float64
                    ).reshape(-1)
            except (TypeError, ValueError):
                pass
        return cls(
            points,
            tag_family=tag_family,
            required_stable_frames=required_stable_frames,
            fingerprint_options=fingerprint_options,
            camera_matrix=camera_matrix,
            distortion=distortion,
        )

    def resolve(
        self, frame_bgr: np.ndarray, observed_at: float
    ) -> PointContext | None:
        tag_observations = self.detector.detect(frame_bgr)
        fingerprint = self.fingerprint_matcher.match(frame_bgr)
        observed_tag_ids = {item.tag_id for item in tag_observations}
        tag_ambiguous = len(observed_tag_ids) > 1
        tag = (
            tag_observations[0]
            if tag_observations and not tag_ambiguous
            else None
        )
        conflict = bool(
            tag is not None
            and fingerprint is not None
            and str(self.by_tag[tag.tag_id]["point_id"]) != fingerprint.point_id
        )
        self.last_diagnostics = {
            "tag_backend_available": self.detector.available,
            "tag_backend_reason": self.detector.unavailable_reason,
            "accepted_tag_ids": [item.tag_id for item in tag_observations],
            "configured_tags_ambiguous": tag_ambiguous,
            "fingerprint_candidate": None if fingerprint is None else fingerprint.point_id,
            "tag_fingerprint_conflict": conflict,
        }

        if tag_ambiguous:
            self._tag_candidate = None
            self._tag_count = 0
            self._reset_fingerprint()
            return None

        # Any accepted configured tag suppresses fingerprint fallback while its
        # consecutive-frame gate is pending.  Once stable, tag always wins.
        if tag is not None:
            self._reset_fingerprint()
            if tag.tag_id == self._tag_candidate:
                self._tag_count += 1
            else:
                self._tag_candidate = tag.tag_id
                self._tag_count = 1
            self.last_diagnostics["tag_stable_count"] = self._tag_count
            if self._tag_count < self.required_stable_frames:
                return None
            return self._context_from_tag(
                tag, observed_at, conflict, fingerprint=fingerprint
            )

        self._tag_candidate = None
        self._tag_count = 0
        if fingerprint is None:
            self._reset_fingerprint()
            return None
        if fingerprint.point_id == self._fingerprint_candidate:
            self._fingerprint_count += 1
        else:
            self._fingerprint_candidate = fingerprint.point_id
            self._fingerprint_count = 1
        self.last_diagnostics["fingerprint_stable_count"] = self._fingerprint_count
        if self._fingerprint_count < self.required_stable_frames:
            return None
        return self._context_from_fingerprint(fingerprint, observed_at)

    def _context_from_tag(
        self,
        tag: TagObservation,
        observed_at: float,
        conflict: bool,
        *,
        fingerprint: FingerprintMatch | None,
    ) -> PointContext:
        point = self.by_tag[tag.tag_id]
        registration_required = bool(point.get("registration_required", False))
        if registration_required:
            legacy_rois = normalized_rois(point)
            if (
                fingerprint is not None
                and fingerprint.point_id == str(point["point_id"])
                and fingerprint.homography is not None
            ):
                projection = project_rois_with_homography(
                    legacy_rois, fingerprint.homography
                )
            else:
                from .projection import RoiProjection

                projection = RoiProjection(
                    {},
                    False,
                    "legacy_orb_ransac",
                    "tag_registration_required_and_legacy_geometry_unavailable",
                )
        else:
            projection = project_point_rois_from_tag(
                point,
                tag,
                camera_matrix=self.camera_matrix,
                distortion=self.distortion,
            )
        diagnostics = {
            **self.last_diagnostics,
            "tag_area_px": tag.area_px,
            "tag_corners": [list(value) for value in tag.corners],
            "projection_method": projection.method,
            "projection_applied": projection.applied,
            "projection_reason": projection.reason,
            "tag_wins_conflict": conflict,
            "tag_registration_required": registration_required,
        }
        safe_rois = projection.rois if projection.applied else {}
        return PointContext(
            point_id=str(point["point_id"]),
            point_name=str(point["point_name"]),
            source="apriltag",
            confidence=float(tag.confidence),
            observed_at=float(observed_at),
            tag_id=int(tag.tag_id),
            rois=safe_rois,
            diagnostics=diagnostics,
        )

    def _context_from_fingerprint(
        self, match: FingerprintMatch, observed_at: float
    ) -> PointContext:
        point = self.by_id[match.point_id]
        rois = normalized_rois(point)
        projection_reason = "static"
        if match.homography is not None:
            projected = project_rois_with_homography(rois, match.homography)
            if projected.applied:
                rois = dict(projected.rois)
                projection_reason = "orb_ransac_homography"
        diagnostics = {
            **self.last_diagnostics,
            **dict(match.diagnostics),
            "projection_method": projection_reason,
            "geometry_verified": match.geometry_verified,
            "fingerprint_margin": match.margin,
        }
        return PointContext(
            point_id=match.point_id,
            point_name=match.point_name,
            source="visual_fingerprint",
            confidence=float(match.score),
            observed_at=float(observed_at),
            tag_id=None,
            rois=rois,
            diagnostics=diagnostics,
        )

    def _reset_fingerprint(self) -> None:
        self._fingerprint_candidate = None
        self._fingerprint_count = 0


def _merge_legacy_points(
    points: list[dict[str, Any]], root: Mapping[str, Any], base_dir: Path
) -> None:
    """Merge old vehicle/trash point files without making them mandatory."""

    references: list[str] = []
    for key in (
        "legacy_visual_fingerprint_paths",
        "legacy_point_files",
        "legacy_vehicle_points_path",
        "legacy_trash_points_path",
    ):
        value = root.get(key)
        if isinstance(value, str):
            references.append(value)
        elif isinstance(value, Sequence):
            references.extend(str(item) for item in value)
    by_id = {
        str(point.get("point_id") or point.get("id") or ""): point
        for point in points
    }
    for reference in references:
        candidate = Path(reference).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if not candidate.is_file():
            continue
        legacy = _load_structured(candidate)
        for item in legacy.get("points", []):
            if not isinstance(item, Mapping):
                continue
            point_id = str(item.get("point_id") or item.get("id") or "")
            if not point_id:
                continue
            target = by_id.get(point_id)
            if target is None:
                target = dict(item)
                target["point_id"] = point_id
                target["point_name"] = str(
                    item.get("point_name")
                    or item.get("name")
                    or item.get("location")
                    or point_id
                )
                points.append(target)
                by_id[point_id] = target
            anchors = list(target.get("visual_fingerprints", []))
            for anchor in _legacy_anchors(item):
                if anchor not in anchors:
                    anchors.append(anchor)
            if anchors:
                target["visual_fingerprints"] = anchors


def _legacy_anchors(point: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for key in ("anchors", "scene_anchors", "visual_fingerprints"):
        raw = point.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.extend(value for value in raw if isinstance(value, Mapping))
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
                result.extend(value for value in extra if isinstance(value, Mapping))
    return result


def _normalize_point_schema(point: dict[str, Any]) -> None:
    tag = point.get("tag")
    if isinstance(tag, Mapping):
        if tag.get("id") is not None:
            point["tag_id"] = int(tag["id"])
        if tag.get("size_m") is not None:
            point["tag_size_m"] = float(tag["size_m"])
        point["tag_family"] = str(tag.get("family") or "tag36h11")
        point["registration_required"] = bool(
            tag.get("registration_required", False)
        )
    rois_3d = point.get("rois_3d")
    if isinstance(rois_3d, Sequence) and not isinstance(rois_3d, (str, bytes)):
        mapped: dict[str, Any] = {}
        for index, roi in enumerate(rois_3d):
            if not isinstance(roi, Mapping):
                continue
            name = str(roi.get("id") or roi.get("name") or f"roi_{index + 1}")
            vertices = (
                roi.get("points_tag_m")
                or roi.get("vertices_tag_m")
                or roi.get("polygon_tag_m")
            )
            if vertices:
                mapped[name] = vertices
        if mapped:
            point["roi_points_tag_m"] = mapped


def _merge_per_point_legacy(
    points: list[dict[str, Any]], base_dir: Path
) -> None:
    for point in points:
        legacy = point.get("legacy")
        if not isinstance(legacy, Mapping):
            continue
        legacy_id = str(legacy.get("legacy_point_id") or point.get("point_id") or "")
        references = [
            legacy.get("vehicle_config"),
            legacy.get("trash_config"),
        ]
        for reference in references:
            if not reference:
                continue
            path = Path(str(reference)).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            if not path.is_file():
                continue
            source = _load_structured(path)
            legacy_point = next(
                (
                    item
                    for item in source.get("points", [])
                    if isinstance(item, Mapping)
                    and str(item.get("point_id") or item.get("id") or "") == legacy_id
                ),
                None,
            )
            if legacy_point is None:
                continue
            if not point.get("rois") and legacy_point.get("rois"):
                point["rois"] = legacy_point["rois"]
            anchors = list(point.get("visual_fingerprints", []))
            anchors.extend(
                anchor
                for anchor in _legacy_anchors(legacy_point)
                if anchor not in anchors
            )
            if anchors:
                point["visual_fingerprints"] = anchors
