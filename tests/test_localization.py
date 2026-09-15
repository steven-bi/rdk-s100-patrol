from __future__ import annotations

import json
import runpy
from pathlib import Path
import tempfile

import cv2
import numpy as np

from rdk_patrol.localization import (
    AprilTagDetector,
    FingerprintMatch,
    PointResolver,
    TagObservation,
    VisualFingerprintMatcher,
    frame_fingerprint,
    project_point_rois_from_tag,
    project_rois_with_homography,
)
from rdk_patrol.localization.fingerprint import _decode_orb


def _tag(tag_id: int) -> TagObservation:
    return TagObservation(
        tag_id=tag_id,
        corners=((20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)),
        center=(40.0, 40.0),
        area_px=1600.0,
        confidence=0.95,
    )


def _fingerprint(point_id: str, point_name: str) -> FingerprintMatch:
    return FingerprintMatch(
        point_id=point_id,
        point_name=point_name,
        score=0.88,
        margin=0.20,
        homography=None,
        geometry_verified=True,
        diagnostics={},
    )


def _resolver(stable_frames: int = 2) -> PointResolver:
    return PointResolver(
        [
            {
                "point_id": "vehicle_01",
                "point_name": "禁停点01",
                "tag_id": 7,
                "rois": {"no_parking": [[1, 1], [10, 1], [10, 10], [1, 10]]},
            },
            {
                "point_id": "trash_01",
                "point_name": "垃圾桶点01",
                "tag_id": 9,
                "rois": {},
            },
        ],
        required_stable_frames=stable_frames,
    )


def test_apriltag_backend_detects_only_configured_tag36h11_ids() -> None:
    detector = AprilTagDetector([101], min_area_px=100)
    if not detector.available:
        assert detector.unavailable_reason
        assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []
        return
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )

    def scene(tag_id: int) -> np.ndarray:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 120)
        canvas = np.full((200, 200), 255, dtype=np.uint8)
        canvas[40:160, 40:160] = marker
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    accepted = detector.detect(scene(101))
    ignored = detector.detect(scene(201))

    assert [item.tag_id for item in accepted] == [101]
    assert ignored == []


def test_tag_requires_consecutive_frames_and_only_configured_ids() -> None:
    resolver = _resolver(stable_frames=2)
    resolver.detector.detect = lambda _frame: [_tag(7)]  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: None  # type: ignore[method-assign]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert resolver.resolve(frame, 1.0) is None
    context = resolver.resolve(frame, 2.0)

    assert context is not None
    assert context.point_id == "vehicle_01"
    assert context.tag_id == 7
    assert context.source == "apriltag"


def test_tag_wins_when_visual_fingerprint_conflicts() -> None:
    resolver = _resolver(stable_frames=1)
    resolver.detector.detect = lambda _frame: [_tag(7)]  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: _fingerprint(  # type: ignore[method-assign]
        "trash_01", "垃圾桶点01"
    )

    context = resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), 10.0)

    assert context is not None
    assert context.point_id == "vehicle_01"
    assert context.source == "apriltag"
    assert context.diagnostics["tag_fingerprint_conflict"] is True
    assert context.diagnostics["tag_wins_conflict"] is True


def test_two_configured_point_tags_are_ambiguous_and_fail_closed() -> None:
    resolver = _resolver(stable_frames=1)
    resolver.detector.detect = lambda _frame: [_tag(7), _tag(9)]  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: _fingerprint(  # type: ignore[method-assign]
        "vehicle_01", "禁停点01"
    )

    context = resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), 10.0)

    assert context is None
    assert resolver.last_diagnostics["configured_tags_ambiguous"] is True


def test_no_tag_falls_back_to_stable_visual_fingerprint() -> None:
    resolver = _resolver(stable_frames=2)
    resolver.detector.detect = lambda _frame: []  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: _fingerprint(  # type: ignore[method-assign]
        "trash_01", "垃圾桶点01"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert resolver.resolve(frame, 1.0) is None
    context = resolver.resolve(frame, 2.0)

    assert context is not None
    assert context.point_id == "trash_01"
    assert context.source == "visual_fingerprint"
    assert context.tag_id is None


def test_both_localizers_fail_returns_none() -> None:
    resolver = _resolver(stable_frames=1)
    resolver.detector.detect = lambda _frame: []  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: None  # type: ignore[method-assign]
    assert resolver.resolve(np.zeros((40, 40, 3), dtype=np.uint8), 1.0) is None


def test_non_coplanar_tag_homography_is_refused() -> None:
    point = {
        "reference_tag_corners": [[10, 10], [30, 10], [30, 30], [10, 30]],
        "rois": {"ground": [[0, 50], [100, 50], [100, 90], [0, 90]]},
        "rois_coplanar_with_tag": False,
    }
    projected = project_point_rois_from_tag(point, _tag(7))
    assert projected.applied is False
    assert projected.rois == {}
    assert projected.reason == "non_coplanar_roi_requires_tag_3d_coordinates"


def test_tag_3d_roi_projects_with_explicit_left_camera_intrinsics() -> None:
    camera = np.asarray(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observation = TagObservation(
        tag_id=7,
        corners=((295.0, 215.0), (345.0, 215.0), (345.0, 265.0), (295.0, 265.0)),
        center=(320.0, 240.0),
        area_px=2500.0,
        confidence=1.0,
    )
    point = {
        "tag_size_m": 0.2,
        "roi_points_tag_m": {
            "ground": [
                [-0.2, -0.1, 0.0],
                [0.2, -0.1, 0.0],
                [0.2, 0.1, 0.0],
                [-0.2, 0.1, 0.0],
            ]
        },
    }

    projected = project_point_rois_from_tag(
        point,
        observation,
        camera_matrix=camera,
        distortion=np.zeros(5),
    )

    assert projected.applied
    assert projected.method == "tag_3d"
    assert np.allclose(projected.rois["ground"][0], (270.0, 265.0), atol=1.0)


def test_fingerprint_orb_ransac_matches_translated_scene() -> None:
    rng = np.random.default_rng(42)
    anchor_image = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    cv2.rectangle(anchor_image, (30, 40), (280, 200), (0, 255, 255), 4)
    transform = np.asarray(
        [[1.0, 0.0, 9.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    current = cv2.warpPerspective(anchor_image, transform, (320, 240))
    point = {
        "point_id": "p1",
        "point_name": "点1",
        "visual_fingerprints": [frame_fingerprint(anchor_image)],
        "fingerprint_min_score": 0.25,
        "fingerprint_min_margin": 0.0,
    }
    match = VisualFingerprintMatcher([point], geometry_required=True).match(current)

    assert match is not None
    assert match.point_id == "p1"
    assert match.geometry_verified
    assert match.homography is not None
    assert abs(match.homography[0][2] - 9.0) < 3.0


def test_from_file_accepts_json_compatible_points_yaml_without_pyyaml() -> None:
    payload = {
        "schema_version": "rdk-patrol-points/v1",
        "points": [
            {
                "point_id": "p1",
                "point_name": "点1",
                "tag_id": 1,
                "rois": {},
            }
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "points.yaml"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        resolver = PointResolver.from_file(path, required_stable_frames=1)
    assert set(resolver.by_id) == {"p1"}
    assert set(resolver.by_tag) == {1}


def test_shipped_points_yaml_loads_nested_tags_and_per_point_legacy() -> None:
    project_root = Path(__file__).resolve().parents[1]
    resolver = PointResolver.from_file(
        project_root / "configs" / "points.yaml", required_stable_frames=1
    )

    assert resolver.by_tag[101]["point_id"] == "no_parking_01"
    assert resolver.by_tag[201]["point_id"] == "trash_01"
    vehicle = resolver.by_id["no_parking_01"]
    assert vehicle["tag_size_m"] == 0.18
    assert vehicle["registration_required"] is True
    assert len(vehicle.get("visual_fingerprints", [])) > 0
    assert len(vehicle.get("rois", [])) > 0


def test_legacy_hex_orb_and_normalized_points_are_decoded() -> None:
    descriptor = bytes(range(32)).hex()
    payload = {
        "frame_width": 640,
        "frame_height": 1280,
        "orb": {
            "descriptors": [descriptor],
            "points": [[0.25, 0.75]],
        },
    }
    rows, points = _decode_orb(payload)
    assert rows is not None and rows.shape == (1, 32)
    assert points is not None
    assert tuple(points[0]) == (160.0, 960.0)


def test_unregistered_tag_identifies_point_but_does_not_load_static_legacy_roi() -> None:
    project_root = Path(__file__).resolve().parents[1]
    resolver = PointResolver.from_file(
        project_root / "configs" / "points.yaml", required_stable_frames=1
    )
    resolver.detector.detect = lambda _frame: [_tag(101)]  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: None  # type: ignore[method-assign]

    context = resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), 5.0)

    assert context is not None
    assert context.point_id == "no_parking_01"
    assert context.rois == {}
    assert context.diagnostics["projection_reason"] == (
        "tag_registration_required_and_legacy_geometry_unavailable"
    )


def test_downscaled_fingerprint_homography_projects_roi_in_original_coordinates() -> None:
    rng = np.random.default_rng(77)
    anchor_image = rng.integers(0, 255, size=(640, 1280, 3), dtype=np.uint8)
    cv2.rectangle(anchor_image, (200, 150), (900, 520), (255, 255, 0), 8)
    transform = np.asarray(
        [[1.0, 0.0, 32.0], [0.0, 1.0, 18.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    current = cv2.warpPerspective(anchor_image, transform, (1280, 640))
    point = {
        "point_id": "large",
        "point_name": "大图点位",
        "visual_fingerprints": [frame_fingerprint(anchor_image)],
        "fingerprint_min_score": 0.20,
        "fingerprint_min_margin": 0.0,
    }

    match = VisualFingerprintMatcher([point], geometry_required=True).match(current)

    assert match is not None and match.homography is not None
    projected = project_rois_with_homography(
        {"zone": [(200, 150), (400, 150), (400, 300), (200, 300)]},
        match.homography,
    )
    assert projected.applied
    first = projected.rois["zone"][0]
    assert abs(first[0] - 232.0) < 5.0
    assert abs(first[1] - 168.0) < 5.0


def test_vehicle_registration_cannot_be_marked_complete_without_safe_roi_mode() -> None:
    project_root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(project_root / "scripts" / "tools" / "register_point.py"),
        run_name="register_point_test",
    )
    validate = namespace["registration_mode_error"]
    vehicle = {"capabilities": ["vehicle_parking"]}

    error = validate(
        vehicle,
        has_pixel_rois=False,
        has_3d=False,
        coplanar=False,
        identity_only=False,
    )

    assert error is not None
    assert "--identity-only" in error
    assert (
        validate(
            vehicle,
            has_pixel_rois=False,
            has_3d=False,
            coplanar=False,
            identity_only=True,
        )
        is None
    )


def test_tag_never_exposes_unprojected_static_vehicle_roi() -> None:
    resolver = PointResolver(
        [
            {
                "point_id": "misconfigured",
                "point_name": "误配置车辆点",
                "capabilities": ["vehicle_parking"],
                "tag_id": 33,
                "registration_required": False,
                "rois": {
                    "no_parking": [[10, 10], [90, 10], [90, 90], [10, 90]]
                },
            }
        ],
        required_stable_frames=1,
    )
    resolver.detector.detect = lambda _frame: [_tag(33)]  # type: ignore[method-assign]
    resolver.fingerprint_matcher.match = lambda _frame: None  # type: ignore[method-assign]

    context = resolver.resolve(np.zeros((100, 100, 3), dtype=np.uint8), 1.0)

    assert context is not None
    assert context.point_id == "misconfigured"
    assert context.rois == {}
    assert context.diagnostics["projection_applied"] is False
