from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdk_patrol.localization import AprilTagDetector, frame_fingerprint  # noqa: E402


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("points"), list):
        raise SystemExit(f"Invalid points YAML: {path}")
    return value


def _read_optional_yaml(path: str | None) -> Any:
    if not path:
        return None
    value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    return value


def registration_mode_error(
    point: dict[str, Any],
    *,
    has_pixel_rois: bool,
    has_3d: bool,
    coplanar: bool,
    identity_only: bool,
) -> str | None:
    capabilities = {str(value) for value in point.get("capabilities", [])}
    if (
        "vehicle_parking" in capabilities
        and not has_3d
        and not coplanar
        and not identity_only
    ):
        return (
            "Vehicle-parking points require --rois-3d-yaml, an explicit "
            "--rois-coplanar-with-tag assertion, or --identity-only."
        )
    if has_pixel_rois and not has_3d and not coplanar and not identity_only:
        return (
            "This point has pixel ROIs but no safe spatial mapping. Supply "
            "--rois-3d-yaml, explicitly assert --rois-coplanar-with-tag, or "
            "use --identity-only (legacy ORB remains active)."
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Register one configured point from 3+ still frames. Vehicle ground "
            "ROIs need --rois-3d-yaml; a 2-D homography is accepted only when "
            "the ROI is physically coplanar with the tag."
        )
    )
    parser.add_argument("--points", default=str(PROJECT_ROOT / "configs" / "points.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--point-id", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--min-good-images", type=int, default=3)
    parser.add_argument("--max-corner-std-px", type=float, default=4.0)
    parser.add_argument("--rois-yaml", help="Optional replacement pixel ROI mapping/list.")
    parser.add_argument(
        "--rois-3d-yaml",
        help="Mapping from ROI name to Nx3 vertices in the AprilTag coordinate frame, metres.",
    )
    parser.add_argument(
        "--rois-coplanar-with-tag",
        action="store_true",
        help="Assert that pixel ROIs lie on the same physical plane as the tag.",
    )
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="Record identity/fingerprint but keep registration_required=true.",
    )
    args = parser.parse_args()

    points_path = Path(args.points).expanduser().resolve()
    payload = _read_yaml(points_path)
    point = next(
        (
            value
            for value in payload["points"]
            if isinstance(value, dict) and str(value.get("point_id")) == args.point_id
        ),
        None,
    )
    if point is None:
        raise SystemExit(f"Unknown point_id: {args.point_id}")
    tag = point.get("tag")
    if not isinstance(tag, dict) or tag.get("id") is None:
        raise SystemExit(f"Point {args.point_id} has no configured tag.id")
    tag_id = int(tag["id"])
    detector = AprilTagDetector([tag_id], tag_family=str(tag.get("family", "tag36h11")))
    if not detector.available:
        raise SystemExit(f"AprilTag detector unavailable: {detector.unavailable_reason}")

    accepted_corners: list[np.ndarray] = []
    candidates: list[tuple[float, np.ndarray, Path]] = []
    hashes: list[str] = []
    seen_hashes: set[str] = set()
    for raw_path in args.images:
        path = Path(raw_path).expanduser().resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_hashes:
            print(f"SKIP duplicate image content: {path}", file=sys.stderr)
            continue
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"SKIP unreadable image: {path}", file=sys.stderr)
            continue
        observations = detector.detect(image)
        selected = next((item for item in observations if item.tag_id == tag_id), None)
        if selected is None:
            print(f"SKIP tag {tag_id} not accepted: {path}", file=sys.stderr)
            continue
        corners = np.asarray(selected.corners, dtype=np.float64)
        accepted_corners.append(corners)
        blur = float(
            cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )
        candidates.append((blur, image, path))
        hashes.append(digest)
        seen_hashes.add(digest)
    if len(candidates) < max(1, args.min_good_images):
        raise SystemExit(
            f"Only {len(candidates)} accepted image(s); require {args.min_good_images}."
        )
    stacked = np.stack(accepted_corners)
    median_corners = np.median(stacked, axis=0)
    corner_std = float(np.max(np.std(stacked, axis=0)))
    if corner_std > args.max_corner_std_px:
        raise SystemExit(
            f"Tag corner instability {corner_std:.2f}px exceeds "
            f"{args.max_corner_std_px:.2f}px; stop the robot and recapture."
        )
    _blur, best_image, best_path = max(candidates, key=lambda value: value[0])
    fingerprint = frame_fingerprint(best_image)
    fingerprint["keyframe_image"] = str(best_path)
    fingerprint["sha256"] = hashlib.sha256(best_path.read_bytes()).hexdigest()

    replacement_rois = _read_optional_yaml(args.rois_yaml)
    if replacement_rois is not None:
        point["rois"] = replacement_rois.get("rois", replacement_rois) if isinstance(
            replacement_rois, dict
        ) else replacement_rois
    rois_3d = _read_optional_yaml(args.rois_3d_yaml)
    if rois_3d is not None:
        mapped_3d = (
            rois_3d.get("roi_points_tag_m", rois_3d)
            if isinstance(rois_3d, dict)
            else rois_3d
        )
        if not isinstance(mapped_3d, dict):
            raise SystemExit("--rois-3d-yaml must contain a ROI-name to Nx3 mapping.")
        point["rois_3d"] = [
            {"id": str(name), "points_tag_m": vertices}
            for name, vertices in mapped_3d.items()
        ]
        point["roi_points_tag_m"] = mapped_3d
    has_pixel_rois = bool(point.get("rois"))
    has_3d = bool(point.get("roi_points_tag_m"))
    unsafe_reason = registration_mode_error(
        point,
        has_pixel_rois=has_pixel_rois,
        has_3d=has_3d,
        coplanar=bool(args.rois_coplanar_with_tag),
        identity_only=bool(args.identity_only),
    )
    if unsafe_reason:
        raise SystemExit(unsafe_reason)

    point["reference_tag_corners"] = [
        [round(float(x), 4), round(float(y), 4)] for x, y in median_corners
    ]
    point["rois_coplanar_with_tag"] = bool(args.rois_coplanar_with_tag)
    existing_fingerprints = [
        *[
            value
            for value in point.get("fingerprints", [])
            if isinstance(value, dict)
        ],
        *[
            value
            for value in point.get("visual_fingerprints", [])
            if isinstance(value, dict)
        ],
    ]
    point["fingerprints"] = [
        fingerprint,
        *[
            value
            for value in existing_fingerprints
            if value.get("sha256") != fingerprint["sha256"]
        ],
    ][:8]
    point.pop("visual_fingerprints", None)
    tag["registration_required"] = bool(args.identity_only)
    point["registration"] = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_images": len(candidates),
        "image_sha256": hashes,
        "max_corner_std_px": round(corner_std, 4),
        "spatial_mode": (
            "identity_only"
            if args.identity_only
            else "tag_3d"
            if has_3d
            else "coplanar_tag_homography"
            if args.rois_coplanar_with_tag
            else "identity_no_roi"
        ),
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else points_path
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"Registered {args.point_id} tag={tag_id} images={len(candidates)} "
        f"corner_std={corner_std:.2f}px output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
