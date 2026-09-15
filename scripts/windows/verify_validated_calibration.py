#!/usr/bin/env python3
"""Read-only, fail-closed verifier for an exported GS130W calibration.

This verifier intentionally understands only the audited three-distance export
format produced by ``validate_gs130w_distance_web.py``.  It never modifies the
validated YAML, its session directory, or the pristine candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn

import yaml


EXPECTED_SCHEMA = "rdk-patrol-gs130w-distance-validation/v1"
EXPECTED_CALIBRATION_SCHEMA = "rdk-patrol-stereo/v1"
APPROVED_CANDIDATE_SHA256 = (
    "10fc69cfcc28b606db40955ea2727d4b5d0de2f378eb5a969d9d5ec0a62300c6"
)
# These are the exact deployed runtime sources approved for this calibration
# campaign.  A session must archive byte-identical copies; accepting merely a
# self-declared hash would make independent recomputation circular.
APPROVED_RUNTIME_DEPTH_SHA256 = "82aa6716cbf43078f0fa563770497badd9c514f0a3d066f6c603d58637d86ed7"
APPROVED_RUNTIME_CALIBRATION_SHA256 = "1377bf7982b96f1ec07664c0f4014b2b26624c76779a2decd7c0d59b0481e65f"
APPROVED_RUNTIME_GS130W_SHA256 = "2db1c0350fc275e1a3d784dfe85b0ecebfa355dc61e558d0fdde96ffb9e6ec9b"
APPROVED_VALIDATOR_SCRIPT_SHA256 = (
    "5de03036e49056645d63fecb093818ab9ccc92928a8f23f6e9011ddba13979af"
)
EXPECTED_BASELINE_M = 0.07870606271667704
BASELINE_ABSOLUTE_TOLERANCE_M = 0.000001
EXPECTED_SLOTS = ("1m", "3m", "5m")
EXPECTED_NOMINAL_M = {"1m": 1.0, "3m": 3.0, "5m": 5.0}
KNOWN_DISTANCE_RANGES_M = {
    "1m": (0.8, 1.2),
    "3m": (2.5, 3.5),
    "5m": (4.3, 5.7),
}
EXPECTED_THRESHOLDS: dict[str, float | int] = {
    "frames_per_attempt": 3,
    "minimum_roi_pixels_after_boundary_erosion": 400,
    "minimum_valid_pixels": 200,
    "minimum_valid_fraction": 0.35,
    "maximum_relative_mad": 0.10,
    "maximum_central80_spread_over_median": 0.25,
    "maximum_relative_error": 0.10,
    "minimum_passing_frames": 2,
    "maximum_passing_estimate_span_over_median": 0.03,
    "minimum_known_distance_separation": 0.05,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_YAML_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
METRIC_RELATIVE_TOLERANCE = 1e-5
METRIC_ABSOLUTE_TOLERANCE = 1e-6


class VerificationError(RuntimeError):
    """An artifact failed a mandatory safety or provenance check."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise VerificationError("YAML contains an unhashable mapping key") from exc
        if duplicate:
            raise VerificationError(f"YAML contains duplicate key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def fail(message: str) -> NoReturn:
    raise VerificationError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes_limited(path: Path, maximum: int, label: str) -> bytes:
    require(path.is_file(), f"{label} is missing: {path}")
    size = path.stat().st_size
    require(0 < size <= maximum, f"{label} has an unsafe size ({size} bytes): {path}")
    return path.read_bytes()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = read_bytes_limited(path, MAX_YAML_BYTES, label)
    try:
        text = raw.decode("utf-8-sig", errors="strict")
        value = yaml.load(text, Loader=UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError, VerificationError) as exc:
        fail(f"{label} is not strict UTF-8 YAML: {exc}")
    require(isinstance(value, dict), f"{label} root must be a mapping")
    return value, raw


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = read_bytes_limited(path, MAX_JSON_BYTES, label)
    try:
        value = json.loads(raw.decode("utf-8-sig", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{label} is not strict UTF-8 JSON: {exc}")
    require(isinstance(value, dict), f"{label} root must be a mapping")
    return value, raw


def mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def exact_integer(value: Any, label: str) -> int:
    number = finite_number(value, label)
    require(number.is_integer(), f"{label} must be an integer")
    return int(number)


def exact_bool(value: Any, label: str) -> bool:
    require(type(value) is bool, f"{label} must be true or false")
    return bool(value)


def sha256_value(value: Any, label: str) -> str:
    text = str(value).lower()
    require(SHA256_RE.fullmatch(text) is not None, f"{label} must be a SHA-256 value")
    return text


def close(a: Any, b: Any, label: str, tolerance: float = 1e-12) -> None:
    first = finite_number(a, f"{label} (first)")
    second = finite_number(b, f"{label} (second)")
    require(abs(first - second) <= tolerance, f"{label} differs: {first} vs {second}")


def safe_child(base: Path, filename: Any, label: str, *, exact_name: str | None = None) -> Path:
    require(isinstance(filename, str) and filename, f"{label} must be a filename")
    require(filename == Path(filename).name, f"{label} must not contain a directory")
    require("/" not in filename and "\\" not in filename, f"{label} must be a plain filename")
    if exact_name is not None:
        require(filename == exact_name, f"{label} must be {exact_name!r}")
    result = (base / filename).resolve(strict=False)
    require(result.parent == base, f"{label} escapes the validation session directory")
    return result


def safe_relative_child(base: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label} must be a relative path")
    relative_path = Path(relative)
    require(not relative_path.is_absolute(), f"{label} must be relative")
    require(".." not in relative_path.parts, f"{label} must not contain '..'")
    result = (base / relative_path).resolve(strict=False)
    try:
        common = os.path.commonpath((str(base), str(result)))
    except ValueError as exc:
        fail(f"{label} is outside the validation session: {exc}")
    require(Path(common) == base, f"{label} is outside the validation session")
    return result


def verify_thresholds(value: Any, label: str) -> dict[str, Any]:
    thresholds = mapping(value, label)
    require(
        set(thresholds) == set(EXPECTED_THRESHOLDS),
        f"{label} keys do not match the approved strict threshold set",
    )
    for key, expected in EXPECTED_THRESHOLDS.items():
        close(thresholds[key], expected, f"{label}.{key}")
    return thresholds


def immutable_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("valid", None)
    validation = result.get("validation")
    if isinstance(validation, dict):
        validation.pop("known_distance_results", None)
        validation.pop("distance_validation", None)
    return result


def numeric_matrix(value: Any, rows: int, columns: int, label: str) -> list[list[float]]:
    outer = sequence(value, label)
    require(len(outer) == rows, f"{label} must have {rows} rows")
    result: list[list[float]] = []
    for row_index, row_value in enumerate(outer):
        row = sequence(row_value, f"{label}[{row_index}]")
        require(len(row) == columns, f"{label}[{row_index}] must have {columns} columns")
        result.append(
            [finite_number(item, f"{label}[{row_index}][{column_index}]") for column_index, item in enumerate(row)]
        )
    return result


def verify_calibration_shape(payload: dict[str, Any]) -> None:
    require(payload.get("schema_version") == EXPECTED_CALIBRATION_SCHEMA, "calibration schema_version mismatch")
    require(exact_bool(payload.get("enabled"), "calibration enabled"), "calibration enabled is not true")
    require(not exact_bool(payload.get("images_are_rectified"), "images_are_rectified"), "GS130W input must be marked unrectified")
    size = sequence(payload.get("image_size"), "image_size")
    require(size == [640, 1280], "image_size must be exactly [640, 1280]")

    left = mapping(payload.get("left"), "left camera")
    right = mapping(payload.get("right"), "right camera")
    left_k = numeric_matrix(left.get("camera_matrix"), 3, 3, "left camera matrix")
    right_k = numeric_matrix(right.get("camera_matrix"), 3, 3, "right camera matrix")
    for side, section, camera_matrix in (("left", left, left_k), ("right", right, right_k)):
        distortion = sequence(section.get("distortion_coefficients"), f"{side} distortion coefficients")
        require(len(distortion) == 5, f"{side} distortion coefficients must contain five values")
        for index, value in enumerate(distortion):
            finite_number(value, f"{side} distortion coefficient {index}")
        require(camera_matrix[0][0] > 0.0 and camera_matrix[1][1] > 0.0, f"{side} focal lengths must be positive")
        require(abs(camera_matrix[2][2] - 1.0) <= 1e-12, f"{side} camera matrix bottom-right value must be 1")

    rotation = numeric_matrix(payload.get("rotation"), 3, 3, "stereo rotation")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    require(abs(determinant - 1.0) <= 0.01, "stereo rotation determinant is not approximately 1")
    translation_values = sequence(payload.get("translation_m"), "translation_m")
    require(len(translation_values) == 3, "translation_m must contain three values")
    translation = [finite_number(item, f"translation_m[{index}]") for index, item in enumerate(translation_values)]
    translation_norm = math.sqrt(sum(item * item for item in translation))
    baseline = finite_number(payload.get("baseline_m"), "baseline_m")
    require(abs(translation_norm - baseline) <= 1e-9, "translation norm does not equal baseline_m")

    rectification = mapping(payload.get("rectification"), "rectification")
    numeric_matrix(rectification.get("left_matrix"), 3, 3, "left rectification matrix")
    numeric_matrix(rectification.get("right_matrix"), 3, 3, "right rectification matrix")
    left_projection = numeric_matrix(rectification.get("left_projection"), 3, 4, "left projection matrix")
    right_projection = numeric_matrix(rectification.get("right_projection"), 3, 4, "right projection matrix")
    q_matrix = numeric_matrix(rectification.get("q_matrix"), 4, 4, "Q matrix")
    focal = left_projection[0][0]
    require(focal > 0.0, "rectified focal length must be positive")
    require(abs(right_projection[0][3] + focal * baseline) <= 1e-6, "right projection baseline term is inconsistent")
    require(abs(q_matrix[3][2] - (1.0 / baseline)) <= 1e-6, "Q matrix baseline term is inconsistent")

    matcher = mapping(payload.get("matcher"), "matcher")
    approved_matcher = {
        "min_disparity": 0,
        "num_disparities": 128,
        "block_size": 5,
        "uniqueness_ratio": 10,
        "speckle_window_size": 80,
        "speckle_range": 2,
        "minimum_valid_pixels": 20,
    }
    for key, expected in approved_matcher.items():
        require(exact_integer(matcher.get(key), f"matcher.{key}") == expected, f"matcher.{key} is not approved")
    close(matcher.get("min_distance_m"), 0.3, "matcher.min_distance_m")
    close(matcher.get("max_distance_m"), 30.0, "matcher.max_distance_m")


class RuntimeRecomputeContext:
    def __init__(self, estimator: Any, matcher: Any, calibration: Any) -> None:
        self.estimator = estimator
        self.matcher = matcher
        self.calibration = calibration


def load_archived_runtime(
    depth_path: Path,
    calibration_path: Path,
    pristine_payload: dict[str, Any],
) -> RuntimeRecomputeContext:
    """Load only hash-approved archived runtime code in an isolated package."""

    depth_raw = read_bytes_limited(depth_path, MAX_JSON_BYTES, "archived runtime depth source")
    calibration_raw = read_bytes_limited(
        calibration_path, MAX_JSON_BYTES, "archived runtime calibration source"
    )
    require(
        sha256_bytes(depth_raw) == APPROVED_RUNTIME_DEPTH_SHA256,
        "runtime_depth.py is not the approved deployed depth source",
    )
    require(
        sha256_bytes(calibration_raw) == APPROVED_RUNTIME_CALIBRATION_SHA256,
        "runtime_calibration.py is not the approved deployed calibration source",
    )

    try:
        import cv2  # noqa: F401 - imported runtime source requires it
        import numpy  # noqa: F401 - imported runtime source requires it
    except ImportError as exc:
        fail(f"OpenCV and NumPy are required for independent stereo recomputation: {exc}")

    package_name = "_gs130w_approved_runtime"
    package = types.ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    contracts_package = types.ModuleType("rdk_patrol")
    contracts_package.__path__ = []  # type: ignore[attr-defined]
    contracts_module = types.ModuleType("rdk_patrol.contracts")

    class _DepthEstimateStub:
        pass

    contracts_module.DepthEstimate = _DepthEstimateStub  # type: ignore[attr-defined]
    names = (
        package_name,
        f"{package_name}.calibration",
        f"{package_name}.depth",
        "rdk_patrol",
        "rdk_patrol.contracts",
    )
    sentinel = object()
    previous_modules = {name: sys.modules.get(name, sentinel) for name in names}
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules[package_name] = package
        sys.modules["rdk_patrol"] = contracts_package
        sys.modules["rdk_patrol.contracts"] = contracts_module
        calibration_spec = importlib.util.spec_from_file_location(
            f"{package_name}.calibration", calibration_path
        )
        require(
            calibration_spec is not None and calibration_spec.loader is not None,
            "cannot create loader for archived runtime calibration source",
        )
        calibration_module = importlib.util.module_from_spec(calibration_spec)
        sys.modules[f"{package_name}.calibration"] = calibration_module
        calibration_spec.loader.exec_module(calibration_module)

        depth_spec = importlib.util.spec_from_file_location(
            f"{package_name}.depth", depth_path
        )
        require(
            depth_spec is not None and depth_spec.loader is not None,
            "cannot create loader for archived runtime depth source",
        )
        depth_module = importlib.util.module_from_spec(depth_spec)
        sys.modules[f"{package_name}.depth"] = depth_module
        depth_spec.loader.exec_module(depth_module)
    except VerificationError:
        raise
    except Exception as exc:
        fail(f"approved archived runtime source could not be loaded: {type(exc).__name__}: {exc}")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        for name, previous in previous_modules.items():
            if previous is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous  # type: ignore[assignment]

    source_payload = copy.deepcopy(pristine_payload)
    nested = source_payload.get("calibration")
    runtime_payload = dict(nested) if isinstance(nested, dict) else dict(source_payload)
    for key in (
        "physical_left_view",
        "physical_right_view",
        "runtime_rotation",
        "view_mapping",
    ):
        if key not in runtime_payload and key in source_payload:
            runtime_payload[key] = copy.deepcopy(source_payload[key])
    runtime_payload["valid"] = True
    try:
        calibration = calibration_module.calibration_from_mapping(runtime_payload)
    except Exception as exc:
        fail(f"archived runtime calibration loader failed: {type(exc).__name__}: {exc}")
    require(bool(getattr(calibration, "valid", False)), "archived runtime rejected the pristine calibration")

    matcher_values = source_payload.get("matcher", runtime_payload.get("matcher", {}))
    if not isinstance(matcher_values, dict):
        matcher_values = {}

    def safe_int(name: str, default: int) -> int:
        try:
            return int(matcher_values.get(name, default))
        except (TypeError, ValueError, OverflowError):
            return int(default)

    def safe_float(name: str, default: float) -> float:
        try:
            value = float(matcher_values.get(name, default))
        except (TypeError, ValueError, OverflowError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    valid_disparity_min = safe_float(
        "valid_disparity_min", safe_float("min_disparity", 0.0) + 0.75
    )
    try:
        estimator = depth_module.StereoDepthEstimator(
            calibration,
            min_valid_pixels=safe_int("minimum_valid_pixels", 24),
            min_disparity=valid_disparity_min,
            min_distance_m=safe_float("min_distance_m", 0.15),
            max_distance_m=safe_float("max_distance_m", 30.0),
            max_relative_mad=safe_float("max_relative_mad", 0.40),
            matcher_min_disparity=safe_int("min_disparity", 0),
            num_disparities=safe_int("num_disparities", 128),
            block_size=safe_int("block_size", 5),
            uniqueness_ratio=safe_int("uniqueness_ratio", 10),
            speckle_window_size=safe_int("speckle_window_size", 80),
            speckle_range=safe_int("speckle_range", 2),
        )
        width, height = tuple(int(value) for value in calibration.image_size)
        require((width, height) == (640, 1280), "archived runtime calibration image size mismatch")
        matcher = estimator._create_matcher(width)
    except VerificationError:
        raise
    except Exception as exc:
        fail(f"approved runtime matcher could not be created: {type(exc).__name__}: {exc}")
    return RuntimeRecomputeContext(estimator, matcher, calibration)


def decode_color_png(path: Path, label: str) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        fail(f"OpenCV and NumPy are required for {label}: {exc}")
    raw = read_bytes_limited(path, 64 * 1024 * 1024, label)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(image is not None and image.size > 0, f"{label} is not a decodable PNG image")
    require(image.dtype == np.uint8, f"{label} must decode as uint8")
    require(image.shape == (1280, 640, 3), f"{label} must be exactly 640x1280 BGR")
    return image


def recompute_frame_metrics(
    context: RuntimeRecomputeContext,
    left_path: Path,
    right_path: Path,
    roi: Mapping[str, Any],
    known_distance_m: float,
) -> dict[str, Any]:
    """Recompute one result from archived rectified images and approved SGBM."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        fail(f"OpenCV and NumPy are required for stereo recomputation: {exc}")
    left = decode_color_png(left_path, "rectified physical-left image")
    right = decode_color_png(right_path, "rectified physical-right image")
    require(left.shape == right.shape, "rectified stereo images have different shapes")
    x = exact_integer(roi.get("x"), "recompute ROI x")
    y = exact_integer(roi.get("y"), "recompute ROI y")
    width = exact_integer(roi.get("width"), "recompute ROI width")
    height = exact_integer(roi.get("height"), "recompute ROI height")

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    try:
        disparity = context.matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        xyz = cv2.reprojectImageTo3D(
            disparity,
            np.asarray(context.calibration.q_matrix, dtype=np.float64),
            handleMissingValues=False,
        )
    except (cv2.error, TypeError, ValueError, AttributeError) as exc:
        fail(f"independent SGBM computation failed: {type(exc).__name__}: {exc}")

    mask = np.zeros((left.shape[0], left.shape[1]), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 1
    erosion = max(1, int(round(min(width, height) * 0.08)))
    if erosion > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion * 2 + 1, erosion * 2 + 1)
        )
        mask = cv2.erode(mask, kernel)
    roi_pixels = int(np.count_nonzero(mask))
    distances = np.linalg.norm(xyz, axis=2)
    valid = (
        (mask > 0)
        & np.isfinite(distances)
        & np.isfinite(disparity)
        & (disparity > context.estimator.min_disparity)
        & (distances >= context.estimator.min_distance_m)
        & (distances <= context.estimator.max_distance_m)
        & (xyz[:, :, 2] > 0)
    )
    values = distances[valid].astype(np.float64)
    disparities = disparity[valid].astype(np.float64)
    valid_pixels = int(values.size)
    valid_fraction = valid_pixels / max(roi_pixels, 1)

    estimate = None
    relative_mad = None
    central80_spread = None
    relative_error = None
    median_disparity = None
    trimmed_pixels = 0
    low = None
    high = None
    if valid_pixels:
        low_value, high_value = np.percentile(values, [10.0, 90.0])
        low, high = float(low_value), float(high_value)
        trimmed = values[(values >= low_value) & (values <= high_value)]
        if trimmed.size < 200:
            trimmed = values
        trimmed_pixels = int(trimmed.size)
        estimate = float(np.median(trimmed))
        mad = float(np.median(np.abs(trimmed - estimate)))
        relative_mad = mad / max(estimate, 1e-9)
        central80_spread = (high - low) / max(estimate, 1e-9)
        relative_error = abs(estimate - known_distance_m) / known_distance_m
        median_disparity = float(np.median(disparities))

    checks = {
        "roi_pixels": roi_pixels >= 400,
        "valid_pixels": valid_pixels >= 200,
        "valid_fraction": valid_fraction >= 0.35,
        "relative_mad": relative_mad is not None and relative_mad <= 0.10,
        "central80_spread": central80_spread is not None
        and central80_spread <= 0.25,
        "relative_error": relative_error is not None and relative_error <= 0.10,
    }
    return {
        "passed": bool(all(checks.values())),
        "failure_checks": [name for name, passed in checks.items() if not passed],
        "known_distance_m": float(known_distance_m),
        "estimated_distance_m": estimate,
        "relative_error": relative_error,
        "roi": {"x": x, "y": y, "width": width, "height": height},
        "roi_pixels_after_boundary_erosion": roi_pixels,
        "boundary_erosion_px": erosion,
        "valid_pixels": valid_pixels,
        "valid_fraction": float(valid_fraction),
        "trimmed_pixels": trimmed_pixels,
        "relative_mad": relative_mad,
        "central80_low_m": low,
        "central80_high_m": high,
        "central80_spread_over_median": central80_spread,
        "median_disparity_px": median_disparity,
        "thresholds": {
            "minimum_roi_pixels": 400,
            "minimum_valid_pixels": 200,
            "minimum_valid_fraction": 0.35,
            "maximum_relative_mad": 0.10,
            "maximum_central80_spread_over_median": 0.25,
            "maximum_relative_error": 0.10,
            "minimum_valid_disparity_px": float(context.estimator.min_disparity),
            "minimum_distance_m": float(context.estimator.min_distance_m),
            "maximum_distance_m": float(context.estimator.max_distance_m),
        },
        "checks": checks,
        "distance_definition": "physical_left_camera_to_roi_euclidean_m",
    }


def metric_float_matches(recorded: Any, recomputed: Any, label: str) -> None:
    if recorded is None or recomputed is None:
        require(recorded is None and recomputed is None, f"{label} availability differs on recomputation")
        return
    first = finite_number(recorded, f"{label} recorded")
    second = finite_number(recomputed, f"{label} recomputed")
    tolerance = max(
        METRIC_ABSOLUTE_TOLERANCE,
        abs(second) * METRIC_RELATIVE_TOLERANCE,
    )
    require(
        abs(first - second) <= tolerance,
        f"{label} differs after independent recomputation: recorded={first}, recomputed={second}",
    )


def verify_recomputed_metrics(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any], label: str
) -> None:
    require(set(recorded) == set(recomputed), f"{label} metric field set differs from recomputation")
    exact_fields = (
        "passed",
        "failure_checks",
        "roi",
        "roi_pixels_after_boundary_erosion",
        "boundary_erosion_px",
        "valid_pixels",
        "trimmed_pixels",
        "thresholds",
        "checks",
        "distance_definition",
    )
    for field in exact_fields:
        require(recorded.get(field) == recomputed.get(field), f"{label}.{field} differs after recomputation")
    float_fields = (
        "known_distance_m",
        "estimated_distance_m",
        "relative_error",
        "valid_fraction",
        "relative_mad",
        "central80_low_m",
        "central80_high_m",
        "central80_spread_over_median",
        "median_disparity_px",
    )
    for field in float_fields:
        metric_float_matches(recorded.get(field), recomputed.get(field), f"{label}.{field}")


def verify_audit_log(
    path: Path,
    session_id: str,
    candidate_sha: str,
    validated_sha: str,
    validated_filename: str,
    before_hash: str,
) -> None:
    raw = read_bytes_limited(path, MAX_JSON_BYTES, "audit log")
    lines = raw.splitlines(keepends=True)
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    require(nonempty_indexes, "audit log is empty")
    final_index = nonempty_indexes[-1]
    require(final_index == len(lines) - 1, "audit log has data after its final event")
    prefix = b"".join(lines[:final_index])
    require(
        sha256_bytes(prefix) == before_hash,
        "audit log prefix does not match audit_log_sha256_before_validation",
    )

    previous = "0" * 64
    final_event: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"audit log line {line_number} is invalid: {exc}")
        require(isinstance(event, dict), f"audit log line {line_number} is not a mapping")
        claimed = sha256_value(event.get("event_sha256"), f"audit line {line_number} event_sha256")
        body = dict(event)
        body.pop("event_sha256", None)
        require(
            body.get("previous_event_sha256") == previous,
            f"audit chain is broken at line {line_number}",
        )
        require(body.get("schema_version") == EXPECTED_SCHEMA, f"audit schema mismatch at line {line_number}")
        require(body.get("session_id") == session_id, f"audit session mismatch at line {line_number}")
        canonical = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        require(sha256_bytes(canonical) == claimed, f"audit event hash mismatch at line {line_number}")
        previous = claimed
        final_event = event

    require(final_event is not None, "audit log has no final event")
    require(final_event.get("event") == "validated_yaml_created", "audit final event is not validated_yaml_created")
    details = mapping(final_event.get("details"), "audit final event details")
    require(details.get("file") == validated_filename, "audit final event has the wrong validated filename")
    require(details.get("sha256") == validated_sha, "audit final event has the wrong validated YAML hash")
    require(
        details.get("source_candidate_sha256") == candidate_sha,
        "audit final event has the wrong source candidate hash",
    )


def verify_checksums(session_dir: Path, required_relative_paths: Iterable[str]) -> None:
    checksum_path = session_dir / "SESSION_SHA256SUMS.txt"
    raw = read_bytes_limited(checksum_path, MAX_JSON_BYTES, "session checksum list")
    entries: dict[str, str] = {}
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        fail(f"session checksum list is not UTF-8: {exc}")
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum-list line {line_number}")
        digest, relative = match.groups()
        require(relative not in entries, f"duplicate checksum-list path: {relative}")
        path = safe_relative_child(session_dir, relative, f"checksum path on line {line_number}")
        require(path.is_file(), f"checksummed file is missing: {relative}")
        require(sha256_file(path) == digest, f"checksummed file changed: {relative}")
        entries[relative] = digest
    for relative in required_relative_paths:
        require(relative in entries, f"required artifact is absent from SESSION_SHA256SUMS.txt: {relative}")


def verify_session(
    manifest: dict[str, Any],
    session_dir: Path,
    validated_path: Path,
    validated_sha: str,
    candidate_sha: str,
    yaml_results: list[dict[str, Any]],
    validator_script_filename: str,
    validator_script_sha256: str,
    recompute_context: RuntimeRecomputeContext,
    additional_required_files: Iterable[str] = (),
) -> list[str]:
    require(manifest.get("schema_version") == EXPECTED_SCHEMA, "session manifest schema mismatch")
    session_id = str(manifest.get("session_id") or "")
    require(session_id != "", "session manifest has no session_id")
    require(session_id == session_dir.name, "session_id must equal the validation session directory name")
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", session_id) is not None,
        "session_id has an unsafe format",
    )
    require(manifest.get("status") == "completed", "validation session is not completed")
    require(manifest.get("candidate_sha256") == candidate_sha, "manifest candidate hash mismatch")
    require(
        manifest.get("validator_script_file") == validator_script_filename,
        "manifest archived-validator filename mismatch",
    )
    require(
        manifest.get("validator_script_sha256") == validator_script_sha256,
        "manifest archived-validator hash mismatch",
    )
    runtime_archives = mapping(
        manifest.get("runtime_source_archives"), "manifest runtime_source_archives"
    )
    require(
        set(runtime_archives) == {"depth", "calibration", "gs130w"},
        "manifest runtime source set is incomplete",
    )
    depth_archive = mapping(runtime_archives.get("depth"), "manifest depth runtime source")
    calibration_archive = mapping(
        runtime_archives.get("calibration"), "manifest calibration runtime source"
    )
    gs130w_archive = mapping(runtime_archives.get("gs130w"), "manifest GS130W runtime source")
    require(depth_archive.get("file") == "runtime_depth.py", "manifest runtime-depth filename mismatch")
    require(
        depth_archive.get("sha256") == APPROVED_RUNTIME_DEPTH_SHA256,
        "manifest runtime-depth SHA is not approved",
    )
    require(
        calibration_archive.get("file") == "runtime_calibration.py",
        "manifest runtime-calibration filename mismatch",
    )
    require(
        calibration_archive.get("sha256") == APPROVED_RUNTIME_CALIBRATION_SHA256,
        "manifest runtime-calibration SHA is not approved",
    )
    require(gs130w_archive.get("file") == "runtime_gs130w.py", "manifest runtime-GS130W filename mismatch")
    require(
        gs130w_archive.get("sha256") == APPROVED_RUNTIME_GS130W_SHA256,
        "manifest runtime-GS130W SHA is not approved",
    )
    require(manifest.get("validated_output_file") == validated_path.name, "manifest validated filename mismatch")
    require(manifest.get("validated_yaml_sha256") == validated_sha, "manifest validated YAML hash mismatch")
    verify_thresholds(manifest.get("thresholds"), "session thresholds")

    completion = mapping(manifest.get("completion"), "session completion")
    require(completion.get("validated_output_file") == validated_path.name, "completion filename mismatch")
    require(completion.get("validated_yaml_sha256") == validated_sha, "completion YAML hash mismatch")
    require(completion.get("source_candidate_sha256") == candidate_sha, "completion candidate hash mismatch")

    slots = mapping(manifest.get("slots"), "session slots")
    require(set(slots) == set(EXPECTED_SLOTS), "session must contain exactly the 1m, 3m and 5m slots")
    required_files: list[str] = [
        "candidate_pristine.yaml",
        validated_path.name,
        *list(additional_required_files),
    ]
    all_combined_hashes: list[str] = []

    result_by_slot = {str(item.get("slot")): item for item in yaml_results}
    for slot in EXPECTED_SLOTS:
        slot_data = mapping(slots[slot], f"session slots.{slot}")
        close(slot_data.get("nominal_distance_m"), EXPECTED_NOMINAL_M[slot], f"session {slot} nominal distance")
        require(exact_bool(slot_data.get("locked"), f"session {slot}.locked"), f"session {slot} is not locked")
        locked_attempt = exact_integer(slot_data.get("locked_attempt"), f"session {slot}.locked_attempt")
        require(locked_attempt >= 1, f"session {slot} locked attempt is invalid")
        aggregate = mapping(slot_data.get("aggregate"), f"session {slot}.aggregate")
        require(aggregate == result_by_slot[slot], f"validated YAML {slot} aggregate differs from session manifest")

        attempts = sequence(slot_data.get("attempts"), f"session {slot}.attempts")
        matches = [item for item in attempts if exact_integer(mapping(item, "attempt").get("number"), "attempt number") == locked_attempt]
        require(len(matches) == 1, f"session {slot} does not have exactly one locked attempt")
        attempt = mapping(matches[0], f"session {slot} locked attempt")
        require(attempt.get("status") == "locked", f"session {slot} locked attempt status is wrong")
        close(
            attempt.get("known_distance_m"),
            aggregate.get("known_distance_m"),
            f"session {slot} attempt known distance",
        )
        summary = mapping(attempt.get("summary"), f"session {slot} attempt summary")
        require(exact_bool(summary.get("passed"), f"session {slot} summary.passed"), f"session {slot} summary did not pass")
        frames = sequence(attempt.get("frames"), f"session {slot} locked frames")
        require(len(frames) == 3, f"session {slot} locked attempt must have exactly three frames")
        require(exact_integer(summary.get("total_frames"), f"session {slot} summary.total_frames") == 3, f"session {slot} summary total is wrong")

        expected_dir = f"slots/{slot}/attempt_{locked_attempt:02d}"
        require(aggregate.get("session_relative_path") == expected_dir, f"validated YAML {slot} session path mismatch")
        aggregate_combined = sequence(aggregate.get("combined_image_sha256"), f"validated YAML {slot} image hashes")
        aggregate_results = sequence(aggregate.get("frame_result_sha256"), f"validated YAML {slot} result hashes")
        require(len(aggregate_combined) == 3 and len(aggregate_results) == 3, f"validated YAML {slot} must bind three frames")

        passing_metrics: list[dict[str, Any]] = []
        shared_rois: list[dict[str, Any]] = []
        roi_source_frames: list[int] = []
        sharpness_by_frame: list[tuple[float, int]] = []
        evaluated_timestamps: list[str] = []
        for frame_index, frame_value in enumerate(frames, 1):
            frame = mapping(frame_value, f"session {slot} frame {frame_index}")
            require(exact_integer(frame.get("index"), f"session {slot} frame index") == frame_index, f"session {slot} frame ordering mismatch")
            require(frame.get("candidate_sha256") == candidate_sha, f"session {slot} frame candidate mismatch")
            combined_hash = sha256_value(frame.get("combined_pixel_sha256"), f"session {slot} frame combined hash")
            require(combined_hash == aggregate_combined[frame_index - 1], f"validated YAML {slot} combined hash mismatch")
            all_combined_hashes.append(combined_hash)
            relative_dir = str(frame.get("relative_dir") or "")
            expected_frame_dir = f"{expected_dir}/frame_{frame_index:02d}"
            require(relative_dir == expected_frame_dir, f"session {slot} frame directory mismatch")
            frame_dir = safe_relative_child(session_dir, relative_dir, f"session {slot} frame directory")
            require(frame_dir.is_dir(), f"session {slot} frame directory is missing")
            hashes = mapping(frame.get("file_sha256"), f"session {slot} frame file hashes")
            require(
                {"result.json", "rectified_left.png", "rectified_right.png"}.issubset(hashes),
                f"session {slot} frame lacks result.json or a rectified stereo image",
            )
            result_hash = sha256_value(hashes["result.json"], f"session {slot} frame result hash")
            require(result_hash == aggregate_results[frame_index - 1], f"validated YAML {slot} result hash mismatch")
            for filename, expected_hash in hashes.items():
                file_path = safe_child(frame_dir, filename, f"session {slot} frame filename")
                require(file_path.is_file(), f"session frame file is missing: {file_path}")
                require(sha256_file(file_path) == sha256_value(expected_hash, f"hash for {filename}"), f"session frame file changed: {file_path}")
                required_files.append(file_path.relative_to(session_dir).as_posix())
            result_doc, _ = load_json(frame_dir / "result.json", f"session {slot} result.json")
            require(result_doc.get("schema_version") == EXPECTED_SCHEMA, f"session {slot} result schema mismatch")
            require(result_doc.get("session_id") == session_id, f"session {slot} result session mismatch")
            require(result_doc.get("slot") == slot, f"session {slot} result slot mismatch")
            require(exact_integer(result_doc.get("attempt"), f"session {slot} result attempt") == locked_attempt, f"session {slot} result attempt mismatch")
            require(exact_integer(result_doc.get("frame"), f"session {slot} result frame") == frame_index, f"session {slot} result frame mismatch")
            require(result_doc.get("candidate_sha256") == candidate_sha, f"session {slot} result candidate mismatch")
            require(result_doc.get("combined_pixel_sha256") == combined_hash, f"session {slot} result image mismatch")
            require(
                exact_bool(
                    result_doc.get("shared_roi_for_three_frames"),
                    f"session {slot} result shared_roi_for_three_frames",
                ),
                f"session {slot} result was not evaluated with one shared ROI",
            )
            roi_source_frames.append(
                exact_integer(result_doc.get("roi_source_frame"), f"session {slot} result roi_source_frame")
            )
            metrics = mapping(result_doc.get("metrics"), f"session {slot} result metrics")
            require(metrics == frame.get("metrics"), f"session {slot} result metrics differ from manifest")
            evaluated_at = str(result_doc.get("evaluated_at") or "")
            require(evaluated_at != "", f"session {slot} result has no evaluated_at")
            require(frame.get("evaluated_at") == evaluated_at, f"session {slot} evaluated_at differs from result")
            evaluated_timestamps.append(evaluated_at)
            roi = mapping(metrics.get("roi"), f"session {slot} result ROI")
            require(set(roi) == {"x", "y", "width", "height"}, f"session {slot} ROI keys are invalid")
            roi_x = exact_integer(roi.get("x"), f"session {slot} ROI x")
            roi_y = exact_integer(roi.get("y"), f"session {slot} ROI y")
            roi_width = exact_integer(roi.get("width"), f"session {slot} ROI width")
            roi_height = exact_integer(roi.get("height"), f"session {slot} ROI height")
            require(
                roi_x >= 0
                and roi_y >= 0
                and roi_width >= 5
                and roi_height >= 5
                and roi_x + roi_width <= 640
                and roi_y + roi_height <= 1280,
                f"session {slot} ROI is outside the rectified physical-left image",
            )
            require(frame.get("roi") == roi, f"session {slot} frame ROI differs from result metrics")
            shared_rois.append(roi)
            sharpness_by_frame.append(
                (finite_number(frame.get("sharpness_score"), f"session {slot} frame sharpness_score"), frame_index)
            )
            metrics_passed = exact_bool(metrics.get("passed"), f"session {slot} metrics.passed")
            recomputed_metrics = recompute_frame_metrics(
                recompute_context,
                frame_dir / "rectified_left.png",
                frame_dir / "rectified_right.png",
                roi,
                finite_number(aggregate.get("known_distance_m"), f"session {slot} aggregate known distance"),
            )
            verify_recomputed_metrics(
                metrics,
                recomputed_metrics,
                f"session {slot} frame {frame_index}",
            )
            require(
                metrics_passed == bool(recomputed_metrics["passed"]),
                f"session {slot} frame pass/fail changed on independent recomputation",
            )
            require(
                frame.get("status") == ("passed" if metrics_passed else "failed"),
                f"session {slot} frame status disagrees with result metrics",
            )
            checks = mapping(metrics.get("checks"), f"session {slot} result checks")
            require(
                set(checks)
                == {"roi_pixels", "valid_pixels", "valid_fraction", "relative_mad", "central80_spread", "relative_error"},
                f"session {slot} result check set is incomplete",
            )
            check_values = {
                key: exact_bool(value, f"session {slot} checks.{key}")
                for key, value in checks.items()
            }
            require(metrics_passed == all(check_values.values()), f"session {slot} passed flag disagrees with checks")
            failures = sequence(metrics.get("failure_checks"), f"session {slot} failure_checks")
            require(
                set(str(item) for item in failures)
                == {key for key, value in check_values.items() if not value},
                f"session {slot} failure_checks disagrees with checks",
            )
            require(
                metrics.get("distance_definition") == "physical_left_camera_to_roi_euclidean_m",
                f"session {slot} used a different distance definition",
            )
            frame_thresholds = mapping(metrics.get("thresholds"), f"session {slot} frame thresholds")
            frame_threshold_expectations = {
                "minimum_roi_pixels": 400,
                "minimum_valid_pixels": 200,
                "minimum_valid_fraction": 0.35,
                "maximum_relative_mad": 0.10,
                "maximum_central80_spread_over_median": 0.25,
                "maximum_relative_error": 0.10,
            }
            for threshold_name, expected_value in frame_threshold_expectations.items():
                close(
                    frame_thresholds.get(threshold_name),
                    expected_value,
                    f"session {slot} frame threshold {threshold_name}",
                )
            known = finite_number(metrics.get("known_distance_m"), f"session {slot} frame known distance")
            close(known, aggregate.get("known_distance_m"), f"session {slot} frame/aggregate known distance")
            roi_pixels = exact_integer(
                metrics.get("roi_pixels_after_boundary_erosion"),
                f"session {slot} ROI pixels after erosion",
            )
            require(roi_pixels > 0, f"session {slot} ROI has no usable pixels")
            require(
                roi_pixels <= roi_width * roi_height,
                f"session {slot} eroded ROI pixel count exceeds the selected ROI",
            )
            valid_pixels = exact_integer(metrics.get("valid_pixels"), f"session {slot} valid pixels")
            require(0 <= valid_pixels <= roi_pixels, f"session {slot} valid-pixel count is impossible")
            valid_fraction = finite_number(metrics.get("valid_fraction"), f"session {slot} valid fraction")
            close(valid_fraction, valid_pixels / roi_pixels, f"session {slot} recomputed valid fraction", tolerance=1e-9)
            if metrics_passed:
                estimate = finite_number(metrics.get("estimated_distance_m"), f"session {slot} frame estimate")
                relative_error = finite_number(metrics.get("relative_error"), f"session {slot} frame relative error")
                close(relative_error, abs(estimate - known) / known, f"session {slot} frame recomputed error", tolerance=1e-9)
                require(roi_pixels >= 400, f"session {slot} passing frame has fewer than 400 ROI pixels")
                require(valid_pixels >= 200, f"session {slot} passing frame has fewer than 200 valid pixels")
                require(valid_fraction >= 0.35, f"session {slot} passing frame valid fraction is below 35%")
                require(relative_error <= 0.10, f"session {slot} passing frame error exceeds 10%")
                require(finite_number(metrics.get("relative_mad"), f"session {slot} frame relative MAD") <= 0.10, f"session {slot} passing frame relative MAD exceeds 10%")
                require(finite_number(metrics.get("central80_spread_over_median"), f"session {slot} frame central-80 spread") <= 0.25, f"session {slot} passing frame central-80 spread exceeds 25%")
                passing_metrics.append(recomputed_metrics)
            else:
                estimate_value = metrics.get("estimated_distance_m")
                error_value = metrics.get("relative_error")
                if estimate_value is not None or error_value is not None:
                    estimate = finite_number(estimate_value, f"session {slot} failed-frame estimate")
                    relative_error = finite_number(error_value, f"session {slot} failed-frame relative error")
                    close(
                        relative_error,
                        abs(estimate - known) / known,
                        f"session {slot} failed-frame recomputed error",
                        tolerance=1e-9,
                    )

        require(len(set(roi_source_frames)) == 1, f"session {slot} result files name different ROI source frames")
        roi_source_frame = roi_source_frames[0]
        require(1 <= roi_source_frame <= 3, f"session {slot} ROI source frame is outside 1-3")
        expected_roi_source = max(sharpness_by_frame, key=lambda pair: pair[0])[1]
        require(roi_source_frame == expected_roi_source, f"session {slot} ROI source is not the sharpest captured frame")
        require(all(item == shared_rois[0] for item in shared_rois), f"session {slot} did not use exactly one ROI for all three frames")
        require(len(set(evaluated_timestamps)) == 1, f"session {slot} frames were not evaluated as one shared-ROI batch")

        passing_count = exact_integer(aggregate.get("passing_frames"), f"validated YAML {slot}.passing_frames")
        require(2 <= passing_count <= 3, f"validated YAML {slot} must have two or three passing frames")
        require(len(passing_metrics) == passing_count, f"validated YAML {slot} passing-frame count is not reproducible")
        require(exact_integer(aggregate.get("total_frames"), f"validated YAML {slot}.total_frames") == 3, f"validated YAML {slot} total-frame count is wrong")
        require(exact_integer(summary.get("passing_frames"), f"session {slot} summary passing frames") == passing_count, f"session {slot} summary passing count mismatch")

        observed_min_pixels = min(exact_integer(item.get("valid_pixels"), "valid_pixels") for item in passing_metrics)
        observed_min_fraction = min(finite_number(item.get("valid_fraction"), "valid_fraction") for item in passing_metrics)
        observed_max_mad = max(finite_number(item.get("relative_mad"), "relative_mad") for item in passing_metrics)
        observed_max_spread = max(finite_number(item.get("central80_spread_over_median"), "central80 spread") for item in passing_metrics)
        passing_estimates = [
            finite_number(item.get("estimated_distance_m"), "passing estimated distance")
            for item in passing_metrics
        ]
        aggregate_estimate = finite_number(
            aggregate.get("estimated_distance_m"), f"validated YAML {slot} estimated distance"
        )
        recomputed_estimate = float(statistics.median(passing_estimates))
        close(aggregate_estimate, recomputed_estimate, f"validated YAML {slot} aggregate estimate")
        aggregate_known = finite_number(
            aggregate.get("known_distance_m"), f"validated YAML {slot} known distance"
        )
        close(
            aggregate.get("relative_error"),
            abs(aggregate_estimate - aggregate_known) / aggregate_known,
            f"validated YAML {slot} aggregate relative error",
            tolerance=1e-9,
        )
        recomputed_span = (
            max(passing_estimates) - min(passing_estimates)
        ) / max(recomputed_estimate, 1e-9)
        close(
            aggregate.get("passing_estimate_span_over_median"),
            recomputed_span,
            f"validated YAML {slot} passing estimate span",
        )
        require(exact_integer(aggregate.get("minimum_valid_pixels"), f"validated YAML {slot} minimum_valid_pixels") == observed_min_pixels, f"validated YAML {slot} minimum valid pixels is not reproducible")
        close(aggregate.get("minimum_valid_fraction"), observed_min_fraction, f"validated YAML {slot} minimum valid fraction")
        close(aggregate.get("maximum_relative_mad"), observed_max_mad, f"validated YAML {slot} maximum relative MAD")
        close(aggregate.get("maximum_central80_spread_over_median"), observed_max_spread, f"validated YAML {slot} maximum central-80 spread")

    require(len(all_combined_hashes) == 9, "final validation must bind exactly nine synchronized images")
    require(len(set(all_combined_hashes)) == 9, "all nine synchronized-image hashes must be different")
    verify_checksums(session_dir, required_files)
    return required_files


def verify(validated_path: Path) -> dict[str, Any]:
    validated_path = validated_path.expanduser().resolve(strict=True)
    require(validated_path.is_file(), "validated YAML path is not a file")
    session_dir = validated_path.parent.resolve()
    validated, validated_raw = load_yaml(validated_path, "validated YAML")
    validated_sha = sha256_bytes(validated_raw)

    verify_calibration_shape(validated)
    require(exact_bool(validated.get("valid"), "validated YAML valid"), "validated YAML valid is not true")
    require(validated.get("camera_model") == "GS130W", "camera_model must be GS130W")
    require(validated.get("physical_left_view") == "bottom", "physical_left_view must be bottom")
    require(validated.get("physical_right_view") == "top", "physical_right_view must be top")
    require(validated.get("combined_layout") == "vertical", "combined_layout must be vertical")
    require(validated.get("runtime_rotation") == "ccw90", "runtime_rotation must be ccw90")

    validation = mapping(validated.get("validation"), "validated YAML validation")
    distance_validation = mapping(validation.get("distance_validation"), "validated YAML distance_validation")
    require(distance_validation.get("schema_version") == EXPECTED_SCHEMA, "distance-validation schema mismatch")
    verify_thresholds(distance_validation.get("thresholds"), "validated YAML strict thresholds")
    source_candidate_sha = sha256_value(
        distance_validation.get("source_candidate_sha256"),
        "distance_validation.source_candidate_sha256",
    )

    pristine_path = safe_child(
        session_dir,
        distance_validation.get("pristine_candidate_filename"),
        "pristine candidate filename",
        exact_name="candidate_pristine.yaml",
    )
    pristine, pristine_raw = load_yaml(pristine_path, "pristine candidate YAML")
    require(sha256_bytes(pristine_raw) == source_candidate_sha, "pristine candidate SHA-256 binding failed")
    require(
        source_candidate_sha == APPROVED_CANDIDATE_SHA256,
        "source candidate is not the approved 2026-08-05 GS130W candidate",
    )
    verify_calibration_shape(pristine)
    require(not exact_bool(pristine.get("valid"), "pristine candidate valid"), "pristine candidate must remain valid:false")
    pristine_validation = mapping(pristine.get("validation"), "pristine candidate validation")
    require(sequence(pristine_validation.get("known_distance_results"), "pristine known-distance results") == [], "pristine candidate already contains known-distance results")

    require(
        immutable_payload(validated) == immutable_payload(pristine),
        "calibration payload changed outside valid/known-distance validation fields",
    )
    baseline = finite_number(validated.get("baseline_m"), "validated baseline_m")
    close(baseline, pristine.get("baseline_m"), "validated/pristine baseline")
    require(
        abs(baseline - EXPECTED_BASELINE_M) <= BASELINE_ABSOLUTE_TOLERANCE_M,
        f"baseline_m {baseline:.12f} is not the approved GS130W baseline",
    )

    metric_names = (
        "stereo_rms_px",
        "left_rms_px",
        "right_rms_px",
        "median_epipolar_error_px",
    )
    metrics: dict[str, float] = {}
    for name in metric_names:
        value = finite_number(validation.get(name), f"validation.{name}")
        close(value, pristine_validation.get(name), f"validated/pristine {name}")
        require(0.0 <= value <= 1.0, f"validation.{name} exceeds 1.0 px")
        metrics[name] = value
    used_pairs = exact_integer(validation.get("used_pair_count"), "validation.used_pair_count")
    require(used_pairs == 30, "validation.used_pair_count must be exactly 30")
    require(exact_integer(pristine_validation.get("used_pair_count"), "pristine used_pair_count") == 30, "pristine candidate was not calibrated from exactly 30 pairs")

    raw_results = sequence(validation.get("known_distance_results"), "known_distance_results")
    require(len(raw_results) == 3, "known_distance_results must contain exactly three slots")
    results = [mapping(item, f"known_distance_results[{index}]") for index, item in enumerate(raw_results)]
    require([item.get("slot") for item in results] == list(EXPECTED_SLOTS), "distance slots must be ordered exactly as 1m, 3m, 5m (P1/P2/P3)")
    known_distances: list[float] = []
    report_points: list[dict[str, Any]] = []
    for slot, item in zip(EXPECTED_SLOTS, results):
        close(item.get("nominal_distance_m"), EXPECTED_NOMINAL_M[slot], f"{slot} nominal_distance_m")
        known = finite_number(item.get("known_distance_m"), f"{slot} known_distance_m")
        estimated = finite_number(item.get("estimated_distance_m"), f"{slot} estimated_distance_m")
        lower, upper = KNOWN_DISTANCE_RANGES_M[slot]
        require(lower <= known <= upper, f"{slot} measured distance is outside {lower}-{upper} m")
        require(estimated > 0.0, f"{slot} estimated distance must be positive")
        recorded_error = finite_number(item.get("relative_error"), f"{slot} relative_error")
        recomputed_error = abs(estimated - known) / known
        close(recorded_error, recomputed_error, f"{slot} relative error", tolerance=1e-9)
        require(recorded_error <= 0.10, f"{slot} relative error exceeds 10%")
        require(exact_bool(item.get("passed"), f"{slot} passed"), f"{slot} did not pass")
        require(exact_integer(item.get("minimum_valid_pixels"), f"{slot} minimum_valid_pixels") >= 200, f"{slot} has fewer than 200 valid pixels")
        require(finite_number(item.get("minimum_valid_fraction"), f"{slot} minimum_valid_fraction") >= 0.35, f"{slot} valid fraction is below 35%")
        require(finite_number(item.get("maximum_relative_mad"), f"{slot} maximum_relative_mad") <= 0.10, f"{slot} relative MAD exceeds 10%")
        require(finite_number(item.get("maximum_central80_spread_over_median"), f"{slot} central80 spread") <= 0.25, f"{slot} central-80 spread exceeds 25%")
        require(finite_number(item.get("passing_estimate_span_over_median"), f"{slot} passing estimate span") <= 0.03, f"{slot} three-frame consistency exceeds 3%")
        known_distances.append(known)
        report_points.append(
            {
                "slot": slot,
                "point": {"1m": "P1", "3m": "P2", "5m": "P3"}[slot],
                "known_distance_m": known,
                "estimated_distance_m": estimated,
                "relative_error": recorded_error,
                "passing_frames": exact_integer(item.get("passing_frames"), f"{slot} passing_frames"),
            }
        )

    for first, second in zip(known_distances, known_distances[1:]):
        separation = abs(second - first) / max(first, second)
        require(separation >= 0.05, "adjacent measured distances differ by less than 5%")

    manifest_path = safe_child(
        session_dir,
        distance_validation.get("session_manifest_filename"),
        "session manifest filename",
        exact_name="session.json",
    )
    manifest, _ = load_json(manifest_path, "session manifest")
    session_id = str(distance_validation.get("session_id") or "")
    require(session_id != "", "distance_validation.session_id is empty")
    require(manifest.get("session_id") == session_id, "validated YAML and manifest session IDs differ")
    archived_validator = safe_child(
        session_dir,
        distance_validation.get("validator_script_filename"),
        "validator script filename",
        exact_name="validator_script.py",
    )
    archived_validator_raw = read_bytes_limited(
        archived_validator, MAX_JSON_BYTES, "archived validator script"
    )
    recorded_validator_sha = sha256_value(
        distance_validation.get("validator_script_sha256"),
        "validator_script_sha256",
    )
    require(
        sha256_bytes(archived_validator_raw) == recorded_validator_sha,
        "archived validator_script.py does not match validator_script_sha256",
    )
    require(
        APPROVED_VALIDATOR_SCRIPT_SHA256 != ""
        and recorded_validator_sha == APPROVED_VALIDATOR_SCRIPT_SHA256,
        "validator_script.py is not the approved final validation tool",
    )
    manifest_runtime_archives = mapping(
        manifest.get("runtime_source_archives"), "manifest runtime_source_archives"
    )
    require(
        set(manifest_runtime_archives) == {"depth", "calibration", "gs130w"},
        "manifest must bind exactly the depth, calibration and GS130W runtime sources",
    )
    manifest_depth_source = mapping(
        manifest_runtime_archives.get("depth"), "manifest depth runtime source"
    )
    manifest_calibration_source = mapping(
        manifest_runtime_archives.get("calibration"),
        "manifest calibration runtime source",
    )
    manifest_gs130w_source = mapping(
        manifest_runtime_archives.get("gs130w"),
        "manifest GS130W runtime source",
    )
    yaml_runtime_archives = mapping(
        distance_validation.get("runtime_source_archives"),
        "distance_validation runtime_source_archives",
    )
    require(
        set(yaml_runtime_archives) == {"depth", "calibration", "gs130w"},
        "validated YAML runtime source set is incomplete",
    )
    for runtime_key, manifest_record in (
        ("depth", manifest_depth_source),
        ("calibration", manifest_calibration_source),
        ("gs130w", manifest_gs130w_source),
    ):
        yaml_record = mapping(
            yaml_runtime_archives.get(runtime_key),
            f"distance_validation {runtime_key} runtime source",
        )
        require(
            set(yaml_record) == {"file", "sha256"},
            f"distance_validation {runtime_key} runtime source fields are invalid",
        )
        require(
            yaml_record.get("file") == manifest_record.get("file")
            and yaml_record.get("sha256") == manifest_record.get("sha256"),
            f"validated YAML and manifest disagree about {runtime_key} runtime source",
        )
    archived_runtime_depth = safe_child(
        session_dir,
        manifest_depth_source.get("file"),
        "runtime depth filename",
        exact_name="runtime_depth.py",
    )
    recorded_runtime_depth_sha = sha256_value(
        manifest_depth_source.get("sha256"),
        "runtime_depth_sha256",
    )
    require(
        recorded_runtime_depth_sha == APPROVED_RUNTIME_DEPTH_SHA256
        and sha256_file(archived_runtime_depth) == APPROVED_RUNTIME_DEPTH_SHA256,
        "runtime_depth.py is not the approved deployed source",
    )
    archived_runtime_calibration = safe_child(
        session_dir,
        manifest_calibration_source.get("file"),
        "runtime calibration filename",
        exact_name="runtime_calibration.py",
    )
    recorded_runtime_calibration_sha = sha256_value(
        manifest_calibration_source.get("sha256"),
        "runtime_calibration_sha256",
    )
    require(
        recorded_runtime_calibration_sha == APPROVED_RUNTIME_CALIBRATION_SHA256
        and sha256_file(archived_runtime_calibration)
        == APPROVED_RUNTIME_CALIBRATION_SHA256,
        "runtime_calibration.py is not the approved deployed source",
    )
    archived_runtime_gs130w = safe_child(
        session_dir,
        manifest_gs130w_source.get("file"),
        "runtime GS130W filename",
        exact_name="runtime_gs130w.py",
    )
    recorded_runtime_gs130w_sha = sha256_value(
        manifest_gs130w_source.get("sha256"),
        "runtime_gs130w_sha256",
    )
    require(
        recorded_runtime_gs130w_sha == APPROVED_RUNTIME_GS130W_SHA256
        and sha256_file(archived_runtime_gs130w) == APPROVED_RUNTIME_GS130W_SHA256,
        "runtime_gs130w.py is not the approved deployed source",
    )
    recompute_context = load_archived_runtime(
        archived_runtime_depth,
        archived_runtime_calibration,
        pristine,
    )
    verify_session(
        manifest,
        session_dir,
        validated_path,
        validated_sha,
        source_candidate_sha,
        results,
        archived_validator.name,
        recorded_validator_sha,
        recompute_context,
        additional_required_files=(
            "validator_script.py",
            "runtime_depth.py",
            "runtime_calibration.py",
            "runtime_gs130w.py",
            "session_before_validation.json",
            "audit_before_validation.jsonl",
            "finalize_transaction.json",
        ),
    )

    manifest_before = sha256_value(
        distance_validation.get("session_manifest_sha256_before_validation"),
        "session_manifest_sha256_before_validation",
    )
    manifest_snapshot_path = safe_child(
        session_dir,
        distance_validation.get("session_manifest_before_validation_filename"),
        "pre-validation manifest snapshot filename",
        exact_name="session_before_validation.json",
    )
    manifest_snapshot, manifest_snapshot_raw = load_json(
        manifest_snapshot_path, "pre-validation manifest snapshot"
    )
    require(
        sha256_bytes(manifest_snapshot_raw) == manifest_before,
        "session_before_validation.json does not match its recorded SHA-256",
    )
    require(manifest_snapshot.get("session_id") == session_id, "pre-validation manifest session mismatch")
    require(
        manifest_snapshot.get("candidate_sha256") == source_candidate_sha,
        "pre-validation manifest candidate mismatch",
    )
    require(
        manifest_snapshot.get("slots") == manifest.get("slots"),
        "pre-validation and final manifests disagree about validated slots",
    )
    require(
        manifest_before != sha256_file(manifest_path),
        "manifest unexpectedly remained in its pre-validation state",
    )

    audit_path = safe_child(
        session_dir,
        distance_validation.get("audit_log_filename"),
        "audit log filename",
        exact_name="audit.jsonl",
    )
    audit_before = sha256_value(
        distance_validation.get("audit_log_sha256_before_validation"),
        "audit_log_sha256_before_validation",
    )
    audit_snapshot_path = safe_child(
        session_dir,
        distance_validation.get("audit_log_before_validation_filename"),
        "pre-validation audit snapshot filename",
        exact_name="audit_before_validation.jsonl",
    )
    audit_snapshot_raw = read_bytes_limited(
        audit_snapshot_path, MAX_JSON_BYTES, "pre-validation audit snapshot"
    )
    require(
        sha256_bytes(audit_snapshot_raw) == audit_before,
        "audit_before_validation.jsonl does not match its recorded SHA-256",
    )
    verify_audit_log(
        audit_path,
        session_id,
        source_candidate_sha,
        validated_sha,
        validated_path.name,
        audit_before,
    )
    return {
        "ok": True,
        "validated_yaml": str(validated_path),
        "validated_yaml_sha256": validated_sha,
        "pristine_candidate": str(pristine_path),
        "source_candidate_sha256": source_candidate_sha,
        "session_id": session_id,
        "baseline_m": baseline,
        "used_pair_count": used_pairs,
        "metrics": metrics,
        "points": report_points,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only strict verifier for a GS130W three-distance calibration export."
    )
    parser.add_argument("--validated", required=True, type=Path, help="Path to the exported valid:true YAML")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON object")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify(args.validated)
    except (OSError, VerificationError) as exc:
        message = str(exc)
        if args.json:
            print(json.dumps({"ok": False, "error": message}, ensure_ascii=True, separators=(",", ":")))
        else:
            print(f"VERIFICATION FAILED: {message}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    else:
        print("STRICT_VALIDATION_OK")
        print(f"validated_sha256={report['validated_yaml_sha256']}")
        print(f"candidate_sha256={report['source_candidate_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
