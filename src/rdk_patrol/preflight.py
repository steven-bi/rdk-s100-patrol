from __future__ import annotations

"""Read-only deployment checks for the development PC and RDK S100."""

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from .config import load_config, load_yaml, resolve_path


EXPECTED_HBM_SHA256 = (
    "bdd41c37a92dbbc73fb6c9ba87db81994bbcfdf9b93e1e4aee856b23376b5612"
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreflightReport:
    config_path: str
    board_checks: bool
    results: tuple[CheckResult, ...]

    @property
    def errors(self) -> int:
        return sum(result.status == "error" for result in self.results)

    @property
    def warnings(self) -> int:
        return sum(result.status == "warning" for result in self.results)

    @property
    def ok(self) -> bool:
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rdk-patrol-preflight/v1",
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "config_path": self.config_path,
            "board_checks": self.board_checks,
            "results": [asdict(result) for result in self.results],
        }


def run_preflight(
    config_path: str | Path,
    *,
    board_checks: bool = False,
    data_dir: str | Path | None = None,
) -> PreflightReport:
    path = Path(config_path).resolve()
    results: list[CheckResult] = []
    try:
        config = load_config(path)
    except Exception as exc:
        results.append(
            CheckResult(
                "configuration",
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )
        return PreflightReport(str(path), bool(board_checks), tuple(results))
    results.append(CheckResult("configuration", "ok", "system.yaml valid"))

    _check_file(
        results,
        "HBM model",
        resolve_path(path, config["model"]["hbm_path"]),
        minimum_bytes=1_000_000,
        expected_sha256=EXPECTED_HBM_SHA256,
    )
    _check_file(
        results,
        "model metadata",
        resolve_path(path, config["model"]["metadata_path"]),
    )
    points_path = resolve_path(path, config["localization"]["points_path"])
    _check_file(results, "point configuration", points_path)
    stereo_path = resolve_path(path, config["stereo"]["calibration_path"])
    _check_file(results, "stereo calibration", stereo_path)
    _check_file(
        results,
        "CJK font",
        resolve_path(path, config["alarms"]["font_path"]),
        minimum_bytes=100_000,
    )

    _check_points(results, points_path, path)
    _check_stereo(results, stereo_path, config)
    _check_minimax(results, config)
    _check_storage(results, path, config, data_dir=data_dir)
    _check_imports(results, ("numpy", "cv2", "yaml", "PIL"))
    _check_apriltag_backend(results)
    if board_checks:
        _check_imports(
            results,
            ("rclpy", "sensor_msgs.msg", "hbm_runtime"),
            prefix="board dependency",
        )
        tros = Path("/opt/tros/humble/setup.bash")
        if os.name != "nt" and not tros.is_file():
            results.append(
                CheckResult(
                    "TROS setup",
                    "error",
                    f"missing {tros}",
                )
            )
        elif os.name != "nt":
            results.append(CheckResult("TROS setup", "ok", str(tros)))
    return PreflightReport(str(path), bool(board_checks), tuple(results))


def _check_file(
    results: list[CheckResult],
    name: str,
    path: Path,
    *,
    minimum_bytes: int = 1,
    expected_sha256: str | None = None,
) -> None:
    if not path.is_file():
        results.append(CheckResult(name, "error", f"missing file: {path}"))
        return
    size = path.stat().st_size
    if size < int(minimum_bytes):
        results.append(
            CheckResult(name, "error", f"file is unexpectedly small: {path}")
        )
        return
    details: dict[str, Any] = {"path": str(path), "bytes": int(size)}
    if expected_sha256 is not None:
        digest = _sha256(path)
        details["sha256"] = digest
        if digest.lower() != expected_sha256.lower():
            results.append(
                CheckResult(
                    name,
                    "error",
                    "SHA256 does not match the accepted HBM artifact",
                    details,
                )
            )
            return
    results.append(CheckResult(name, "ok", "file present", details))


def _check_points(
    results: list[CheckResult],
    points_path: Path,
    system_config_path: Path,
) -> None:
    if not points_path.is_file():
        return
    try:
        document = load_yaml(points_path)
        points = document.get("points") or []
        if not isinstance(points, list) or not points:
            raise ValueError("points list is empty")
    except Exception as exc:
        results.append(
            CheckResult(
                "point readiness",
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )
        return
    warnings: list[str] = []
    for item in points:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        point_id = str(item.get("point_id") or "<unnamed>")
        tag = item.get("tag") if isinstance(item.get("tag"), dict) else {}
        if tag.get("registration_required", False):
            warnings.append(f"{point_id}: AprilTag现场登记未完成")
        legacy = item.get("legacy")
        if isinstance(legacy, dict):
            for key in ("vehicle_config", "trash_config"):
                relative = legacy.get(key)
                if not relative:
                    continue
                legacy_path = (points_path.parent / str(relative)).resolve()
                if not legacy_path.is_file():
                    warnings.append(f"{point_id}: 缺少旧视觉指纹文件 {legacy_path}")
        fingerprints = (
            item.get("fingerprints")
            or item.get("visual_fingerprints")
            or item.get("anchors")
        )
        capabilities = set(str(value) for value in item.get("capabilities") or [])
        if "trash_review" in capabilities and not fingerprints:
            # A legacy trash file may contain anchors; the localization loader
            # decides whether they are usable. This warning intentionally asks
            # for an onsite capture rather than pretending an empty list works.
            warnings.append(f"{point_id}: 垃圾桶视觉指纹尚未现场采集")
        if (
            "vehicle_parking" in capabilities
            and not tag.get("registration_required", False)
            and not item.get("roi_points_tag_m")
            and not item.get("rois_3d")
            and not bool(item.get("rois_coplanar_with_tag", False))
        ):
            warnings.append(
                f"{point_id}: Tag登记缺少安全的三维/共面ROI映射，车辆区域将跳过"
            )
    if warnings:
        results.append(
            CheckResult(
                "point readiness",
                "warning",
                "；".join(warnings),
                {"point_count": len(points)},
            )
        )
    else:
        results.append(
            CheckResult(
                "point readiness",
                "ok",
                "AprilTag/visual registration present",
                {"point_count": len(points)},
            )
        )


def _check_stereo(
    results: list[CheckResult],
    path: Path,
    config: dict[str, Any],
) -> None:
    if not path.is_file():
        return
    try:
        document = load_yaml(path)
    except Exception as exc:
        results.append(
            CheckResult(
                "stereo readiness",
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )
        return
    nested = document.get("calibration")
    calibration = nested if isinstance(nested, dict) else document
    view_mapping = document.get("view_mapping")
    if not isinstance(view_mapping, dict):
        view_mapping = {}
    physical_left = str(
        calibration.get("physical_left_view")
        or document.get("physical_left_view")
        or view_mapping.get("physical_left_view")
        or ""
    )
    physical_right = str(
        calibration.get("physical_right_view")
        or document.get("physical_right_view")
        or view_mapping.get("physical_right_view")
        or ""
    )
    mapping_ok = {physical_left, physical_right} == {"top", "bottom"}
    valid = bool(calibration.get("valid"))
    runtime_rotation = str(
        calibration.get("runtime_rotation")
        or document.get("runtime_rotation")
        or "none"
    )
    configured_rotation = str(config["camera"].get("rotation") or "none")
    rotation_ok = _normalized_rotation(runtime_rotation) == _normalized_rotation(
        configured_rotation
    )
    if not valid or not mapping_ok:
        results.append(
            CheckResult(
                "stereo readiness",
                "warning",
                "GS130W物理左右目或双目标定未完成；明火仍报警但距离显示“不可用”",
            )
        )
    elif not rotation_ok:
        results.append(
            CheckResult(
                "stereo readiness",
                "warning",
                "标定图像旋转与运行时相机旋转不一致；明火仍报警但测距已安全禁用",
                {
                    "calibration_rotation": runtime_rotation,
                    "camera_rotation": configured_rotation,
                },
            )
        )
    else:
        results.append(
            CheckResult(
                "stereo readiness",
                "ok",
                "physical view mapping, rotation and calibration marked valid",
                {
                    "physical_left_view": physical_left,
                    "physical_right_view": physical_right,
                    "runtime_rotation": runtime_rotation,
                },
            )
        )


def _check_apriltag_backend(results: list[CheckResult]) -> None:
    try:
        import cv2

        aruco = getattr(cv2, "aruco", None)
        if aruco is None or not hasattr(aruco, "DICT_APRILTAG_36h11"):
            raise AttributeError("cv2.aruco.DICT_APRILTAG_36h11 is unavailable")
    except Exception as exc:
        results.append(
            CheckResult(
                "AprilTag backend",
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )
    else:
        results.append(
            CheckResult(
                "AprilTag backend",
                "ok",
                "OpenCV tag36h11 support is available",
            )
        )


def _normalized_rotation(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "")
    return {
        "": "none",
        "0": "none",
        "none": "none",
        "cw90": "cw90",
        "90cw": "cw90",
        "ccw90": "ccw90",
        "90ccw": "ccw90",
        "rot180": "rot180",
        "rotate180": "rot180",
        "180": "rot180",
    }.get(normalized, f"unknown:{normalized}")


def _check_minimax(
    results: list[CheckResult],
    config: dict[str, Any],
) -> None:
    if not bool(config["minimax"].get("enabled", True)):
        results.append(
            CheckResult(
                "MiniMax",
                "warning",
                "MiniMax disabled; garbage review jobs will not be classified",
            )
        )
        return
    env_name = str(config["minimax"].get("api_key_env") or "MINIMAX_API_KEY")
    if os.environ.get(env_name):
        results.append(
            CheckResult("MiniMax", "ok", f"{env_name} is configured")
        )
    else:
        results.append(
            CheckResult(
                "MiniMax",
                "warning",
                f"{env_name} is not set; review images remain in the durable retry queue",
            )
        )


def _check_storage(
    results: list[CheckResult],
    config_path: Path,
    config: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
) -> None:
    output = (
        Path(data_dir).resolve() / "recordings"
        if data_dir is not None
        else resolve_path(config_path, config["recording"]["output_dir"])
    )
    probe = output
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        results.append(
            CheckResult("video storage", "error", f"{type(exc).__name__}: {exc}")
        )
        return
    free_gb = usage.free / (1024**3)
    minimum = float(config["recording"].get("minimum_free_gb", 20.0))
    if free_gb < minimum:
        status = "error"
    elif free_gb < 512.0:
        status = "warning"
    else:
        status = "ok"
    message = (
        f"{free_gb:.1f} GiB free; configured hard minimum {minimum:.1f} GiB. "
        "For two 15 FPS streams retained 72 hours, use at least a 512 GB "
        "high-endurance external volume and verify measured bitrate onsite."
    )
    results.append(
        CheckResult(
            "video storage",
            status,
            message,
            {"path": str(output), "free_gib": round(free_gb, 2)},
        )
    )


def _check_imports(
    results: list[CheckResult],
    modules: Iterable[str],
    *,
    prefix: str = "Python dependency",
) -> None:
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            results.append(
                CheckResult(
                    f"{prefix}: {module}",
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            results.append(
                CheckResult(f"{prefix}: {module}", "ok", "import succeeded")
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
