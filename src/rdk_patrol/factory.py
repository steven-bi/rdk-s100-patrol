from __future__ import annotations

"""Construct the production dependency graph from ``system.yaml``."""

import inspect
from pathlib import Path
from typing import Any, Mapping

from .alarms import AlarmPublisher, AlarmRepository, PillowEvidenceRenderer
from .application import RuntimeApplication
from .async_workers import AsyncDepthEstimator, AsyncPointResolver
from .config import load_config, load_yaml, resolve_path
from .delivery import DisabledDingTalkOutbox
from .engines import FlameEngine, NightPeopleEngine, ParkingEngine
from .garbage_runtime import (
    GarbagePointCoordinator,
    GarbageReviewWorkerThread,
)
from .health import HealthFileReporter, HealthRegistry
from .inference import (
    ClassAwareTracker,
    HbmRuntimeBackend,
    UnifiedInference,
    YoloDecoder,
)
from .io import FrameHub
from .navigation import NavigationBridge
from .processor import PatrolFrameProcessor
from .recording import DualStreamRecorder, OpenCVWriterFactory
from .recording_worker import AsyncDualStreamRecorder
from .review import (
    GarbageReviewCollector,
    GarbageReviewWorker,
    MiniMaxVisionClient,
    PersistentReviewQueue,
)
from .ros2_source import Ros2FrameSource
from .video_source import (
    OpenCvVideoSource,
    VideoFrameHubCamera,
)
from .web import ReadOnlyAlarmServer


def build_application(
    config_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    source: str = "ros2",
    video_path: str | Path | None = None,
    topic_override: str | None = None,
    message_type_override: str | None = None,
    web_enabled_override: bool | None = None,
    recording_enabled_override: bool | None = None,
) -> RuntimeApplication:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    data_root = None if data_dir is None else Path(data_dir).resolve()
    if data_root is not None:
        data_root.mkdir(parents=True, exist_ok=True)

    health = HealthRegistry(version="1.0.0")
    stereo_path = resolve_path(
        config_file, config["stereo"]["calibration_path"]
    )
    stereo_estimator = _stereo_estimator(config, stereo_path)
    (
        detection_view,
        auxiliary_view,
        stereo_ready,
        stereo_reason,
    ) = _effective_stereo_views(config, stereo_estimator)
    frame_hub, camera_source, source_exhausted = _camera(
        config,
        source=source,
        video_path=video_path,
        topic_override=topic_override,
        message_type_override=message_type_override,
        detection_view_override=detection_view,
        auxiliary_view_override=auxiliary_view,
    )
    inference = _inference(config_file, config)
    parking, night, flame = _engines(config)

    alarm_root = _data_path(
        data_root,
        "alarms",
        config_file,
        config["alarms"]["output_dir"],
    )
    font_path = resolve_path(config_file, config["alarms"]["font_path"])
    renderer = PillowEvidenceRenderer(font_path=font_path)
    repository = AlarmRepository(
        alarm_root,
        renderer=renderer,
        jpeg_quality=int(config["alarms"].get("jpeg_quality", 92)),
    )
    publisher = AlarmPublisher(
        repository,
        cooldown_seconds={
            "车辆违停": float(config["rules"]["vehicle"]["cooldown_seconds"]),
            "垃圾桶已满": float(
                config["rules"]["trash"]["point_cooldown_seconds"]
            ),
            "明火报警": float(config["rules"]["flame"]["repeat_alarm_seconds"]),
            "夜间人员逗留": 0.0,
            "夜间人群聚集": 0.0,
        },
    )

    points_path = resolve_path(
        config_file, config["localization"]["points_path"]
    )
    point_document = load_yaml(points_path)
    capabilities = _point_capabilities(point_document)
    calibration = (
        None if stereo_estimator is None else stereo_estimator.calibration
    )
    point_worker = _point_worker(
        config,
        points_path,
        camera_matrix=(
            calibration.left_camera_matrix
            if stereo_ready and calibration is not None
            else None
        ),
        distortion=(
            calibration.left_distortion
            if stereo_ready and calibration is not None
            else None
        ),
    )
    depth_worker = _depth_worker(
        config,
        stereo_estimator if stereo_ready else None,
    )
    if stereo_ready:
        health.heartbeat(
            "stereo",
            message="双目标定、旋转和物理左右目映射有效",
            metrics={
                "physical_left_view": detection_view,
                "physical_right_view": auxiliary_view,
            },
        )
    else:
        health.update(
            "stereo",
            state="degraded",
            message=stereo_reason,
            metrics={"distance_available": False},
        )

    review_root = _data_path(
        data_root,
        "review_queue",
        config_file,
        config["minimax"]["queue_dir"],
    )
    review_queue = PersistentReviewQueue(
        review_root,
        jpeg_quality=int(config["minimax"].get("jpeg_quality", 85)),
    )
    review_collector = GarbageReviewCollector(review_queue)
    garbage_coordinator = GarbagePointCoordinator(
        review_collector,
        capabilities,
    )
    review_thread = None
    if bool(config["minimax"].get("enabled", True)):
        classifier = MiniMaxVisionClient.from_system_config(config)
        review_thread = GarbageReviewWorkerThread(
            GarbageReviewWorker(
                review_queue,
                classifier,
                publisher,
                retry_delay_seconds=float(
                    config["minimax"].get("retry_interval_seconds", 10.0)
                ),
            )
        )

    recorder_worker = _recorder_worker(
        config_file,
        config,
        data_root,
        recording_enabled_override,
    )
    health_path = (
        data_root / "state" / "health.json"
        if data_root is not None
        else resolve_path(config_file, config["system"]["health_path"])
    )
    health_reporter = HealthFileReporter(
        health,
        health_path,
        interval_seconds=float(
            config["system"].get("status_interval_seconds", 1.0)
        ),
    )
    web_server = _web_server(
        config,
        repository,
        health,
        web_enabled_override,
    )
    processor = PatrolFrameProcessor(
        inference=inference,
        parking_engine=parking,
        night_engine=night,
        flame_engine=flame,
        alarm_publisher=publisher,
        delivery_outbox=DisabledDingTalkOutbox(),
        point_worker=point_worker,
        depth_worker=depth_worker,
        garbage_coordinator=garbage_coordinator,
        recorder_worker=recorder_worker,
        health=health,
        target_fps=float(config["system"]["target_fps"]),
        point_context_max_age_seconds=float(
            config["localization"].get("context_max_age_seconds", 2.0)
        ),
        depth_max_age_seconds=float(
            config["stereo"].get("max_result_age_seconds", 1.0)
        ),
        physical_left_view=detection_view if stereo_ready else "",
        physical_right_view=auxiliary_view if stereo_ready else "",
    )
    return RuntimeApplication(
        frame_hub=frame_hub,
        camera_source=camera_source,
        processor=processor,
        health=health,
        health_reporter=health_reporter,
        point_worker=point_worker,
        depth_worker=depth_worker,
        recorder_worker=recorder_worker,
        review_worker=review_thread,
        review_queue=review_queue,
        web_server=web_server,
        camera_stale_seconds=float(config["camera"]["stale_after_seconds"]),
        target_fps=float(config["system"]["target_fps"]),
        max_consecutive_inference_failures=int(
            config["system"].get(
                "max_consecutive_inference_failures", 10
            )
        ),
        source_exhausted=source_exhausted,
        navigation_bridge=NavigationBridge(),
    )


def _camera(
    config: Mapping[str, Any],
    *,
    source: str,
    video_path: str | Path | None,
    topic_override: str | None,
    message_type_override: str | None,
    detection_view_override: str | None = None,
    auxiliary_view_override: str | None = None,
) -> tuple[FrameHub, Any, Any | None]:
    camera = config["camera"]
    common = {
        "layout": str(camera["combined_layout"]),
        "detection_view": str(
            detection_view_override or camera["detection_view"]
        ),
        "auxiliary_view": str(
            auxiliary_view_override or camera["auxiliary_view"]
        ),
        "rotation": str(camera["rotation"]),
    }
    if source == "video":
        if video_path is None:
            raise ValueError("video source requires video_path")
        reader = OpenCvVideoSource(
            video_path,
            fps=float(config["system"]["target_fps"]),
            pace=True,
        )
        hub = FrameHub(source=reader, **common)
        return hub, VideoFrameHubCamera(hub, reader), lambda: reader.eof
    if source != "ros2":
        raise ValueError(f"unsupported camera source: {source}")
    hub = FrameHub(**common)
    requested_type = (
        str(message_type_override)
        if message_type_override
        else str(camera.get("message_type") or "auto")
    )
    ros_source = Ros2FrameSource(
        hub,
        topic=topic_override or str(camera["topic"]),
        message_type=requested_type,
        discovery_timeout_seconds=float(
            camera.get("discovery_timeout_seconds", 8.0)
        ),
    )
    return hub, ros_source, None


def _inference(
    config_path: Path,
    config: Mapping[str, Any],
) -> UnifiedInference:
    model = config["model"]
    backend = HbmRuntimeBackend(resolve_path(config_path, model["hbm_path"]))
    thresholds = {
        str(key): float(value)
        for key, value in dict(model.get("confidence") or {}).items()
    }
    decoder = YoloDecoder(
        model["classes"],
        confidence_threshold=1.0,
        class_thresholds=thresholds,
        iou_threshold=float(model["nms_iou"]),
    )
    return UnifiedInference(
        backend,
        target_height=int(model["input_height"]),
        target_width=int(model["input_width"]),
        input_format=str(model["input_format"]),
        decoder=decoder,
        tracker=ClassAwareTracker(),
    )


def _engines(
    config: Mapping[str, Any],
) -> tuple[ParkingEngine, NightPeopleEngine, FlameEngine]:
    thresholds = config["model"]["confidence"]
    vehicle = config["rules"]["vehicle"]
    night = config["rules"]["night_people"]
    flame = config["rules"]["flame"]
    active_start, active_end = night["active_time_windows"][0]
    return (
        ParkingEngine(
            event_name=str(vehicle["event_name"]),
            min_confidence=float(thresholds["vehicle"]),
            dwell_seconds=float(vehicle["dwell_seconds"]),
            cooldown_seconds=float(vehicle["cooldown_seconds"]),
        ),
        NightPeopleEngine(
            person_event_name=str(night["person_event_name"]),
            crowd_event_name=str(night["crowd_event_name"]),
            min_confidence=float(thresholds["person"]),
            active_start=str(active_start),
            active_end=str(active_end),
            person_dwell_seconds=float(night["person_dwell_seconds"]),
            crowd_min_count=int(night["crowd_min_count"]),
            crowd_dwell_seconds=float(night["crowd_dwell_seconds"]),
            rearm_absence_seconds=float(night["rearm_empty_seconds"]),
            track_lost_grace_seconds=float(
                night["track_lost_grace_seconds"]
            ),
        ),
        FlameEngine(
            event_name=str(flame["event_name"]),
            min_confidence=float(thresholds["fire"]),
            min_width_px=float(flame["min_box_width_px"]),
            min_height_px=float(flame["min_box_height_px"]),
            min_area_px=float(flame["min_box_area_px"]),
            confirmation_seconds=float(flame["confirmation_seconds"]),
            repeat_alarm_seconds=float(flame["repeat_alarm_seconds"]),
            clear_absence_seconds=float(flame["clear_absence_seconds"]),
        ),
    )


def _point_worker(
    config: Mapping[str, Any],
    points_path: Path,
    *,
    camera_matrix: Any | None = None,
    distortion: Any | None = None,
) -> AsyncPointResolver:
    from .localization import PointResolver

    resolver = PointResolver.from_file(
        points_path,
        tag_family=str(config["localization"]["tag_family"]),
        required_stable_frames=int(
            config["localization"]["required_stable_frames"]
        ),
        camera_matrix=camera_matrix,
        distortion=distortion,
    )
    return AsyncPointResolver(
        resolver,
        minimum_interval_seconds=float(
            config["localization"].get("scan_interval_seconds", 0.10)
        ),
    )


def _stereo_estimator(
    config: Mapping[str, Any],
    stereo_path: Path,
) -> Any | None:
    if not bool(config["stereo"].get("enabled", True)):
        return None
    from .stereo import StereoDepthEstimator

    return StereoDepthEstimator.from_file(stereo_path)


def _depth_worker(
    config: Mapping[str, Any],
    estimator: Any | None,
) -> AsyncDepthEstimator | None:
    if estimator is None:
        return None
    return AsyncDepthEstimator(
        estimator,
        minimum_interval_seconds=float(
            config["stereo"].get("compute_interval_seconds", 0.5)
        ),
    )


def _effective_stereo_views(
    config: Mapping[str, Any],
    estimator: Any | None,
) -> tuple[str, str, bool, str]:
    """Make the detector's coordinate system the calibrated physical left view."""

    camera = config["camera"]
    configured_detection = str(camera["detection_view"])
    configured_auxiliary = str(camera["auxiliary_view"])
    if estimator is None:
        return (
            configured_detection,
            configured_auxiliary,
            False,
            "stereo_disabled",
        )
    calibration = estimator.calibration
    if not calibration.valid:
        return (
            configured_detection,
            configured_auxiliary,
            False,
            str(calibration.reason),
        )
    physical_left = str(estimator.physical_left_view or "")
    physical_right = str(estimator.physical_right_view or "")
    if not physical_left or not physical_right:
        return (
            configured_detection,
            configured_auxiliary,
            False,
            "physical_view_mapping_missing",
        )
    if not _same_rotation(
        str(camera.get("rotation") or "none"),
        str(calibration.runtime_rotation or "none"),
    ):
        # The physical mapping is still safe to use for all detection rules,
        # but the calibration matrices are in another pixel coordinate system.
        return (
            physical_left,
            physical_right,
            False,
            "stereo_runtime_rotation_mismatch",
        )
    mapping_ok, reason = estimator.validate_view_mapping(
        physical_left,
        physical_right,
    )
    return physical_left, physical_right, bool(mapping_ok), str(reason)


def _same_rotation(first: str, second: str) -> bool:
    aliases = {
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
    }
    return aliases.get(str(first).lower().replace("_", "")) == aliases.get(
        str(second).lower().replace("_", "")
    )


def _recorder_worker(
    config_path: Path,
    config: Mapping[str, Any],
    data_root: Path | None,
    enabled_override: bool | None,
) -> AsyncDualStreamRecorder | None:
    enabled = bool(config["recording"].get("enabled", True))
    if enabled_override is not None:
        enabled = bool(enabled_override)
    if not enabled:
        return None
    root = _data_path(
        data_root,
        "recordings",
        config_path,
        config["recording"]["output_dir"],
    )
    codec = str(config["recording"].get("codec") or "mp4v")
    if codec == "auto":
        codec = "mp4v"
    kwargs: dict[str, Any] = {
        "fps": float(config["recording"]["fps"]),
        "segment_seconds": float(config["recording"]["segment_seconds"]),
        "writer_factory": OpenCVWriterFactory(codec),
    }
    if "retention_hours" in inspect.signature(DualStreamRecorder).parameters:
        kwargs["retention_hours"] = float(
            config["recording"]["retention_hours"]
        )
    recorder = DualStreamRecorder(root, **kwargs)
    return AsyncDualStreamRecorder(
        recorder,
        queue_capacity=int(config["recording"].get("queue_capacity", 3)),
        max_fps=float(config["recording"]["fps"]),
    )


def _web_server(
    config: Mapping[str, Any],
    repository: AlarmRepository,
    health: HealthRegistry,
    enabled_override: bool | None,
) -> ReadOnlyAlarmServer | None:
    enabled = bool(config["web"].get("enabled", True))
    if enabled_override is not None:
        enabled = bool(enabled_override)
    if not enabled:
        return None
    kwargs: dict[str, Any] = {
        "host": str(config["web"]["bind"]),
        "port": int(config["web"]["port"]),
    }
    if "health_provider" in inspect.signature(
        ReadOnlyAlarmServer
    ).parameters:
        kwargs["health_provider"] = health.snapshot
    return ReadOnlyAlarmServer(repository, **kwargs)


def _point_capabilities(
    document: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for item in document.get("points") or []:
        if not isinstance(item, Mapping) or not item.get("enabled", True):
            continue
        point_id = str(item.get("point_id") or "")
        if point_id:
            result[point_id] = frozenset(
                str(value) for value in item.get("capabilities") or []
            )
    return result


def _data_path(
    data_root: Path | None,
    child: str,
    config_path: Path,
    configured: str | Path,
) -> Path:
    path = (
        data_root / child
        if data_root is not None
        else resolve_path(config_path, configured)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path
