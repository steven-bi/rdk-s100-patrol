from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULTS: dict[str, Any] = {
    "schema_version": "rdk-patrol-final/v1",
    "system": {
        "timezone": "Asia/Shanghai",
        "target_fps": 15.0,
        "latest_frame_only": True,
        "status_interval_seconds": 1.0,
        "high_precision_mode": True,
        "health_path": "../runtime/status/health.json",
        "max_consecutive_inference_failures": 10,
    },
    "camera": {
        "topic": "/image_combine_jpeg",
        "message_type": "auto",
        "combined_layout": "vertical",
        "detection_view": "bottom",
        "auxiliary_view": "top",
        "rotation": "ccw90",
        "stale_after_seconds": 1.0,
        "discovery_timeout_seconds": 8.0,
    },
    "model": {
        "classes": ["person", "vehicle", "trash_bin", "garbage", "fire", "smoke"],
        "input_height": 640,
        "input_width": 320,
        "input_format": "rgb_i8_centered",
        "nms_iou": 0.50,
        "confidence": {
            "person": 0.45,
            "vehicle": 0.30,
            "fire": 0.55,
            "trash_bin": 1.0,
            "garbage": 1.0,
            "smoke": 1.0,
        },
    },
    "rules": {
        "vehicle": {
            "event_name": "车辆违停",
            "dwell_seconds": 2.0,
            "cooldown_seconds": 300.0,
        },
        "night_people": {
            "person_event_name": "夜间人员逗留",
            "crowd_event_name": "夜间人群聚集",
            "active_time_windows": [["22:00", "06:00"]],
            "person_dwell_seconds": 5.0,
            "crowd_min_count": 3,
            "crowd_dwell_seconds": 5.0,
            "track_lost_grace_seconds": 1.0,
            "rearm_empty_seconds": 30.0,
        },
        "flame": {
            "event_name": "明火报警",
            "confirmation_seconds": 2.0,
            "repeat_alarm_seconds": 600.0,
            "clear_absence_seconds": 600.0,
            "min_box_width_px": 24,
            "min_box_height_px": 24,
            "min_box_area_px": 576,
            "suppression_zones": [],
        },
        "trash": {
            "event_name": "垃圾桶已满",
            "point_cooldown_seconds": 300.0,
            "best_frame_window_seconds": 5.0,
        },
    },
    "localization": {
        "tag_family": "tag36h11",
        "tag_priority": True,
        "required_stable_frames": 3,
        "fingerprint_fallback": True,
        "point_hold_seconds": 5.0,
        "scan_interval_seconds": 0.10,
        "context_max_age_seconds": 2.0,
    },
    "recording": {
        "enabled": True,
        "fps": 15.0,
        "segment_seconds": 300.0,
        "retention_hours": 72.0,
        "raw_enabled": True,
        "annotated_enabled": True,
        "minimum_free_gb": 20.0,
        "output_dir": "../runtime/videos",
        "queue_capacity": 32,
    },
    "web": {
        "enabled": True,
        "bind": "0.0.0.0",
        "port": 8081,
        "read_only": True,
        "authentication": False,
    },
    "navigation": {
        "adapter": "null",
        "hold_seconds": 5.0,
    },
    "alarms": {
        "output_dir": "../runtime/alarms",
        "jpeg_quality": 92,
        "font_path": "../assets/fonts/NotoSansSC-VF.ttf",
        "ding_talk_enabled": False,
    },
    "minimax": {
        "enabled": True,
        "base_url": "https://api.minimaxi.com/v1",
        "model": "MiniMax-M3",
        "api_key_env": "MINIMAX_API_KEY",
        "timeout_seconds": 2.0,
        "retry_interval_seconds": 10.0,
        "image_detail": "low",
        "max_long_side_pixel": 960,
        "jpeg_quality": 85,
        "queue_dir": "../runtime/review_queue",
    },
    "stereo": {
        "enabled": True,
        "calibration_path": "stereo_gs130w.yaml",
        "compute_only_for_flame_candidates": True,
        "compute_interval_seconds": 0.5,
        "max_result_age_seconds": 1.0,
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    document = _deep_merge(copy.deepcopy(DEFAULTS), load_yaml(path))
    validate_config(document)
    return document


def validate_config(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != "rdk-patrol-final/v1":
        raise ValueError("unsupported schema_version")
    target_fps = float(document["system"]["target_fps"])
    if target_fps < 15.0:
        raise ValueError("target_fps must remain at least 15")
    camera = document["camera"]
    if camera["combined_layout"] not in {"vertical", "horizontal"}:
        raise ValueError("camera.combined_layout must be vertical or horizontal")
    if camera["detection_view"] == camera["auxiliary_view"]:
        raise ValueError("detection_view and auxiliary_view must differ")
    if document["web"].get("read_only") is not True:
        raise ValueError("the first-release web interface must be read-only")
    if document["web"].get("authentication") is not False:
        raise ValueError("the confirmed first-release LAN viewer has no login")
    if float(document["navigation"]["hold_seconds"]) != 5.0:
        raise ValueError("the confirmed first-release point hold is 5 seconds")
    if document.get("alarms", {}).get("ding_talk_enabled") is not False:
        raise ValueError("DingTalk delivery must remain disabled in the first release")
    expected_names = {
        ("vehicle", "event_name"): "车辆违停",
        ("night_people", "person_event_name"): "夜间人员逗留",
        ("night_people", "crowd_event_name"): "夜间人群聚集",
        ("flame", "event_name"): "明火报警",
        ("trash", "event_name"): "垃圾桶已满",
    }
    for (section, key), expected in expected_names.items():
        if document["rules"][section].get(key) != expected:
            raise ValueError(f"rules.{section}.{key} must remain {expected!r}")
    windows = document["rules"]["night_people"].get("active_time_windows")
    if not isinstance(windows, list) or len(windows) != 1 or len(windows[0]) != 2:
        raise ValueError(
            "rules.night_people.active_time_windows must contain one adjustable "
            "[start, end] Beijing-time window"
        )
    if document["rules"]["flame"].get("suppression_zones"):
        raise ValueError("the confirmed flame rule is full-frame")
    if document["recording"].get("raw_enabled") is not True:
        raise ValueError("raw stereo recording must remain enabled")
    if document["recording"].get("annotated_enabled") is not True:
        raise ValueError("annotated main-view recording must remain enabled")


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()
