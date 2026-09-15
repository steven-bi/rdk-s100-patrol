from __future__ import annotations

"""Normalize engine evidence and draw the lightweight live debug stream."""

from typing import Iterable

import cv2
import numpy as np

from .contracts import AlarmCandidate, Detection, PointContext


_COLORS = {
    "person": (80, 220, 80),
    "vehicle": (0, 190, 255),
    "fire": (30, 30, 255),
}


def enrich_alarm_candidate(
    candidate: AlarmCandidate,
    detections: Iterable[Detection] = (),
) -> AlarmCandidate:
    """Add renderer-friendly boxes/polygons without changing public fields."""

    evidence = dict(candidate.evidence or {})
    detection_rows = list(detections)
    if detection_rows and "boxes" not in evidence:
        evidence["boxes"] = [
            {
                "box": [float(value) for value in detection.box],
                "label": "{} {:.1%}".format(
                    _display_class(detection.class_name),
                    float(detection.confidence),
                ),
            }
            for detection in detection_rows
        ]
    if "roi_polygon" in evidence and "polygons" not in evidence:
        evidence["polygons"] = [
            {
                "points": evidence["roi_polygon"],
                "label": "禁停敏感区域",
            }
        ]

    lines = list(evidence.get("overlay_lines") or [])
    if candidate.event_name == "车辆违停":
        duration = evidence.get("duration_seconds")
        if duration is not None:
            lines.append("持续时间：{:.1f} 秒".format(float(duration)))
        source = evidence.get("point_source")
        if source:
            lines.append("点位加载：{}".format(source))
    elif candidate.event_name in {"夜间人员逗留", "夜间人群聚集"}:
        count = evidence.get("person_count")
        duration = evidence.get("duration_seconds")
        if count is not None:
            lines.append("人数：{}".format(count))
        if duration is not None:
            lines.append("持续时间：{:.1f} 秒".format(float(duration)))
    elif candidate.event_name == "明火报警":
        detection = evidence.get("detection")
        if isinstance(detection, dict) and detection.get("confidence") is not None:
            lines.append(
                "明火置信度：{:.1%}".format(float(detection["confidence"]))
            )
        distance = evidence.get("distance_m")
        if distance is None:
            lines.append("距离：不可用")
        else:
            lines.append("距离：{:.2f} 米".format(float(distance)))
    elif candidate.event_name == "垃圾桶已满":
        if evidence.get("review_result"):
            lines.append("MiniMax结论：{}".format(evidence["review_result"]))
    if lines:
        # Preserve order while removing duplicates.
        evidence["overlay_lines"] = list(dict.fromkeys(str(line) for line in lines))
    candidate.evidence = evidence
    return candidate


def annotate_live_frame(
    frame_bgr: np.ndarray,
    detections: Iterable[Detection],
    *,
    point_context: PointContext | None = None,
    loop_fps: float | None = None,
) -> np.ndarray:
    """Draw only evidence used by enabled local rules.

    ``smoke`` is intentionally not drawn because smoke does not participate in
    the confirmed first-release fire rule.
    """

    canvas = np.ascontiguousarray(frame_bgr).copy()
    for detection in detections:
        color = _COLORS.get(detection.class_name)
        if color is None:
            continue
        x1, y1, x2, y2 = (
            int(round(float(value))) for value in detection.box
        )
        x1 = max(0, min(canvas.shape[1] - 1, x1))
        x2 = max(0, min(canvas.shape[1] - 1, x2))
        y1 = max(0, min(canvas.shape[0] - 1, y1))
        y2 = max(0, min(canvas.shape[0] - 1, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = "{} {:.2f}".format(
            detection.class_name,
            float(detection.confidence),
        )
        if detection.track_id is not None:
            text += " id={}".format(detection.track_id)
        cv2.putText(
            canvas,
            text,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if point_context is not None:
        for name, polygon in point_context.rois.items():
            points = np.asarray(polygon, dtype=np.int32)
            if points.ndim == 2 and points.shape[0] >= 3:
                cv2.polylines(
                    canvas,
                    [points.reshape(-1, 1, 2)],
                    True,
                    (0, 190, 255),
                    2,
                    cv2.LINE_AA,
                )
                anchor = tuple(int(value) for value in points[0])
                cv2.putText(
                    canvas,
                    str(name),
                    anchor,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 190, 255),
                    2,
                    cv2.LINE_AA,
                )
        status = "point={} source={} conf={:.2f}".format(
            point_context.point_id,
            point_context.source,
            float(point_context.confidence),
        )
        cv2.putText(
            canvas,
            status,
            (10, canvas.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if loop_fps is not None:
        cv2.putText(
            canvas,
            "pipeline {:.1f} FPS".format(float(loop_fps)),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _display_class(name: str) -> str:
    return {
        "person": "人员",
        "vehicle": "车辆",
        "fire": "明火",
    }.get(name, name)
