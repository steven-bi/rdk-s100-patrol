from __future__ import annotations

import math
from typing import Sequence

from ..contracts import Box, Point2D


def point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    """Boundary-inclusive ray casting without an OpenCV dependency."""

    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= crossing_x:
                inside = not inside
        previous = current
    return inside


def box_iou(first: Box, second: Box) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / max(1e-9, first_area + second_area - intersection)


def box_center_distance(first: Box, second: Box) -> float:
    return math.dist(
        (
            (float(first[0]) + float(first[2])) * 0.5,
            (float(first[1]) + float(first[3])) * 0.5,
        ),
        (
            (float(second[0]) + float(second[2])) * 0.5,
            (float(second[1]) + float(second[3])) * 0.5,
        ),
    )


def _point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-6:
        return False
    return (
        min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6
        and min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6
    )
