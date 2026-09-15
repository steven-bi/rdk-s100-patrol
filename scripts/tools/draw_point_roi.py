#!/usr/bin/env python3
"""Draw or validate vehicle ROI polygons and write register_point YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import yaml


Point = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在现场参考图上画车辆 ROI，输出 register_point.py 可读取的 YAML。"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-id", default="no_parking_zone_01")
    parser.add_argument(
        "--polygon",
        action="append",
        default=[],
        metavar="NAME:x1,y1;x2,y2;x3,y3",
        help="无界面输入；可重复传入以创建多个 ROI。",
    )
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def parse_polygon(spec: str) -> Tuple[str, List[Point]]:
    if ":" not in spec:
        raise ValueError("--polygon 必须以 NAME: 开头")
    name, raw_points = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError("ROI 名称不能为空")
    points: List[Point] = []
    for raw in raw_points.split(";"):
        values = [part.strip() for part in raw.split(",")]
        if len(values) != 2:
            raise ValueError(f"无效坐标：{raw}")
        points.append((int(values[0]), int(values[1])))
    return name, points


def orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if value == 0:
        return 0
    return 1 if value > 0 else 2


def on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def validate_polygon(
    name: str, points: Sequence[Point], width: int, height: int
) -> None:
    if len(points) < 3:
        raise ValueError(f"{name}: ROI 至少需要 3 个顶点")
    if len(set(points)) != len(points):
        raise ValueError(f"{name}: ROI 含重复顶点")
    if any(x < 0 or x >= width or y < 0 or y >= height for x, y in points):
        raise ValueError(f"{name}: ROI 顶点超出 {width}×{height} 图像")
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if abs(float(cv2.contourArea(contour))) < 25.0:
        raise ValueError(f"{name}: ROI 面积过小")
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count}:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if segments_intersect(a, b, c, d):
                raise ValueError(f"{name}: ROI 多边形自交")


def draw_interactive(image: np.ndarray, roi_id: str) -> List[Point]:
    height, width = image.shape[:2]
    scale = min(1.0, 1280.0 / width, 800.0 / height)
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    points: List[Point] = []
    window = "ROI: left=add right=undo Enter=finish Esc=cancel"

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(
                (
                    min(width - 1, max(0, round(x / scale))),
                    min(height - 1, max(0, round(y / scale))),
                )
            )
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, *display_size)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            preview = image.copy()
            if points:
                contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(preview, [contour], len(points) >= 3, (0, 255, 255), 2)
                for index, (x, y) in enumerate(points, start=1):
                    cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        preview,
                        str(index),
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )
            cv2.putText(
                preview,
                roi_id,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, cv2.resize(preview, display_size))
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                return points
            if key == 27:
                raise KeyboardInterrupt
    finally:
        cv2.destroyWindow(window)


def write_preview(
    image: np.ndarray,
    rois: Sequence[Tuple[str, Sequence[Point]]],
    destination: Path,
) -> None:
    preview = image.copy()
    for name, points in rois:
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [contour], True, (0, 255, 255), 3)
        x, y = points[0]
        cv2.putText(
            preview,
            name,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, preview)
    if not ok:
        raise ValueError(f"无法编码预览图：{destination}")
    encoded.tofile(str(destination))


def main() -> int:
    args = parse_args()
    preview_path = args.preview or args.output.with_suffix(".preview.jpg")
    if args.output.exists() and not args.force:
        print(f"ERROR: 输出已存在；确认后使用 --force：{args.output}", file=sys.stderr)
        return 73
    if preview_path.exists() and not args.force:
        print(f"ERROR: 预览已存在；确认后使用 --force：{preview_path}", file=sys.stderr)
        return 73
    try:
        image = read_image(args.image)
        height, width = image.shape[:2]
        rois: List[Tuple[str, List[Point]]]
        if args.polygon:
            rois = [parse_polygon(spec) for spec in args.polygon]
        else:
            rois = [(args.roi_id, draw_interactive(image, args.roi_id))]
        seen = set()
        for name, points in rois:
            if name in seen:
                raise ValueError(f"重复 ROI 名称：{name}")
            seen.add(name)
            validate_polygon(name, points, width, height)
        payload = {
            "schema_version": "rdk-patrol-rois/v1",
            "source_image": str(args.image.resolve()),
            "image_size": [width, height],
            "rois": [
                {
                    "id": name,
                    "name": name,
                    "polygon": [[int(x), int(y)] for x, y in points],
                }
                for name, points in rois
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        write_preview(image, rois, preview_path)
    except KeyboardInterrupt:
        print("已取消，未写入 ROI。", file=sys.stderr)
        return 130
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 65
    print(f"ROI YAML：{args.output}")
    print(f"预览图：{preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
