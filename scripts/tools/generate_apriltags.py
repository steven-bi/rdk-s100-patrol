#!/usr/bin/env python3
"""Generate print-ready tag36h11 PNG files with a measurable marker square."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 tag36h11 现场打印文件。")
    parser.add_argument("--ids", type=int, nargs="+", default=[101, 201])
    parser.add_argument("--size-mm", type=float, default=180.0)
    parser.add_argument("--quiet-zone-mm", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("apriltags_tag36h11"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.size_mm <= 0 or args.quiet_zone_mm < 5 or args.dpi < 150:
        raise SystemExit("ERROR: 尺寸、静区或 DPI 参数无效。")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise SystemExit("ERROR: 当前 OpenCV 不含 cv2.aruco AprilTag 支持。")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker_count = int(dictionary.bytesList.shape[0])
    invalid = [tag_id for tag_id in args.ids if tag_id < 0 or tag_id >= marker_count]
    if invalid:
        raise SystemExit(f"ERROR: tag36h11 ID 超出范围：{invalid}")

    args.output.mkdir(parents=True, exist_ok=True)
    marker_px = int(round(args.size_mm / 25.4 * args.dpi))
    quiet_px = int(round(args.quiet_zone_mm / 25.4 * args.dpi))
    label_px = int(round(14.0 / 25.4 * args.dpi))
    outputs: List[Path] = []

    for tag_id in args.ids:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px)
        marker_image = Image.fromarray(np.asarray(marker, dtype=np.uint8), mode="L")
        canvas = Image.new(
            "L",
            (marker_px + quiet_px * 2, marker_px + quiet_px * 2 + label_px),
            color=255,
        )
        canvas.paste(marker_image, (quiet_px, quiet_px))
        draw = ImageDraw.Draw(canvas)
        label = f"tag36h11  ID {tag_id}  BLACK SQUARE {args.size_mm:g} mm"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(18, label_px // 3))
        except OSError:
            font = ImageFont.load_default()
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            ((canvas.width - text_width) // 2, quiet_px * 2 + marker_px),
            label,
            fill=0,
            font=font,
        )
        output = args.output / f"tag36h11_id_{tag_id}_{args.size_mm:g}mm.png"
        canvas.save(output, dpi=(args.dpi, args.dpi), compress_level=9)
        outputs.append(output)
        print(f"已生成：{output}")

    manifest = args.output / "打印说明.txt"
    lines = [
        "AprilTag 打印说明",
        "家族：tag36h11",
        f"黑色标记外边长：{args.size_mm:g} mm",
        f"白色静区：每边至少 {args.quiet_zone_mm:g} mm",
        f"文件 DPI：{args.dpi}",
        "打印要求：原始尺寸/100%，关闭“适合页面”和任何缩放。",
        "打印后必须用钢尺测量黑色正方形外边；误差建议不超过 1 mm。",
        "使用哑光纸或哑光硬板，保持平整，不覆高反光膜。",
        "",
        "SHA256：",
    ]
    lines.extend(f"{sha256(path)}  {path.name}" for path in outputs)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"打印清单：{manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
