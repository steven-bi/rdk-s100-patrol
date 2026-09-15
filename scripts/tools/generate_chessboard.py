#!/usr/bin/env python3
"""Generate a dimensioned SVG chessboard for stereo calibration."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成精确毫米尺寸的双目标定棋盘 SVG。")
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--margin-mm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("chessboard_9x6_30mm.svg"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.inner_cols < 3
        or args.inner_rows < 3
        or args.square_mm <= 0
        or args.margin_mm < args.square_mm / 2
    ):
        raise SystemExit("ERROR: 内角点、方格或白边尺寸无效。")
    square_cols = args.inner_cols + 1
    square_rows = args.inner_rows + 1
    board_width = square_cols * args.square_mm
    board_height = square_rows * args.square_mm
    page_width = board_width + 2 * args.margin_mm
    page_height = board_height + 2 * args.margin_mm

    rectangles = []
    for row in range(square_rows):
        for col in range(square_cols):
            if (row + col) % 2 == 0:
                x = args.margin_mm + col * args.square_mm
                y = args.margin_mm + row * args.square_mm
                rectangles.append(
                    f'<rect x="{x:g}" y="{y:g}" width="{args.square_mm:g}" '
                    f'height="{args.square_mm:g}" fill="#000"/>'
                )
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{page_width:g}mm" height="{page_height:g}mm"
     viewBox="0 0 {page_width:g} {page_height:g}">
  <rect x="0" y="0" width="{page_width:g}" height="{page_height:g}" fill="#fff"/>
  {' '.join(rectangles)}
</svg>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    note_path = args.output.with_suffix(".打印说明.txt")
    note_path.write_text(
        "\n".join(
            [
                "GS130W 双目标定棋盘打印说明",
                f"内角点：{args.inner_cols} 列 × {args.inner_rows} 行",
                f"方格：{square_cols} 列 × {square_rows} 行",
                f"单格理论边长：{args.square_mm:g} mm",
                f"棋盘理论外尺寸：{board_width:g} mm × {board_height:g} mm",
                f"含白边页面尺寸：{page_width:g} mm × {page_height:g} mm",
                "打印：100%/实际尺寸，关闭适合页面和缩放；尺寸放不下时改用更大纸张。",
                "安装：粘贴到完全平整的哑光硬板，不能拉伸、翘曲或覆反光膜。",
                "复测：至少测量横纵各 5 格总长，用实测平均单格尺寸执行标定。",
                f"SVG SHA256：{digest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已生成：{args.output}")
    print(f"打印说明：{note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
