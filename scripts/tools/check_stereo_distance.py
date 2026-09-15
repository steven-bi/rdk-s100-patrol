#!/usr/bin/env python3
"""Validate a stereo calibration against a known-distance target."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用一对同步图像核验 GS130W 已知距离。")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--roi",
        help="左目校正图上的 x,y,w,h；省略时打开交互框选窗口。",
    )
    parser.add_argument("--known-distance-m", type=float)
    parser.add_argument("--max-relative-error", type=float, default=0.10)
    parser.add_argument("--annotated-output", type=Path)
    parser.add_argument(
        "--record-result",
        action="store_true",
        help="将结果追加到标定 YAML；累计至少 3 个通过点后才会设 valid:true。",
    )
    return parser.parse_args()


def array(payload: Dict[str, Any], *keys: str) -> np.ndarray:
    value: Any = payload
    for key in keys:
        value = value[key]
    return np.asarray(value, dtype=np.float64)


def parse_roi(value: str | None, image: np.ndarray) -> Tuple[int, int, int, int]:
    if value:
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("--roi 必须是 x,y,w,h")
        return tuple(parts)  # type: ignore[return-value]
    selected = cv2.selectROI("在已知距离目标内部框选纹理区域", image, False, False)
    cv2.destroyWindow("在已知距离目标内部框选纹理区域")
    return tuple(int(value) for value in selected)


def main() -> int:
    args = parse_args()
    if args.record_result and (args.known_distance_m is None or args.known_distance_m <= 0):
        print("ERROR: --record-result 必须同时提供正数 --known-distance-m。", file=sys.stderr)
        return 64

    left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
    if left is None or right is None or left.shape != right.shape:
        print("ERROR: 左右图读取失败或尺寸不一致。", file=sys.stderr)
        return 66
    payload = yaml.safe_load(args.calibration.read_text(encoding="utf-8"))
    expected_size = tuple(int(value) for value in payload["image_size"])
    if (left.shape[1], left.shape[0]) != expected_size:
        print(
            f"ERROR: 图像尺寸 {(left.shape[1], left.shape[0])} 与标定尺寸 {expected_size} 不同。",
            file=sys.stderr,
        )
        return 65

    k1 = array(payload, "left", "camera_matrix")
    d1 = array(payload, "left", "distortion_coefficients")
    k2 = array(payload, "right", "camera_matrix")
    d2 = array(payload, "right", "distortion_coefficients")
    r1 = array(payload, "rectification", "left_matrix")
    r2 = array(payload, "rectification", "right_matrix")
    p1 = array(payload, "rectification", "left_projection")
    p2 = array(payload, "rectification", "right_projection")
    rectification = payload["rectification"]
    q = np.asarray(
        rectification.get("q_matrix", rectification.get("q")),
        dtype=np.float64,
    )
    if q.shape != (4, 4) or not np.isfinite(q).all():
        print("ERROR: 标定文件缺少有效的 4×4 Q 矩阵。", file=sys.stderr)
        return 65
    map1x, map1y = cv2.initUndistortRectifyMap(
        k1, d1, r1, p1, expected_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        k2, d2, r2, p2, expected_size, cv2.CV_32FC1
    )
    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    matcher_config = payload.get("matcher", {})
    num_disparities = int(matcher_config.get("num_disparities", 128))
    num_disparities = max(16, ((num_disparities + 15) // 16) * 16)
    block_size = int(matcher_config.get("block_size", 5))
    block_size = max(3, block_size | 1)
    matcher = cv2.StereoSGBM_create(
        minDisparity=int(matcher_config.get("min_disparity", 0)),
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        uniquenessRatio=int(matcher_config.get("uniqueness_ratio", 10)),
        speckleWindowSize=int(matcher_config.get("speckle_window_size", 80)),
        speckleRange=int(matcher_config.get("speckle_range", 2)),
        disp12MaxDiff=1,
    )
    disparity = matcher.compute(
        cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY),
    ).astype(np.float32) / 16.0
    points_3d = cv2.reprojectImageTo3D(disparity, q)

    try:
        x, y, width, height = parse_roi(args.roi, left_rect)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 64
    x = max(0, x)
    y = max(0, y)
    width = min(width, left_rect.shape[1] - x)
    height = min(height, left_rect.shape[0] - y)
    if width < 5 or height < 5:
        print("ERROR: ROI 太小或超出图像。", file=sys.stderr)
        return 65

    xyz = points_3d[y : y + height, x : x + width]
    disp_roi = disparity[y : y + height, x : x + width]
    distances = np.linalg.norm(xyz, axis=2)
    minimum = float(matcher_config.get("min_distance_m", 0.3))
    maximum = float(matcher_config.get("max_distance_m", 30.0))
    valid = (
        np.isfinite(distances)
        & np.isfinite(disp_roi)
        & (disp_roi > 0)
        & (distances >= minimum)
        & (distances <= maximum)
    )
    samples = distances[valid]
    minimum_pixels = int(matcher_config.get("minimum_valid_pixels", 20))
    if samples.size < minimum_pixels:
        print(
            f"ERROR: ROI 有效深度像素仅 {samples.size}，至少需要 {minimum_pixels}。",
            file=sys.stderr,
        )
        return 2
    estimate = float(np.median(samples))
    mad = float(np.median(np.abs(samples - estimate)))
    relative_error = None
    passed = True
    if args.known_distance_m is not None:
        relative_error = abs(estimate - args.known_distance_m) / args.known_distance_m
        passed = relative_error <= args.max_relative_error

    print(f"估计直线距离：{estimate:.3f} m")
    print(f"有效像素：{samples.size}")
    print(f"距离中位绝对偏差：{mad:.3f} m")
    if relative_error is not None:
        print(f"已知距离：{args.known_distance_m:.3f} m")
        print(f"相对误差：{relative_error:.2%}")
        print(f"结果：{'通过' if passed else '不通过'}")

    annotated = left_rect.copy()
    cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"distance={estimate:.3f}m",
        (x, max(24, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if args.annotated_output:
        args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.annotated_output), annotated)

    if args.record_result:
        validation = payload.setdefault("validation", {})
        results = validation.setdefault("known_distance_results", [])
        results.append(
            {
                "checked_at": dt.datetime.now(
                    dt.timezone(dt.timedelta(hours=8))
                ).isoformat(timespec="seconds"),
                "known_distance_m": float(args.known_distance_m),
                "estimated_distance_m": estimate,
                "relative_error": float(relative_error),
                "valid_pixels": int(samples.size),
                "passed": bool(passed),
                "source_left": args.left.name,
                "source_right": args.right.name,
            }
        )
        passing = [result for result in results if bool(result.get("passed"))]
        distinct_known_distances = []
        for result in sorted(
            passing, key=lambda item: float(item.get("known_distance_m", 0.0))
        ):
            known = float(result.get("known_distance_m", 0.0))
            if known <= 0:
                continue
            if not distinct_known_distances or (
                abs(known - distinct_known_distances[-1])
                / max(known, distinct_known_distances[-1])
                >= 0.05
            ):
                distinct_known_distances.append(known)
        calibration_metrics_ok = (
            float(validation.get("stereo_rms_px", float("inf"))) <= 1.0
            and float(validation.get("median_epipolar_error_px", float("inf"))) <= 1.0
        )
        payload["valid"] = bool(
            len(distinct_known_distances) >= 3 and calibration_metrics_ok
        )
        validation["note"] = (
            "已通过至少三个彼此相差不小于 5% 的已知距离点验证。"
            if payload["valid"]
            else "需要至少三个彼此相差不小于 5%、分布在近中远距离的通过点，且 RMS/极线误差均不大于 1 px。"
        )
        args.calibration.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"已记录到标定文件；通过点 {len(passing)} 个，"
            f"不同距离 {len(distinct_known_distances)} 个，"
            f"valid={str(payload['valid']).lower()}。"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
