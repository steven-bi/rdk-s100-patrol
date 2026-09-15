#!/usr/bin/env python3
"""Calibrate GS130W stereo image pairs and emit the project YAML schema."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用同步棋盘格图像标定 GS130W 双目相机。")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--board-cols", type=int, default=9)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--square-size-mm", type=float, required=True)
    parser.add_argument(
        "--physical-left-view", choices=("top", "bottom"), required=True
    )
    parser.add_argument(
        "--layout",
        choices=("vertical",),
        default="vertical",
        help="本项目 GS130W 运行时固定为上下拼接。",
    )
    parser.add_argument(
        "--runtime-rotation",
        choices=("none", "cw90", "ccw90", "rotate180"),
        default="ccw90",
    )
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rms-px", type=float, default=1.0)
    parser.add_argument("--max-epipolar-error-px", type=float, default=1.0)
    parser.add_argument(
        "--accept",
        action="store_true",
        help="仅在已完成遮挡确认且指标通过时使用；否则输出 valid:false。",
    )
    return parser.parse_args()


def list_pairs(root: Path) -> List[Tuple[Path, Path]]:
    left_dir = root / "left"
    right_dir = root / "right"
    right_by_stem = {
        path.stem: path
        for path in right_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    }
    return [
        (path, right_by_stem[path.stem])
        for path in sorted(left_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        and path.stem in right_by_stem
    ]


def board_corners(
    gray: np.ndarray, pattern_size: Tuple[int, int]
) -> Tuple[bool, np.ndarray | None]:
    if hasattr(cv2, "findChessboardCornersSB"):
        return cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found and corners is not None:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
        )
    return bool(found), corners


def nested_list(array: np.ndarray) -> list:
    return np.asarray(array, dtype=float).tolist()


def median_epipolar_error(
    left_points: Sequence[np.ndarray],
    right_points: Sequence[np.ndarray],
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    errors: List[float] = []
    for left, right in zip(left_points, right_points):
        left_rectified = cv2.undistortPoints(left, k1, d1, R=r1, P=p1)
        right_rectified = cv2.undistortPoints(right, k2, d2, R=r2, P=p2)
        errors.extend(
            np.abs(left_rectified[:, 0, 1] - right_rectified[:, 0, 1]).tolist()
        )
    return float(np.median(np.asarray(errors, dtype=float)))


def main() -> int:
    args = parse_args()
    if args.square_size_mm <= 0 or args.board_cols < 3 or args.board_rows < 3:
        print("ERROR: 棋盘参数无效。", file=sys.stderr)
        return 64
    if not (args.pairs / "left").is_dir() or not (args.pairs / "right").is_dir():
        print("ERROR: --pairs 下必须包含 left 和 right 目录。", file=sys.stderr)
        return 66

    pairs = list_pairs(args.pairs)
    if len(pairs) < 12:
        print(f"ERROR: 同名左右图像仅 {len(pairs)} 对，至少需要 12 对，建议 20–30 对。", file=sys.stderr)
        return 65

    pattern_size = (args.board_cols, args.board_rows)
    square_m = args.square_size_mm / 1000.0
    object_template = np.zeros((args.board_cols * args.board_rows, 3), np.float32)
    object_template[:, :2] = (
        np.mgrid[0 : args.board_cols, 0 : args.board_rows].T.reshape(-1, 2)
        * square_m
    )
    object_points: List[np.ndarray] = []
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    for left_path, right_path in pairs:
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            print(f"跳过尺寸/读取异常图像：{left_path.name}")
            continue
        current_size = (left.shape[1], left.shape[0])
        if image_size is not None and current_size != image_size:
            print(f"跳过尺寸不一致图像：{left_path.name}")
            continue
        image_size = current_size
        left_found, left_corners = board_corners(left, pattern_size)
        right_found, right_corners = board_corners(right, pattern_size)
        if left_found and right_found and left_corners is not None and right_corners is not None:
            object_points.append(object_template.copy())
            left_points.append(np.asarray(left_corners, dtype=np.float32))
            right_points.append(np.asarray(right_corners, dtype=np.float32))

    if image_size is None or len(object_points) < 12:
        print(
            f"ERROR: 双目同时成功检出棋盘的图像仅 {len(object_points)} 对。",
            file=sys.stderr,
        )
        return 65

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )
    left_rms, k1, d1, _, _ = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None
    )
    right_rms, k2, d2, _, _ = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None
    )
    stereo_rms, k1, d1, k2, d2, rotation, translation, _, _ = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        k1,
        d1,
        k2,
        d2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        k1,
        d1,
        k2,
        d2,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    epipolar_error = median_epipolar_error(
        left_points, right_points, k1, d1, k2, d2, r1, r2, p1, p2
    )
    baseline_m = float(np.linalg.norm(translation))
    metrics_pass = (
        float(stereo_rms) <= args.max_rms_px
        and epipolar_error <= args.max_epipolar_error_px
        and baseline_m > 0
    )
    accepted = bool(args.accept and metrics_pass)
    physical_right_view = {
        "top": "bottom",
        "bottom": "top",
    }[args.physical_left_view]
    now_bj = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(
        timespec="seconds"
    )

    payload = {
        "schema_version": "rdk-patrol-stereo/v1",
        "camera_model": "GS130W",
        "enabled": True,
        "valid": accepted,
        "physical_left_view": args.physical_left_view,
        "physical_right_view": physical_right_view,
        "combined_layout": args.layout,
        "runtime_rotation": args.runtime_rotation,
        "image_size": list(image_size),
        "images_are_rectified": False,
        "camera_serial": args.camera_serial,
        "calibrated_at": now_bj,
        "baseline_m": baseline_m,
        "left": {
            "camera_matrix": nested_list(k1),
            "distortion_coefficients": nested_list(d1.reshape(-1)),
        },
        "right": {
            "camera_matrix": nested_list(k2),
            "distortion_coefficients": nested_list(d2.reshape(-1)),
        },
        "rotation": nested_list(rotation),
        "translation_m": nested_list(translation.reshape(-1)),
        "rectification": {
            "left_matrix": nested_list(r1),
            "right_matrix": nested_list(r2),
            "left_projection": nested_list(p1),
            "right_projection": nested_list(p2),
            "q_matrix": nested_list(q),
        },
        "matcher": {
            "min_disparity": 0,
            "num_disparities": 128,
            "block_size": 5,
            "uniqueness_ratio": 10,
            "speckle_window_size": 80,
            "speckle_range": 2,
            "minimum_valid_pixels": 20,
            "min_distance_m": 0.3,
            "max_distance_m": 30.0,
        },
        "validation": {
            "stereo_rms_px": float(stereo_rms),
            "left_rms_px": float(left_rms),
            "right_rms_px": float(right_rms),
            "median_epipolar_error_px": epipolar_error,
            "used_pair_count": len(object_points),
            "known_distance_results": [],
            "note": (
                "遮挡法已确认物理左右目；仍需至少三个已知距离点验证。"
                if accepted
                else "未启用：需要指标通过，并在确认物理左右目后显式使用 --accept。"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"有效图像对：{len(object_points)}")
    print(f"左目 RMS：{left_rms:.4f} px")
    print(f"右目 RMS：{right_rms:.4f} px")
    print(f"双目 RMS：{stereo_rms:.4f} px")
    print(f"中位极线误差：{epipolar_error:.4f} px")
    print(f"基线：{baseline_m:.6f} m")
    print(f"valid：{str(accepted).lower()}")
    print(f"输出：{args.output}")
    if not metrics_pass:
        print("ERROR: 标定指标未通过，请重新采集覆盖更充分的图像。", file=sys.stderr)
        return 2
    if not args.accept:
        print("提示：指标通过，但未使用 --accept，因此安全地保持 valid:false。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
