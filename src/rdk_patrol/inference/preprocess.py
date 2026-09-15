from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


SUPPORTED_INPUT_FORMATS = frozenset(
    {"rgb_u8", "bgr_u8", "rgb_i8_centered", "rgb_i8_norm127"}
)


@dataclass(frozen=True)
class PreprocessMeta:
    scale: float
    pad_left: int
    pad_top: int
    original_width: int
    original_height: int
    target_width: int
    target_height: int

    @property
    def input_width(self) -> int:
        """Compatibility name used by the proven standalone decoders."""

        return self.original_width

    @property
    def input_height(self) -> int:
        return self.original_height

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "pad_left": int(self.pad_left),
            "pad_top": int(self.pad_top),
            "input_width": int(self.original_width),
            "input_height": int(self.original_height),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
        }


@dataclass(frozen=True)
class PreprocessResult:
    model_input: np.ndarray
    letterboxed_bgr: np.ndarray
    meta: PreprocessMeta


def letterbox_bgr(
    image_bgr: np.ndarray,
    target_height: int,
    target_width: int,
    *,
    color: int = 114,
) -> tuple[np.ndarray, PreprocessMeta]:
    _validate_bgr(image_bgr)
    target_height = int(target_height)
    target_width = int(target_width)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target dimensions must be positive")
    height, width = image_bgr.shape[:2]
    scale = min(target_height / height, target_width / width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full(
        (target_height, target_width, 3),
        int(np.clip(color, 0, 255)),
        dtype=np.uint8,
    )
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    return canvas, PreprocessMeta(
        scale=float(scale),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        original_width=int(width),
        original_height=int(height),
        target_width=target_width,
        target_height=target_height,
    )


def build_model_input(letterboxed_bgr: np.ndarray, input_format: str) -> np.ndarray:
    _validate_bgr(letterboxed_bgr)
    normalized = str(input_format).strip().lower()
    if normalized not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"unsupported input_format={input_format!r}; "
            f"choices={sorted(SUPPORTED_INPUT_FORMATS)}"
        )
    if normalized == "bgr_u8":
        return np.ascontiguousarray(letterboxed_bgr.astype(np.uint8, copy=False))
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    if normalized == "rgb_u8":
        return np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
    if normalized == "rgb_i8_centered":
        return np.ascontiguousarray((rgb.astype(np.int16) - 128).astype(np.int8))
    return np.ascontiguousarray(
        np.round(rgb.astype(np.float32) / 255.0 * 127.0).astype(np.int8)
    )


def preprocess_bgr(
    image_bgr: np.ndarray,
    *,
    target_height: int = 640,
    target_width: int = 640,
    input_format: str = "rgb_u8",
    letterbox_color: int = 114,
) -> PreprocessResult:
    letterboxed, meta = letterbox_bgr(
        image_bgr,
        target_height,
        target_width,
        color=letterbox_color,
    )
    return PreprocessResult(
        model_input=build_model_input(letterboxed, input_format),
        letterboxed_bgr=letterboxed,
        meta=meta,
    )


def _validate_bgr(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("image must be a non-empty HxWx3 BGR array")
