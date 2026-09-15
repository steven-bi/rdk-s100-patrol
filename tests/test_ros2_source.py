from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rdk_patrol.ros2_source import (
    COMPRESSED_IMAGE_TYPE,
    RAW_IMAGE_TYPE,
    decode_ros_image,
    message_stamp_seconds,
)


def _message(**values):
    return SimpleNamespace(**values)


def test_decode_compressed_image() -> None:
    source = np.zeros((12, 18, 3), dtype=np.uint8)
    source[:, :, 1] = 180
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok
    result = decode_ros_image(_message(data=encoded.tobytes()), COMPRESSED_IMAGE_TYPE)
    assert result.shape == source.shape
    assert float(result[:, :, 1].mean()) > 150.0


def test_decode_padded_rgb_image() -> None:
    rgb_row = np.array([255, 0, 0, 0, 255, 0, 9, 9], dtype=np.uint8)
    payload = np.tile(rgb_row, 2)
    message = _message(
        height=2,
        width=2,
        step=8,
        encoding="rgb8",
        data=payload.tobytes(),
    )
    result = decode_ros_image(message, RAW_IMAGE_TYPE)
    assert result.shape == (2, 2, 3)
    assert result[0, 0].tolist() == [0, 0, 255]
    assert result[0, 1].tolist() == [0, 255, 0]


def test_decode_rejects_truncated_raw_image() -> None:
    message = _message(
        height=10,
        width=10,
        step=30,
        encoding="bgr8",
        data=b"\0" * 20,
    )
    with pytest.raises(ValueError, match="truncated"):
        decode_ros_image(message, RAW_IMAGE_TYPE)


def test_message_stamp_seconds() -> None:
    message = _message(header=_message(stamp=_message(sec=12, nanosec=250_000_000)))
    assert message_stamp_seconds(message) == pytest.approx(12.25)
    assert message_stamp_seconds(_message()) is None
