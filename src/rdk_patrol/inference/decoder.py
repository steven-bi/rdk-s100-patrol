from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import Detection
from .backend import first_numpy_output
from .preprocess import PreprocessMeta


DEFAULT_MODEL_CLASSES = (
    "person",
    "vehicle",
    "trash_bin",
    "garbage",
    "fire",
    "smoke",
)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def box_iou_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    other_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(box_area + other_area - intersection, 1e-9)


def nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    if boxes.size == 0:
        return []
    order = np.argsort(scores, kind="stable")[::-1]
    keep: list[int] = []
    while order.size:
        selected = int(order[0])
        keep.append(selected)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = box_iou_many(boxes[selected], boxes[remaining])
        order = remaining[ious <= float(iou_threshold)]
    return keep


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """NMS independently per class so overlapping fire/person boxes survive."""

    kept: list[int] = []
    for class_id in sorted(set(int(item) for item in class_ids.tolist())):
        indices = np.flatnonzero(class_ids == class_id)
        for local_index in nms_indices(boxes[indices], scores[indices], iou_threshold):
            kept.append(int(indices[local_index]))
    kept.sort(key=lambda index: (-float(scores[index]), int(class_ids[index]), index))
    return kept


class YoloDecoder:
    """Decode YOLO11 ``4 + classes`` predictions into original-image boxes."""

    def __init__(
        self,
        class_names: Sequence[str] = DEFAULT_MODEL_CLASSES,
        *,
        confidence_threshold: float = 0.25,
        class_thresholds: Mapping[str, float] | None = None,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> None:
        if not class_names:
            raise ValueError("class_names cannot be empty")
        self.class_names = tuple(str(item) for item in class_names)
        self.confidence_threshold = float(confidence_threshold)
        self.class_thresholds = {
            str(key): float(value) for key, value in (class_thresholds or {}).items()
        }
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max(1, int(max_detections))

    def decode(
        self,
        output: Any,
        meta: PreprocessMeta | Mapping[str, Any],
    ) -> list[Detection]:
        prediction, has_objectness = self._prediction_matrix(output)
        if prediction.size == 0:
            return []
        boxes = xywh_to_xyxy(prediction[:, :4].astype(np.float32, copy=False))
        score_offset = 5 if has_objectness else 4
        class_scores = prediction[:, score_offset : score_offset + len(self.class_names)]
        if has_objectness:
            class_scores = class_scores * prediction[:, 4:5]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids].astype(np.float32)
        thresholds = np.asarray(
            [
                self.class_thresholds.get(
                    self.class_names[int(class_id)],
                    self.confidence_threshold,
                )
                for class_id in class_ids
            ],
            dtype=np.float32,
        )
        mask = np.isfinite(scores) & (scores >= thresholds)
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if not len(boxes):
            return []
        keep = class_aware_nms(boxes, scores, class_ids, self.iou_threshold)
        scale, pad_left, pad_top, width, height = _meta_values(meta)
        detections: list[Detection] = []
        for index in keep[: self.max_detections]:
            box = boxes[index].copy()
            box[[0, 2]] = (box[[0, 2]] - pad_left) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_top) / scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0.0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0.0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=float(scores[index]),
                    box=tuple(float(value) for value in box),
                )
            )
        return detections

    def _prediction_matrix(self, output: Any) -> tuple[np.ndarray, bool]:
        array = np.asarray(first_numpy_output(output))
        array = np.squeeze(array)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"unsupported YOLO output shape: {array.shape}")
        no_objectness = 4 + len(self.class_names)
        with_objectness = 5 + len(self.class_names)
        if array.shape[1] in {no_objectness, with_objectness}:
            matrix = array
        elif array.shape[0] in {no_objectness, with_objectness}:
            matrix = array.T
        else:
            raise ValueError(
                f"YOLO output must contain {no_objectness} (YOLO11) or "
                f"{with_objectness} channels, got {array.shape}"
            )
        has_objectness = matrix.shape[1] == with_objectness
        return matrix.astype(np.float32, copy=False), has_objectness


def _meta_values(
    meta: PreprocessMeta | Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    if isinstance(meta, PreprocessMeta):
        return (
            max(1e-12, float(meta.scale)),
            float(meta.pad_left),
            float(meta.pad_top),
            float(meta.original_width),
            float(meta.original_height),
        )
    return (
        max(1e-12, float(meta["scale"])),
        float(meta["pad_left"]),
        float(meta["pad_top"]),
        float(meta.get("original_width", meta.get("input_width"))),
        float(meta.get("original_height", meta.get("input_height"))),
    )
