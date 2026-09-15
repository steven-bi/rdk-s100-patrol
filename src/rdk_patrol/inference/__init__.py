"""Single-resident HBM inference, YOLO decoding, and lightweight tracking."""

from .backend import (
    FakeBackend,
    HBMRuntimeBackend,
    HbmRuntimeBackend,
    InferenceBackend,
)
from .decoder import DEFAULT_MODEL_CLASSES, YoloDecoder, class_aware_nms
from .pipeline import InferencePipeline, InferenceResult, UnifiedInference
from .preprocess import PreprocessMeta, PreprocessResult, preprocess_bgr
from .tracker import ClassAwareTracker, TrackSnapshot

__all__ = [
    "ClassAwareTracker",
    "DEFAULT_MODEL_CLASSES",
    "FakeBackend",
    "HbmRuntimeBackend",
    "HBMRuntimeBackend",
    "InferenceBackend",
    "InferencePipeline",
    "InferenceResult",
    "PreprocessMeta",
    "PreprocessResult",
    "TrackSnapshot",
    "UnifiedInference",
    "YoloDecoder",
    "class_aware_nms",
    "preprocess_bgr",
]
