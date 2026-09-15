from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Mapping

import numpy as np

from ..contracts import Detection, FrameEnvelope
from .backend import InferenceBackend, first_numpy_output
from .decoder import DEFAULT_MODEL_CLASSES, YoloDecoder
from .preprocess import PreprocessMeta, preprocess_bgr
from .tracker import ClassAwareTracker


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    sequence: int | None
    inference_ms: float
    preprocessing_ms: float
    decoding_ms: float
    meta: PreprocessMeta
    cached: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class UnifiedInference:
    """One resident backend call per unique frame sequence."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        target_height: int = 640,
        target_width: int = 640,
        input_format: str = "rgb_u8",
        decoder: YoloDecoder | None = None,
        tracker: ClassAwareTracker | None = None,
    ) -> None:
        self.backend = backend
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.input_format = str(input_format)
        self.decoder = decoder or YoloDecoder(DEFAULT_MODEL_CLASSES)
        self.tracker = tracker or ClassAwareTracker()
        self._lock = threading.Lock()
        self._last_sequence: int | None = None
        self._last_result: InferenceResult | None = None

    def infer(
        self,
        image_bgr: np.ndarray,
        *,
        sequence: int | None = None,
        now: float | None = None,
    ) -> InferenceResult:
        with self._lock:
            if (
                sequence is not None
                and self._last_sequence == int(sequence)
                and self._last_result is not None
            ):
                previous = self._last_result
                return InferenceResult(
                    detections=previous.detections,
                    sequence=previous.sequence,
                    inference_ms=previous.inference_ms,
                    preprocessing_ms=previous.preprocessing_ms,
                    decoding_ms=previous.decoding_ms,
                    meta=previous.meta,
                    cached=True,
                    diagnostics=dict(previous.diagnostics),
                )
            preprocess_started = time.perf_counter()
            preprocessed = preprocess_bgr(
                image_bgr,
                target_height=self.target_height,
                target_width=self.target_width,
                input_format=self.input_format,
            )
            preprocessing_ms = (time.perf_counter() - preprocess_started) * 1000.0

            call_started = time.perf_counter()
            response = self.backend.infer(preprocessed.model_input)
            measured_ms = (time.perf_counter() - call_started) * 1000.0
            inference_ms, output = _normalize_backend_response(response, measured_ms)

            decode_started = time.perf_counter()
            decoded = self.decoder.decode(first_numpy_output(output), preprocessed.meta)
            tracked = self.tracker.update(decoded, now=now)
            decoding_ms = (time.perf_counter() - decode_started) * 1000.0
            result = InferenceResult(
                detections=tuple(tracked),
                sequence=None if sequence is None else int(sequence),
                inference_ms=float(inference_ms),
                preprocessing_ms=float(preprocessing_ms),
                decoding_ms=float(decoding_ms),
                meta=preprocessed.meta,
                diagnostics={"backend": type(self.backend).__name__},
            )
            if sequence is not None:
                self._last_sequence = int(sequence)
                self._last_result = result
            return result

    def infer_envelope(
        self,
        envelope: FrameEnvelope,
        *,
        now: float | None = None,
    ) -> InferenceResult:
        return self.infer(
            envelope.detection_bgr,
            sequence=envelope.sequence,
            now=envelope.monotonic_ts if now is None else now,
        )

    process = infer

    def reset_tracking(self) -> None:
        with self._lock:
            self.tracker.reset()
            self._last_sequence = None
            self._last_result = None


InferencePipeline = UnifiedInference


def _normalize_backend_response(
    response: Any,
    measured_ms: float,
) -> tuple[float, Any]:
    if (
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], (int, float))
    ):
        return float(response[0]), response[1]
    return float(measured_ms), response
