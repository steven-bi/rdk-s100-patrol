from __future__ import annotations

"""HBM backend adapters.

Only this layer imports ``hbm_runtime`` and the import occurs lazily in the
constructor.  Development-PC tests use :class:`FakeBackend`.
"""

from collections import deque
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class InferenceBackend(Protocol):
    def infer(self, model_input: np.ndarray) -> Any:
        ...


class FakeBackend:
    """Deterministic backend that returns queued arrays or calls a function."""

    def __init__(
        self,
        outputs: Iterable[Any] | None = None,
        *,
        infer_fn: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        if outputs is not None and infer_fn is not None:
            raise ValueError("provide outputs or infer_fn, not both")
        self.outputs = deque(outputs or [])
        self.infer_fn = infer_fn
        self.call_count = 0
        self.last_input: np.ndarray | None = None

    def infer(self, model_input: np.ndarray) -> Any:
        self.call_count += 1
        self.last_input = np.asarray(model_input).copy()
        started = time.perf_counter()
        if self.infer_fn is not None:
            output = self.infer_fn(model_input)
        elif self.outputs:
            output = self.outputs.popleft()
        else:
            raise RuntimeError("FakeBackend has no output queued")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Preserve an explicitly supplied (elapsed_ms, output) response.
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[0], (int, float))
        ):
            return output
        return elapsed_ms, output


class HbmRuntimeBackend:
    """Load one HBM model once and keep it resident for the process lifetime."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            from hbm_runtime import HB_HBMRuntime
        except ImportError as exc:
            raise RuntimeError(
                "hbm_runtime is only available in the RDK S100 board image"
            ) from exc
        self.model_path = path
        self.model = HB_HBMRuntime(str(path))
        self.model_name: str | None = None
        self.input_name: str | None = None
        try:
            names = list(self.model.model_names)
            self.model_name = str(names[0]) if names else None
        except Exception:
            self.model_name = None
        self.input_shape = self._first_input_shape()
        self.call_count = 0

    def infer(self, model_input: np.ndarray) -> tuple[float, np.ndarray]:
        input_array = runtime_input_array(model_input, self.input_shape)
        started = time.perf_counter()
        try:
            outputs = self.model.run(input_array)
        except Exception:
            if not self.input_name:
                raise
            outputs = self.model.run({self.input_name: input_array})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.call_count += 1
        if self.model_name and isinstance(outputs, dict) and self.model_name in outputs:
            return elapsed_ms, first_numpy_output(outputs[self.model_name])
        return elapsed_ms, first_numpy_output(outputs)

    def _first_input_shape(self) -> Any:
        for attribute in ("input_shapes", "inputs"):
            try:
                value = getattr(self.model, attribute)
            except Exception:
                continue
            if isinstance(value, dict) and value:
                self.input_name = str(next(iter(value)))
                return next(iter(value.values()))
            if isinstance(value, (list, tuple)) and value:
                first = value[0]
                if isinstance(first, dict):
                    for key in ("name", "input_name"):
                        if key in first:
                            self.input_name = str(first[key])
                    for key in ("shape", "dims"):
                        if key in first:
                            return first[key]
                return first
        return None


def first_numpy_output(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_numpy_output(item)
            except TypeError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_numpy_output(item)
            except TypeError:
                continue
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    raise TypeError(f"no numpy output tensor found in {type(value).__name__}")


def numeric_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("shape", "dims", "input_shape"):
            if key in value:
                shape = numeric_shape(value[key])
                if shape:
                    return shape
        for item in value.values():
            shape = numeric_shape(item)
            if shape:
                return shape
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        try:
            shape = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        if shape and all(item > 0 for item in shape):
            return shape
    return None


def runtime_input_array(model_input: np.ndarray, expected_shape: Any) -> np.ndarray:
    array = np.ascontiguousarray(model_input)
    shape = numeric_shape(expected_shape)
    if shape is None or int(np.prod(shape)) != int(array.size):
        return array
    return np.ascontiguousarray(array.reshape(shape))


# Backward-compatible spelling used in the standalone modules.
HBMRuntimeBackend = HbmRuntimeBackend
