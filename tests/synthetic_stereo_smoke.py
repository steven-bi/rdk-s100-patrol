"""Offline smoke test for the deployed-runtime stereo path.

This is not a substitute for the real 1/3/5 m target.  It verifies that the
validator uses positive GS130W disparity, the candidate Q matrix, the runtime
SGBM configuration, and the strict ROI quality gates coherently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r"D:\RDK_S100_Patrol_Final")
PRISTINE = PROJECT / "calibration_data/GS130W_20260805/known_distance_validation/stereo_gs130w_20260805_v2_candidate_01_pristine.yaml"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "gs130w_distance_validator", ROOT / "validate_gs130w_distance_web.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shifted_pair(disparity_px: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = 1280, 640
    rng = np.random.default_rng(20260806)
    texture = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.45)
    right = np.zeros_like(texture)
    right[:, : width - disparity_px] = texture[:, disparity_px:]
    return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR), cv2.cvtColor(
        right, cv2.COLOR_GRAY2BGR
    )


def main() -> int:
    validator = load_validator()
    payload = yaml.safe_load(PRISTINE.read_text(encoding="utf-8"))
    payload["images_are_rectified"] = True
    with tempfile.TemporaryDirectory(prefix="gs130w-synthetic-depth-") as folder:
        candidate = Path(folder) / "synthetic_candidate.yaml"
        candidate.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        engine = validator.RuntimeStereoEngine(candidate, PROJECT)
        roi = (235, 585, 135, 145)
        cases = ((44, 1.0), (15, 3.0), (9, 5.0))
        for disparity, known in cases:
            left, right = shifted_pair(disparity)
            metrics, _annotated, _visual = engine.evaluate(left, right, roi, known)
            print(
                f"SYNTHETIC_{known:g}M disparity={disparity} "
                f"estimate={metrics['estimated_distance_m']:.4f} "
                f"error={metrics['relative_error']:.3%} "
                f"valid={metrics['valid_pixels']} pass={metrics['passed']}"
            )
            assert metrics["passed"], metrics
    print("SYNTHETIC_RUNTIME_STEREO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
