from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rdk_patrol.factory as factory


class _FakeHbmBackend:
    instances: list["_FakeHbmBackend"] = []

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.call_count = 0
        self.instances.append(self)

    def infer(self, model_input: np.ndarray):
        self.call_count += 1
        return 0.1, np.zeros((1, 10, 4200), dtype=np.float32)


class _CalibratedStereoEstimator:
    def __init__(self) -> None:
        self.physical_left_view = "top"
        self.physical_right_view = "bottom"
        self.calibration = SimpleNamespace(
            valid=True,
            reason="ok",
            runtime_rotation="ccw90",
            left_camera_matrix=np.asarray(
                [[900.0, 0.0, 320.0], [0.0, 900.0, 640.0], [0.0, 0.0, 1.0]]
            ),
            left_distortion=np.zeros(5, dtype=np.float64),
        )

    def validate_view_mapping(
        self,
        detection_view: str,
        auxiliary_view: str,
    ) -> tuple[bool, str]:
        valid = (
            detection_view == self.physical_left_view
            and auxiliary_view == self.physical_right_view
        )
        return valid, "ok" if valid else "mapping_mismatch"

    def estimate(self, *args, **kwargs):  # pragma: no cover - worker is not started
        raise AssertionError("factory construction must not run stereo inference")


def test_factory_builds_one_backend_and_uses_calibrated_left_view(
    monkeypatch,
    tmp_path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    _FakeHbmBackend.instances.clear()
    estimator = _CalibratedStereoEstimator()
    monkeypatch.setattr(factory, "HbmRuntimeBackend", _FakeHbmBackend)
    monkeypatch.setattr(
        factory,
        "_stereo_estimator",
        lambda _config, _path: estimator,
    )

    application = factory.build_application(
        project / "configs" / "system.yaml",
        data_dir=tmp_path,
        web_enabled_override=False,
        recording_enabled_override=False,
    )

    assert len(_FakeHbmBackend.instances) == 1
    assert application.processor.inference.backend is _FakeHbmBackend.instances[0]
    assert application.frame_hub.detection_view == "top"
    assert application.frame_hub.auxiliary_view == "bottom"
    assert application.processor.physical_left_view == "top"
    assert application.processor.physical_right_view == "bottom"
    assert application.depth_worker is not None
    assert np.array_equal(
        application.point_worker.resolver.camera_matrix,
        estimator.calibration.left_camera_matrix,
    )


def test_rotation_mismatch_disables_depth_but_keeps_safe_left_view() -> None:
    config = {
        "camera": {
            "detection_view": "bottom",
            "auxiliary_view": "top",
            "rotation": "ccw90",
        }
    }
    estimator = _CalibratedStereoEstimator()
    estimator.calibration.runtime_rotation = "cw90"

    detection, auxiliary, ready, reason = factory._effective_stereo_views(
        config,
        estimator,
    )

    assert (detection, auxiliary) == ("top", "bottom")
    assert ready is False
    assert reason == "stereo_runtime_rotation_mismatch"
