from __future__ import annotations

from pathlib import Path

from rdk_patrol.preflight import EXPECTED_HBM_SHA256, run_preflight


def test_preflight_detects_bundled_model_and_pending_field_work() -> None:
    project = Path(__file__).resolve().parents[1]
    report = run_preflight(project / "configs" / "system.yaml")
    assert report.ok
    model = next(result for result in report.results if result.name == "HBM model")
    assert model.details["sha256"] == EXPECTED_HBM_SHA256
    stereo = next(
        result for result in report.results if result.name == "stereo readiness"
    )
    assert stereo.status == "warning"
