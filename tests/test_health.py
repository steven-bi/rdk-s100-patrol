from __future__ import annotations

import json

from rdk_patrol.health import HealthFileReporter, HealthRegistry, RollingRate


def test_rolling_rate() -> None:
    rate = RollingRate(window_seconds=2.0)
    rate.mark(10.0)
    rate.mark(10.5)
    rate.mark(11.0)
    assert rate.rate(11.0) == 2.0
    assert rate.count(13.1) == 0


def test_stale_required_component_fails_health() -> None:
    registry = HealthRegistry()
    registry.register("camera", stale_after_seconds=1.0)
    registry.update("camera", timestamp=10.0)
    snapshot = registry.snapshot(now_monotonic=12.0, now_wall=0.0)
    assert snapshot["status"] == "failed"
    assert snapshot["components"]["camera"]["state"] == "failed"


def test_optional_failure_only_degrades() -> None:
    registry = HealthRegistry()
    registry.register("ding_talk", required=False)
    registry.failure("ding_talk", "disabled", failed=True)
    assert registry.snapshot()["status"] == "degraded"


def test_health_file_reporter_writes_utf8_json(tmp_path) -> None:
    registry = HealthRegistry()
    registry.register("runtime")
    registry.heartbeat("runtime", message="运行中")
    destination = tmp_path / "status" / "health.json"
    reporter = HealthFileReporter(registry, destination)
    reporter.write_once()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["components"]["runtime"]["message"] == "运行中"
