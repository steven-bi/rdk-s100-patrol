from __future__ import annotations

import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rdk_patrol.alarms import AlarmPublisher, AlarmRepository
from rdk_patrol.config import DEFAULTS, load_config
from rdk_patrol.contracts import AlarmCandidate
from rdk_patrol.web import ReadOnlyAlarmServer, export_offline_html


def _candidate(
    event_name: str = "垃圾桶已满",
    point_name: str = "垃圾桶一号",
    occurred_at: float = 1_700_000_000.0,
) -> AlarmCandidate:
    frame = np.zeros((180, 300, 3), dtype=np.uint8)
    frame[50:130, 80:220] = (30, 80, 220)
    return AlarmCandidate(
        event_name=event_name,
        occurred_at=occurred_at,
        frame_bgr=frame,
        point_name=point_name,
        evidence={
            "boxes": [{"box": [80, 50, 220, 130], "label": "垃圾桶"}],
            "review_result": "已满",
        },
    )


def test_alarm_repository_strict_order_and_atomic_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repository = AlarmRepository(tmp)
        record = repository.save(_candidate())

        assert list(record) == [
            "keyframe_image",
            "event_name",
            "beijing_time",
            "point_name",
        ]
        assert record["event_name"] == "垃圾桶已满"
        assert record["point_name"] == "垃圾桶一号"
        image_path = Path(tmp) / record["keyframe_image"]
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        latest = json.loads((Path(tmp) / "latest.json").read_text(encoding="utf-8"))
        assert list(latest) == list(record)
        rows = (Path(tmp) / "records.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1
        assert list(json.loads(rows[0])) == list(record)
        sidecars = list((Path(tmp) / "sidecars").rglob("*.json"))
        assert len(sidecars) == 1
        assert list(json.loads(sidecars[0].read_text(encoding="utf-8"))) == list(record)


def test_non_point_alarm_omits_point_and_cooldown_is_durable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        now = [1000.0]
        repository = AlarmRepository(tmp)
        publisher = AlarmPublisher(repository, clock=lambda: now[0])
        candidate = _candidate(
            event_name="明火报警",
            point_name=None,  # type: ignore[arg-type]
        )
        candidate.evidence = {"distance_m": None, "confidence": 0.91}

        first = publisher.try_publish(candidate)
        second = publisher.try_publish(candidate)
        assert first is not None
        assert list(first) == ["keyframe_image", "event_name", "beijing_time"]
        assert second is None
        now[0] += 600.0
        assert publisher.try_publish(candidate) is not None


def test_patrol_web_defaults_to_8081_and_avoids_camera_port() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "system.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        server = ReadOnlyAlarmServer(AlarmRepository(tmp))

    assert DEFAULTS["web"]["port"] == 8081
    assert config["web"]["port"] == 8081
    assert server.port == 8081


def test_offline_export_embeds_images_and_readonly_http() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repository = AlarmRepository(Path(tmp) / "alarms")
        saved = repository.save(_candidate())
        offline_path = export_offline_html(Path(tmp) / "handover.html", repository)
        offline = offline_path.read_text(encoding="utf-8")
        assert "data:image/jpeg;base64," in offline
        assert "垃圾桶已满" in offline
        assert "http://" not in offline and "https://" not in offline

        server = ReadOnlyAlarmServer(
            repository,
            host="127.0.0.1",
            port=0,
            health_provider=lambda: {
                "status": "degraded",
                "metrics": {"measured_fps": 14.5, "target_fps": 15},
                "components": {"camera": {"state": "ok", "metrics": {}}},
            },
        )
        host, port = server.start_background()
        base = "http://{}:{}".format(host, port)
        try:
            with urllib.request.urlopen(base + "/health", timeout=3.0) as response:
                health = json.loads(response.read().decode("utf-8"))
            assert health["read_only"] is True
            assert health["alarm_count"] == 1
            assert health["status"] == "degraded"
            assert health["metrics"]["measured_fps"] == 14.5

            with urllib.request.urlopen(base + "/api/alarms/latest", timeout=3.0) as response:
                latest = json.loads(response.read().decode("utf-8"))
            assert list(latest) == list(saved)

            with urllib.request.urlopen(base + "/api/alarms?limit=10", timeout=3.0) as response:
                listing = json.loads(response.read().decode("utf-8"))
            assert listing["count"] == 1
            assert list(listing["alarms"][0]) == list(saved)

            with urllib.request.urlopen(
                base + "/" + saved["keyframe_image"], timeout=3.0
            ) as response:
                assert response.headers.get_content_type() == "image/jpeg"
                assert response.read(2) == b"\xff\xd8"

            with urllib.request.urlopen(base + "/download/offline.html", timeout=3.0) as response:
                downloaded = response.read().decode("utf-8")
            assert "data:image/jpeg;base64," in downloaded
            with urllib.request.urlopen(base + "/", timeout=3.0) as response:
                dashboard = response.read().decode("utf-8")
            assert "health-components" in dashboard
            assert "实测 FPS" in dashboard
            assert 'fetch("/health"' in dashboard

            for method in ("HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
                request = urllib.request.Request(
                    base + "/api/alarms",
                    data=b"{}" if method in {"POST", "PUT", "PATCH"} else None,
                    method=method,
                )
                try:
                    urllib.request.urlopen(request, timeout=3.0)
                    raise AssertionError("{} unexpectedly succeeded".format(method))
                except urllib.error.HTTPError as exc:
                    assert exc.code == 405
        finally:
            server.stop()
