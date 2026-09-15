from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from rdk_patrol.alarms import AlarmRepository

from .offline import build_live_dashboard_html, build_offline_html


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _ReadOnlyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ReadOnlyAlarmServer:
    """Unauthenticated LAN viewer exposing GET-only, read-only resources."""

    def __init__(
        self,
        repository: AlarmRepository,
        host: str = "0.0.0.0",
        port: int = 8081,
        health_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> None:
        self.repository = repository
        self.host = str(host)
        self.port = int(port)
        self.health_provider = health_provider
        self._httpd: Optional[_ReadOnlyHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Optional[Tuple[str, int]]:
        if self._httpd is None:
            return None
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        if self._httpd is not None:
            raise RuntimeError("server is already running")
        self._httpd = _ReadOnlyHTTPServer(
            (self.host, self.port),
            self._handler_class(),
        )
        self._httpd.serve_forever(poll_interval=0.25)

    def start_background(self) -> Tuple[str, int]:
        if self._httpd is not None:
            raise RuntimeError("server is already running")
        self._httpd = _ReadOnlyHTTPServer(
            (self.host, self.port),
            self._handler_class(),
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="rdk-alarm-readonly-http",
            daemon=True,
        )
        self._thread.start()
        address = self.address
        assert address is not None
        return address

    def stop(self) -> None:
        httpd = self._httpd
        thread = self._thread
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._httpd = None
        self._thread = None

    def _handler_class(self) -> type:
        repository = self.repository
        health_provider = self.health_provider

        class Handler(BaseHTTPRequestHandler):
            server_version = "RDKPatrolReadOnly/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                parsed = urlsplit(self.path)
                path = parsed.path
                if path == "/":
                    self._send(
                        HTTPStatus.OK,
                        build_live_dashboard_html().encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/health":
                    snapshot: Dict[str, Any] = {}
                    if health_provider is not None:
                        try:
                            provided = health_provider()
                            if isinstance(provided, Mapping):
                                snapshot.update(dict(provided))
                            else:
                                raise TypeError("health_provider must return a mapping")
                        except Exception as exc:
                            snapshot.update(
                                {
                                    "status": "degraded",
                                    "health_provider_error": "{}: {}".format(
                                        type(exc).__name__, exc
                                    ),
                                }
                            )
                    snapshot.setdefault("status", "ok")
                    snapshot.setdefault("service", "rdk-s100-patrol-alarm-viewer")
                    snapshot["read_only"] = True
                    snapshot["alarm_count"] = len(repository.list_records())
                    self._send_json(
                        HTTPStatus.OK,
                        snapshot,
                    )
                    return
                if path == "/api/alarms/latest":
                    self._send_json(HTTPStatus.OK, repository.latest())
                    return
                if path == "/api/alarms":
                    query = parse_qs(parsed.query, keep_blank_values=False)
                    try:
                        limit = min(1000, max(0, int(query.get("limit", ["200"])[0])))
                    except ValueError:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "limit must be an integer"},
                        )
                        return
                    event_name = query.get("event_name", [None])[0]
                    point_name = query.get("point_name", [None])[0]
                    records = repository.list_records(
                        limit=limit,
                        event_name=event_name,
                        point_name=point_name,
                        newest_first=True,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {"count": len(records), "alarms": records},
                    )
                    return
                if path.startswith("/images/"):
                    relative = unquote(path.lstrip("/"))
                    image_path = repository.resolve_image(relative)
                    if image_path is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "image not found"})
                        return
                    content_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
                    self._send(
                        HTTPStatus.OK,
                        image_path.read_bytes(),
                        content_type,
                        extra_headers={"Cache-Control": "private, max-age=60"},
                    )
                    return
                if path == "/download/alarms.jsonl":
                    self._send(
                        HTTPStatus.OK,
                        repository.records_bytes(),
                        "application/x-ndjson; charset=utf-8",
                        extra_headers={
                            "Content-Disposition": 'attachment; filename="alarms.jsonl"'
                        },
                    )
                    return
                if path == "/download/offline.html":
                    self._send(
                        HTTPStatus.OK,
                        build_offline_html(repository).encode("utf-8"),
                        "text/html; charset=utf-8",
                        extra_headers={
                            "Content-Disposition": 'attachment; filename="alarm-ledger.html"'
                        },
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_HEAD(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_POST(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_PUT(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_PATCH(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_DELETE(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def _method_not_allowed(self) -> None:
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "read-only service; GET is the only allowed method"},
                    extra_headers={"Allow": "GET"},
                )

            def _send_json(
                self,
                status: HTTPStatus,
                value: Any,
                extra_headers: Optional[Dict[str, str]] = None,
            ) -> None:
                self._send(
                    status,
                    _json_bytes(value),
                    "application/json; charset=utf-8",
                    extra_headers=extra_headers,
                )

            def _send(
                self,
                status: HTTPStatus,
                payload: bytes,
                content_type: str,
                extra_headers: Optional[Dict[str, str]] = None,
            ) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                for key, value in (extra_headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                # Integration may attach its own structured HTTP logger.
                return

        return Handler
