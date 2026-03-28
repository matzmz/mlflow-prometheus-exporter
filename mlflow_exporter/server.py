"""HTTP server with /healthz, /readyz, and /metrics endpoints."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    generate_latest,
)


class _ExporterHandler(BaseHTTPRequestHandler):
    """Serve health probes alongside Prometheus metrics."""

    _ready: threading.Event
    _registry: CollectorRegistry

    def do_GET(self) -> None:  # noqa: N802
        """Route GET requests to the appropriate handler."""
        if self.path == "/healthz":
            self._text_response(200, b"ok\n")
        elif self.path == "/readyz":
            if self._ready.is_set():
                self._text_response(200, b"ok\n")
            else:
                self._text_response(503, b"not ready\n")
        elif self.path == "/metrics":
            output = generate_latest(self._registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        else:
            self._text_response(404, b"not found\n")

    def _text_response(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default stderr access logging."""
        return


class ExporterServer:
    """HTTP server that exposes health probes and Prometheus metrics."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        """Prepare server state without starting any thread."""
        self._registry = registry
        self._ready = threading.Event()
        self._httpd: Optional[HTTPServer] = None

    def start(self, port: int, addr: str = "0.0.0.0") -> None:
        """Bind and start the HTTP server in a daemon thread."""
        handler_class = type(
            "_BoundHandler",
            (_ExporterHandler,),
            {"_ready": self._ready, "_registry": self._registry},
        )
        self._httpd = HTTPServer((addr, port), handler_class)
        thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="http",
        )
        thread.start()

    def mark_ready(self) -> None:
        """Signal that the exporter has completed bootstrap."""
        self._ready.set()
