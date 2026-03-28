"""Unit tests for the exporter HTTP server."""

import threading
import time
import urllib.request

from prometheus_client import CollectorRegistry

from mlflow_exporter.server import ExporterServer


def _get(port: int, path: str) -> tuple[int, bytes]:
    """Issue a GET request and return (status_code, body)."""
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=2
        )
        return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _free_port() -> int:
    """Reserve an ephemeral TCP port."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_healthz_returns_200() -> None:
    """The /healthz endpoint always returns 200 once the server is up."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)

    status, body = _get(port, "/healthz")

    assert status == 200
    assert body == b"ok\n"


def test_readyz_returns_503_before_ready() -> None:
    """/readyz returns 503 until mark_ready() is called."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)

    status, _ = _get(port, "/readyz")

    assert status == 503


def test_readyz_returns_200_after_mark_ready() -> None:
    """/readyz returns 200 once the server is marked ready."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)
    server.mark_ready()

    status, body = _get(port, "/readyz")

    assert status == 200
    assert body == b"ok\n"


def test_metrics_endpoint_serves_prometheus_output() -> None:
    """/metrics returns Prometheus text exposition format."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)

    status, body = _get(port, "/metrics")

    assert status == 200


def test_unknown_path_returns_404() -> None:
    """Unknown paths return 404."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)

    status, _ = _get(port, "/unknown")

    assert status == 404
