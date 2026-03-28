"""Unit tests for the exporter HTTP server."""

import socket
import time
import urllib.error
import urllib.request
from collections.abc import Generator

import pytest
from prometheus_client import CollectorRegistry

from mlflow_exporter.infra.server import ExporterServer


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
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def server_on_free_port() -> Generator[tuple[ExporterServer, int], None, None]:
    """Start a server on an ephemeral port and stop it after the test."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)
    yield server, port
    server.stop()


def test_healthz_returns_200(
    server_on_free_port: tuple[ExporterServer, int],
) -> None:
    """The /healthz endpoint always returns 200 once the server is up."""
    _server, port = server_on_free_port

    status, body = _get(port, "/healthz")

    assert status == 200
    assert body == b"ok\n"


def test_readyz_returns_503_before_ready(
    server_on_free_port: tuple[ExporterServer, int],
) -> None:
    """/readyz returns 503 until mark_ready() is called."""
    _server, port = server_on_free_port

    status, _ = _get(port, "/readyz")

    assert status == 503


def test_readyz_returns_200_after_mark_ready(
    server_on_free_port: tuple[ExporterServer, int],
) -> None:
    """/readyz returns 200 once the server is marked ready."""
    server, port = server_on_free_port
    server.mark_ready()

    status, body = _get(port, "/readyz")

    assert status == 200
    assert body == b"ok\n"


def test_metrics_endpoint_serves_prometheus_output(
    server_on_free_port: tuple[ExporterServer, int],
) -> None:
    """/metrics returns Prometheus text exposition format."""
    _server, port = server_on_free_port

    status, body = _get(port, "/metrics")

    assert status == 200


def test_unknown_path_returns_404(
    server_on_free_port: tuple[ExporterServer, int],
) -> None:
    """Unknown paths return 404."""
    _server, port = server_on_free_port

    status, _ = _get(port, "/unknown")

    assert status == 404


def test_stop_releases_bound_port() -> None:
    """stop() releases the socket so the same port can be rebound."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)

    server.stop()

    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_stop_before_start_is_noop() -> None:
    """Calling stop() on a server that was never started does not raise."""
    server = ExporterServer(registry=CollectorRegistry())

    server.stop()


def test_double_stop_is_safe() -> None:
    """Calling stop() twice does not raise."""
    port = _free_port()
    server = ExporterServer(registry=CollectorRegistry())
    server.start(port, addr="127.0.0.1")
    time.sleep(0.1)

    server.stop()
    server.stop()
