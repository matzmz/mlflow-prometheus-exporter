"""Unit tests for the exporter runtime service."""

from typing import cast
from unittest.mock import MagicMock

import pytest

from mlflow_exporter.models import MlflowSnapshot
from mlflow_exporter.runtime import ExporterRuntime
from tests.helpers import make_settings, make_snapshot


def test_runtime_starts_server_before_bootstrap() -> None:
    """The HTTP server starts before the initial baseline is built."""
    collector = MagicMock()
    metrics = MagicMock()
    server = MagicMock()
    snapshot = make_snapshot()
    collector.initialize.return_value = snapshot
    events: list[str] = []

    server.start.side_effect = lambda _port, **_kw: events.append("http")

    def _record_initialize() -> MlflowSnapshot:
        events.append("init")
        return snapshot

    collector.initialize.side_effect = _record_initialize
    collector.run_delta_refresh_loop.side_effect = StopIteration()

    runtime = ExporterRuntime(
        settings=make_settings(),
        collector=collector,
        metrics=metrics,
        server=server,
    )

    with pytest.raises(StopIteration):
        runtime.run()

    assert events[:2] == ["http", "init"]


def test_runtime_marks_ready_after_bootstrap() -> None:
    """The server is marked ready only after bootstrap is published."""
    collector = MagicMock()
    metrics = MagicMock()
    server = MagicMock()
    snapshot = make_snapshot()
    collector.initialize.return_value = snapshot
    collector.run_delta_refresh_loop.side_effect = StopIteration()

    runtime = ExporterRuntime(
        settings=make_settings(),
        collector=collector,
        metrics=metrics,
        server=server,
    )

    with pytest.raises(StopIteration):
        runtime.run()

    server.start.assert_called_once_with(9999, addr="0.0.0.0")
    server.mark_ready.assert_called_once()


def test_runtime_publishes_bootstrap_snapshot_and_starts_workers() -> None:
    """The runtime publishes the first snapshot before handing off control."""
    collector = MagicMock()
    metrics = MagicMock()
    server = MagicMock()
    snapshot = make_snapshot()
    collector.initialize.return_value = snapshot
    collector.run_delta_refresh_loop.side_effect = StopIteration()

    runtime = ExporterRuntime(
        settings=make_settings(),
        collector=collector,
        metrics=metrics,
        server=server,
    )

    with pytest.raises(StopIteration):
        runtime.run()

    metrics.update_snapshot.assert_called_once_with(snapshot)
    metrics.mark_success.assert_called_once()
    metrics.mark_baseline_success.assert_called_once()
    collector.start_baseline_worker_with_callbacks.assert_called_once()


def test_runtime_delegates_delta_loop_with_runtime_callbacks() -> None:
    """The runtime hands callback ownership to the collector delta loop."""
    collector = MagicMock()
    metrics = MagicMock()
    server = MagicMock()
    snapshot = make_snapshot()
    collector.initialize.return_value = snapshot

    runtime = ExporterRuntime(
        settings=make_settings(poll_interval_seconds=45),
        collector=collector,
        metrics=metrics,
        server=server,
    )
    collector.run_delta_refresh_loop.side_effect = StopIteration()

    with pytest.raises(StopIteration):
        runtime.run()

    collector.run_delta_refresh_loop.assert_called_once()
    kwargs = collector.run_delta_refresh_loop.call_args.kwargs
    assert kwargs["poll_interval_seconds"] == 45
    assert kwargs["on_snapshot"] == runtime._publish_snapshot
    assert kwargs["on_failure"] == metrics.mark_failure


def test_publish_snapshot_updates_metrics_and_marks_success() -> None:
    """The runtime snapshot publisher keeps metrics concerns local."""
    runtime = ExporterRuntime(
        settings=make_settings(),
        collector=MagicMock(),
        metrics=MagicMock(),
        server=MagicMock(),
    )
    snapshot = make_snapshot()
    metrics = cast(MagicMock, runtime._metrics)

    runtime._publish_snapshot(snapshot, 0.7)

    metrics.update_snapshot.assert_called_once_with(snapshot)
    metrics.mark_success.assert_called_once_with(0.7)


def test_publish_baseline_snapshot_updates_baseline_health() -> None:
    """Baseline publication updates both snapshot and baseline health metrics."""
    runtime = ExporterRuntime(
        settings=make_settings(),
        collector=MagicMock(),
        metrics=MagicMock(),
        server=MagicMock(),
    )
    snapshot = make_snapshot()
    metrics = cast(MagicMock, runtime._metrics)

    runtime._publish_baseline_snapshot(snapshot, 1.2)

    metrics.update_snapshot.assert_called_once_with(snapshot)
    metrics.mark_success.assert_called_once_with(1.2)
    metrics.mark_baseline_success.assert_called_once_with(1.2)


def test_runtime_stops_collector_when_run_exits() -> None:
    """The runtime always stops collector-owned loops on exit."""
    collector = MagicMock()
    metrics = MagicMock()
    server = MagicMock()
    collector.initialize.return_value = make_snapshot()
    collector.run_delta_refresh_loop.side_effect = StopIteration()
    runtime = ExporterRuntime(
        settings=make_settings(),
        collector=collector,
        metrics=metrics,
        server=server,
    )

    with pytest.raises(StopIteration):
        runtime.run()

    collector.stop.assert_called_once_with()
