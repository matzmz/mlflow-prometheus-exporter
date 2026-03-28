#!/usr/bin/env python3

"""Unit tests for the exporter runtime service."""

from typing import cast
from unittest.mock import MagicMock

import pytest

from mlflow_exporter.runtime import ExporterRuntime
from mlflow_exporter.settings import ExporterSettings, MlflowSnapshot


def _make_settings(**overrides: object) -> ExporterSettings:
    """Return runtime settings with test-friendly defaults."""
    defaults: dict = dict(
        port=9999,
        listen_address="0.0.0.0",
        poll_interval_seconds=30,
        baseline_interval_seconds=3600,
        tracking_uri="http://localhost:5000/",
        tracking_username=None,
        tracking_password=None,
    )
    defaults.update(overrides)
    return ExporterSettings(**defaults)


def _make_snapshot() -> MlflowSnapshot:
    """Return a minimal snapshot for runtime publication tests."""
    return MlflowSnapshot(
        experiments_total=1,
        experiments_active_total=1,
        experiments_deleted_total=0,
        runs_total=1,
        runs_by_status={},
        registered_models_total=1,
        model_versions_total=1,
        model_versions_by_stage={},
    )


def test_runtime_waits_for_bootstrap_before_starting_http_server() -> None:
    """The HTTP server starts only after the initial baseline is published."""
    collector = MagicMock()
    metrics = MagicMock()
    start_server = MagicMock()
    snapshot = _make_snapshot()
    collector.initialize.return_value = snapshot
    events: list[str] = []

    def _record_initialize() -> MlflowSnapshot:
        events.append("init")
        return snapshot

    collector.initialize.side_effect = _record_initialize
    start_server.side_effect = lambda _port, **_kwargs: events.append("http")

    runtime = ExporterRuntime(
        settings=_make_settings(),
        collector=collector,
        metrics=metrics,
        start_http_server=start_server,
    )
    collector.run_delta_refresh_loop.side_effect = StopIteration()

    with pytest.raises(StopIteration):
        runtime.run()

    assert events[:2] == ["init", "http"]


def test_runtime_publishes_bootstrap_snapshot_and_starts_workers() -> None:
    """The runtime publishes the first snapshot before handing off control."""
    collector = MagicMock()
    metrics = MagicMock()
    start_server = MagicMock()
    snapshot = _make_snapshot()
    collector.initialize.return_value = snapshot
    collector.run_delta_refresh_loop.side_effect = StopIteration()

    runtime = ExporterRuntime(
        settings=_make_settings(),
        collector=collector,
        metrics=metrics,
        start_http_server=start_server,
    )

    with pytest.raises(StopIteration):
        runtime.run()

    metrics.update_snapshot.assert_called_once_with(snapshot)
    metrics.mark_success.assert_called_once()
    metrics.mark_baseline_success.assert_called_once()
    collector.start_baseline_worker_with_callbacks.assert_called_once()
    start_server.assert_called_once_with(9999, addr="0.0.0.0")


def test_runtime_delegates_delta_loop_with_runtime_callbacks() -> None:
    """The runtime hands callback ownership to the collector delta loop."""
    collector = MagicMock()
    metrics = MagicMock()
    start_server = MagicMock()
    snapshot = _make_snapshot()
    collector.initialize.return_value = snapshot

    runtime = ExporterRuntime(
        settings=_make_settings(poll_interval_seconds=45),
        collector=collector,
        metrics=metrics,
        start_http_server=start_server,
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
        settings=_make_settings(),
        collector=MagicMock(),
        metrics=MagicMock(),
        start_http_server=MagicMock(),
    )
    snapshot = _make_snapshot()
    metrics = cast(MagicMock, runtime._metrics)

    runtime._publish_snapshot(snapshot, 0.7)

    metrics.update_snapshot.assert_called_once_with(snapshot)
    metrics.mark_success.assert_called_once_with(0.7)


def test_publish_baseline_snapshot_updates_baseline_health() -> None:
    """Baseline publication updates both snapshot and baseline health metrics."""
    runtime = ExporterRuntime(
        settings=_make_settings(),
        collector=MagicMock(),
        metrics=MagicMock(),
        start_http_server=MagicMock(),
    )
    snapshot = _make_snapshot()
    metrics = cast(MagicMock, runtime._metrics)

    runtime._publish_baseline_snapshot(snapshot, 1.2)

    metrics.update_snapshot.assert_called_once_with(snapshot)
    metrics.mark_success.assert_called_once_with(1.2)
    metrics.mark_baseline_success.assert_called_once_with(1.2)


def test_runtime_stops_collector_when_run_exits() -> None:
    """The runtime always stops collector-owned loops on exit."""
    collector = MagicMock()
    metrics = MagicMock()
    start_server = MagicMock()
    collector.initialize.return_value = _make_snapshot()
    collector.run_delta_refresh_loop.side_effect = StopIteration()
    runtime = ExporterRuntime(
        settings=_make_settings(),
        collector=collector,
        metrics=metrics,
        start_http_server=start_server,
    )

    with pytest.raises(StopIteration):
        runtime.run()

    collector.stop.assert_called_once_with()
