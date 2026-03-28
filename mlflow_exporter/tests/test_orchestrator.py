#!/usr/bin/env python3

"""Unit tests for the mlflow_exporter orchestrator module."""

from unittest.mock import MagicMock, patch

from mlflow_exporter.main import build_runtime, main
from mlflow_exporter.settings import ExporterSettings

_MOD = "mlflow_exporter.main"


def _make_settings(**overrides: object) -> ExporterSettings:
    """Return an ExporterSettings with test-friendly defaults."""
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


def test_build_runtime_composes_runtime_with_dependencies() -> None:
    """build_runtime() wires client, collector, metrics, and runtime together."""
    settings = _make_settings()
    mock_client = object()
    with (
        patch(f"{_MOD}.configure_mlflow_client", return_value=mock_client),
        patch(f"{_MOD}.MlflowObservabilityCollector") as mock_collector_cls,
        patch(f"{_MOD}.PrometheusMetrics") as mock_metrics_cls,
        patch(f"{_MOD}.ExporterRuntime") as mock_runtime_cls,
    ):
        build_runtime(settings)

    mock_collector_cls.assert_called_once_with(
        mock_client,
        baseline_interval_seconds=settings.baseline_interval_seconds,
    )
    mock_metrics_cls.assert_called_once_with()
    mock_runtime_cls.assert_called_once()
    mock_runtime_cls.return_value.run.assert_not_called()


def test_main_parses_args_builds_runtime_and_runs_it() -> None:
    """main() parses args, builds a runtime, and runs it."""
    settings = _make_settings()
    runtime = MagicMock()
    with (
        patch(f"{_MOD}.parse_args", return_value=settings) as mock_parse,
        patch(f"{_MOD}.build_runtime", return_value=runtime) as mock_build,
    ):
        main()

    mock_parse.assert_called_once_with(None)
    mock_build.assert_called_once_with(settings)
    runtime.run.assert_called_once_with()
