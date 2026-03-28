"""Unit tests for the exporter entrypoint module."""

import signal
from unittest.mock import MagicMock, patch

from mlflow_exporter.main import build_runtime, main
from tests.helpers import make_settings

_MOD = "mlflow_exporter.main"


def test_build_runtime_composes_runtime_with_dependencies() -> None:
    """build_runtime() wires client, collector, metrics, and runtime together."""
    settings = make_settings()
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
    settings = make_settings()
    runtime = MagicMock()
    with (
        patch(f"{_MOD}.parse_args", return_value=settings) as mock_parse,
        patch(f"{_MOD}.build_runtime", return_value=runtime) as mock_build,
    ):
        main()

    mock_parse.assert_called_once_with(None)
    mock_build.assert_called_once_with(settings)
    runtime.run.assert_called_once_with()


def test_main_installs_shutdown_handler_and_restores_default_signals() -> None:
    """The entrypoint restores signal handlers even after shutdown is requested."""
    settings = make_settings()
    runtime = MagicMock()
    default_sigint = object()
    default_sigterm = object()
    installed_handlers: dict[int, object] = {}

    def _record_signal_handler(signum: int, handler: object) -> None:
        installed_handlers[signum] = handler

    def _run_and_request_shutdown() -> None:
        handler = installed_handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, object())

    runtime.run.side_effect = _run_and_request_shutdown

    with (
        patch(f"{_MOD}.parse_args", return_value=settings),
        patch(f"{_MOD}.build_runtime", return_value=runtime),
        patch(f"{_MOD}.signal.getsignal") as mock_getsignal,
        patch(f"{_MOD}.signal.signal", side_effect=_record_signal_handler),
    ):
        mock_getsignal.side_effect = [default_sigint, default_sigterm]
        main()

    assert runtime.stop.call_count == 1
    assert installed_handlers[signal.SIGINT] is default_sigint
    assert installed_handlers[signal.SIGTERM] is default_sigterm
