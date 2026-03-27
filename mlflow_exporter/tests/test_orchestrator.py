#!/usr/bin/env python3

"""Unit tests for the mlflow_exporter orchestrator module."""

from contextlib import ExitStack
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from mlflow_exporter.mlflow_exporter import main, run_exporter
from mlflow_exporter.settings import ExporterSettings

_MOD = "mlflow_exporter.mlflow_exporter"


def _make_settings(**overrides: object) -> ExporterSettings:
    """Return an ExporterSettings with test-friendly defaults."""
    defaults: dict = dict(
        port=9999,
        poll_interval_seconds=30,
        tracking_uri="http://localhost:5000/",
        tracking_username=None,
        tracking_password=None,
    )
    defaults.update(overrides)
    return ExporterSettings(**defaults)


@pytest.fixture()
def mock_deps() -> Generator[dict, None, None]:
    """Patch all runtime I/O dependencies of run_exporter.

    Yields a dict with 'collector' and 'metrics' mock instances.
    The patched time.sleep raises StopIteration on the second call so
    tests can stop the infinite poll loop after two iterations.
    """
    with ExitStack() as stack:
        stack.enter_context(patch(f"{_MOD}.configure_mlflow_client"))
        mock_collector = stack.enter_context(
            patch(f"{_MOD}.MlflowObservabilityCollector")
        ).return_value
        mock_metrics = stack.enter_context(
            patch(f"{_MOD}.PrometheusMetrics")
        ).return_value
        stack.enter_context(patch(f"{_MOD}.start_http_server"))
        stack.enter_context(
            patch(
                f"{_MOD}.time.sleep",
                side_effect=[None, StopIteration()],
            )
        )
        mock_collector.collect.return_value = MagicMock()
        yield {"collector": mock_collector, "metrics": mock_metrics}


# ---------------------------------------------------------------------------
# run_exporter
# ---------------------------------------------------------------------------


def test_run_exporter_starts_http_server(mock_deps: dict) -> None:
    """run_exporter() calls start_http_server with the configured port."""
    with patch(f"{_MOD}.start_http_server") as mock_http:
        with pytest.raises(StopIteration):
            run_exporter(_make_settings(port=9999))

    mock_http.assert_called_once_with(9999)


def test_run_exporter_collects_on_each_iteration(mock_deps: dict) -> None:
    """run_exporter() calls collector.collect() on every poll cycle."""
    with pytest.raises(StopIteration):
        run_exporter(_make_settings())

    assert mock_deps["collector"].collect.call_count == 2


def test_run_exporter_marks_success_after_collection(mock_deps: dict) -> None:
    """run_exporter() calls mark_success() when collection succeeds."""
    with pytest.raises(StopIteration):
        run_exporter(_make_settings())

    mock_deps["metrics"].mark_success.assert_called()
    mock_deps["metrics"].mark_failure.assert_not_called()


def test_run_exporter_marks_failure_on_collection_error(
    mock_deps: dict,
) -> None:
    """run_exporter() calls mark_failure() when collect() raises."""
    mock_deps["collector"].collect.side_effect = RuntimeError("conn failed")

    with pytest.raises(StopIteration):
        run_exporter(_make_settings())

    mock_deps["metrics"].mark_failure.assert_called()
    mock_deps["metrics"].mark_success.assert_not_called()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_parses_args_and_calls_run_exporter() -> None:
    """main() parses CLI arguments and passes the result to run_exporter."""
    settings = _make_settings()
    with (
        patch(f"{_MOD}.parse_args", return_value=settings) as mock_parse,
        patch(f"{_MOD}.run_exporter") as mock_run,
    ):
        main()

    mock_parse.assert_called_once_with(None)
    mock_run.assert_called_once_with(settings)
