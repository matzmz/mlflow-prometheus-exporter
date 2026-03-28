"""Unit tests for PrometheusMetrics."""

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from mlflow_exporter.metrics import PrometheusMetrics
from mlflow_exporter.settings import MODEL_STAGES, RUN_STATUSES
from tests.helpers import make_snapshot


def _new_mock(*args: Any, **kwargs: Any) -> MagicMock:
    """Return a fresh MagicMock for each prometheus_client constructor call."""
    return MagicMock()


@pytest.fixture()
def metrics() -> Generator[PrometheusMetrics, None, None]:
    """Yield a PrometheusMetrics with prometheus_client classes replaced."""
    with (
        patch("mlflow_exporter.metrics.Gauge", side_effect=_new_mock),
        patch("mlflow_exporter.metrics.Histogram", side_effect=_new_mock),
        patch("mlflow_exporter.metrics.Counter", side_effect=_new_mock),
        patch("mlflow_exporter.metrics.Info", side_effect=_new_mock),
    ):
        yield PrometheusMetrics()


# ---------------------------------------------------------------------------
# update_snapshot
# ---------------------------------------------------------------------------


def test_update_snapshot_sets_scalar_gauges(
    metrics: PrometheusMetrics,
) -> None:
    """update_snapshot() sets each scalar Gauge to the snapshot value."""
    snapshot = make_snapshot(
        experiments_total=10,
        experiments_active_total=8,
        experiments_deleted_total=2,
        runs_total=50,
        registered_models_total=3,
        model_versions_total=15,
    )

    metrics.update_snapshot(snapshot)

    metrics.experiments_total.set.assert_called_once_with(10)
    metrics.experiments_active_total.set.assert_called_once_with(8)
    metrics.experiments_deleted_total.set.assert_called_once_with(2)
    metrics.runs_total.set.assert_called_once_with(50)
    metrics.registered_models_total.set.assert_called_once_with(3)
    metrics.model_versions_total.set.assert_called_once_with(15)


def test_update_snapshot_labels_runs_by_status(
    metrics: PrometheusMetrics,
) -> None:
    """update_snapshot() calls labels() exactly once per run status."""
    metrics.update_snapshot(make_snapshot())

    assert metrics.runs_by_status.labels.call_count == len(RUN_STATUSES)


def test_update_snapshot_labels_model_versions_by_stage(
    metrics: PrometheusMetrics,
) -> None:
    """update_snapshot() calls labels() exactly once per model stage."""
    metrics.update_snapshot(make_snapshot())

    assert metrics.model_versions_by_stage.labels.call_count == len(
        MODEL_STAGES
    )


# ---------------------------------------------------------------------------
# mark_success / mark_failure
# ---------------------------------------------------------------------------


def test_mark_success_records_duration_and_sets_flag(
    metrics: PrometheusMetrics,
) -> None:
    """mark_success() observes duration and sets collect_success to 1."""
    metrics.mark_success(1.5)

    metrics.collect_duration_seconds.observe.assert_called_once_with(1.5)
    metrics.collect_success.set.assert_called_once_with(1)
    metrics.collect_timestamp_seconds.set.assert_called_once()


def test_mark_failure_records_duration_and_increments_counter(
    metrics: PrometheusMetrics,
) -> None:
    """mark_failure() observes duration, sets collect_success to 0, increments error."""
    metrics.mark_failure(0.3)

    metrics.collect_duration_seconds.observe.assert_called_once_with(0.3)
    metrics.collect_success.set.assert_called_once_with(0)
    metrics.collect_timestamp_seconds.set.assert_called_once()
    metrics.collect_errors_total.inc.assert_called_once()


def test_mark_baseline_success_records_duration_and_sets_flag(
    metrics: PrometheusMetrics,
) -> None:
    """mark_baseline_success() observes duration and sets success to 1."""
    metrics.mark_baseline_success(2.0)

    metrics.baseline_duration_seconds.observe.assert_called_once_with(2.0)
    metrics.baseline_success.set.assert_called_once_with(1)
    metrics.baseline_timestamp_seconds.set.assert_called_once()


def test_mark_baseline_failure_records_duration_and_increments_counter(
    metrics: PrometheusMetrics,
) -> None:
    """mark_baseline_failure() records the failed baseline rebuild."""
    metrics.mark_baseline_failure(0.8)

    metrics.baseline_duration_seconds.observe.assert_called_once_with(0.8)
    metrics.baseline_success.set.assert_called_once_with(0)
    metrics.baseline_timestamp_seconds.set.assert_called_once()
    metrics.baseline_errors_total.inc.assert_called_once()
