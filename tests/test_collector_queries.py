"""Unit tests for the MLflow query adapter used by the collector."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from mlflow_exporter.collector_queries import MlflowCollectorQueries
from mlflow_exporter.collector_state import (
    _ExperimentRef,
    _RunCountsByExperimentScanResult,
)
from tests.helpers import (
    FakePage,
    make_experiment,
    make_model_version,
    make_run,
)


def test_scan_runs_returns_zeroes_for_empty_experiment_list() -> None:
    """No experiment IDs means no MLflow run query is issued."""
    client = MagicMock()
    queries = MlflowCollectorQueries(client)

    result = queries._scan_runs_by_experiment([])

    assert result.counts_by_experiment == {}
    client.search_runs.assert_not_called()


def test_scan_runs_counts_runs_by_status() -> None:
    """Run scans aggregate counts per experiment and per status."""
    client = MagicMock()
    client.search_runs.return_value = FakePage(
        [
            make_run("RUNNING", experiment_id="exp1"),
            make_run("FINISHED", experiment_id="exp1"),
            make_run("FINISHED", experiment_id="exp1"),
            make_run("FAILED", experiment_id="exp1"),
        ]
    )
    queries = MlflowCollectorQueries(client)

    result = queries._scan_runs_by_experiment(["exp1"])

    assert result.counts_by_experiment["exp1"]["RUNNING"] == 1
    assert result.counts_by_experiment["exp1"]["FINISHED"] == 2
    assert result.counts_by_experiment["exp1"]["FAILED"] == 1
    assert result.counts_by_experiment["exp1"]["KILLED"] == 0


def test_scan_runs_ignores_unknown_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown MLflow run statuses are logged and ignored."""
    client = MagicMock()
    client.search_runs.return_value = FakePage(
        [make_run("UNKNOWN", experiment_id="exp1")]
    )
    queries = MlflowCollectorQueries(client)

    with caplog.at_level(logging.WARNING):
        result = queries._scan_runs_by_experiment(["exp1"])

    assert all(
        value == 0 for value in result.counts_by_experiment["exp1"].values()
    )
    assert "UNKNOWN" in caplog.text


def test_scan_all_experiments_returns_every_visible_experiment() -> None:
    """Full experiment scans return every visible experiment without filters."""
    client = MagicMock()
    client.search_experiments.return_value = FakePage(
        [
            make_experiment("exp-old", last_update_time=1),
            make_experiment("exp-new", last_update_time=2),
        ]
    )
    queries = MlflowCollectorQueries(client)

    result = queries.scan_all_experiments()

    assert result.experiments == (
        _ExperimentRef(
            experiment_id="exp-old",
            last_update_time=1,
            lifecycle_stage="active",
        ),
        _ExperimentRef(
            experiment_id="exp-new",
            last_update_time=2,
            lifecycle_stage="active",
        ),
    )


def test_scan_dirty_experiments_applies_horizon_filter_to_api() -> None:
    """Dirty-experiment scans push the horizon filter down to MLflow."""
    horizon_ms = 9_999_999
    client = MagicMock()
    client.search_experiments.return_value = FakePage(
        [make_experiment("exp-fresh", last_update_time=horizon_ms + 1)]
    )
    queries = MlflowCollectorQueries(client)

    result = queries.scan_dirty_experiments(horizon_ms)

    called_kwargs = client.search_experiments.call_args.kwargs
    assert str(horizon_ms) in called_kwargs.get("filter_string", "")
    assert result.experiments == (
        _ExperimentRef(
            experiment_id="exp-fresh",
            last_update_time=horizon_ms + 1,
            lifecycle_stage="active",
        ),
    )


def test_scan_stable_runs_by_experiment_applies_baseline_filter() -> None:
    """Stable run scans only fetch terminal runs older than the horizon."""
    client = MagicMock()
    client.search_runs.return_value = FakePage([])
    queries = MlflowCollectorQueries(client)

    queries.scan_stable_runs_by_experiment(["exp1"], horizon_ms=1234)

    called_kwargs = client.search_runs.call_args.kwargs
    assert "status != 'RUNNING'" in called_kwargs["filter_string"]
    assert "end_time <= 1234" in called_kwargs["filter_string"]


def test_scan_volatile_runs_by_experiment_merges_running_and_recent_terminal_runs() -> (
    None
):
    """Volatile scans merge running runs with recently completed terminal runs."""
    queries = MlflowCollectorQueries(MagicMock())

    with patch.object(
        queries,
        "_scan_runs_by_experiment",
        side_effect=[
            _RunCountsByExperimentScanResult(
                counts_by_experiment={
                    "exp1": {
                        "RUNNING": 1,
                        "FINISHED": 0,
                        "FAILED": 0,
                        "KILLED": 0,
                    }
                }
            ),
            _RunCountsByExperimentScanResult(
                counts_by_experiment={
                    "exp1": {
                        "RUNNING": 0,
                        "FINISHED": 2,
                        "FAILED": 0,
                        "KILLED": 0,
                    }
                }
            ),
        ],
    ) as mock_scan:
        result = queries.scan_volatile_runs_by_experiment(
            ["exp1"],
            horizon_ms=1234,
        )

    assert result.counts_by_experiment["exp1"]["RUNNING"] == 1
    assert result.counts_by_experiment["exp1"]["FINISHED"] == 2
    assert mock_scan.call_args_list[0].kwargs["filter_string"] == (
        "attributes.status = 'RUNNING'"
    )
    assert (
        "end_time > 1234"
        in mock_scan.call_args_list[1].kwargs["filter_string"]
    )


def test_scan_model_versions_counts_by_stage() -> None:
    """Model version scans aggregate totals and per-stage counts."""
    client = MagicMock()
    client.search_model_versions.return_value = FakePage(
        [
            make_model_version("Production"),
            make_model_version("Production"),
            make_model_version("Staging"),
            make_model_version("None"),
        ]
    )
    queries = MlflowCollectorQueries(client)

    result = queries.scan_model_versions()

    assert result.total == 4
    assert result.by_stage["Production"] == 2
    assert result.by_stage["Staging"] == 1
    assert result.by_stage["None"] == 1


def test_count_paginated_accumulates_across_multiple_pages() -> None:
    """The shared pagination helper sums every page."""
    page1 = FakePage(["a", "b", "c"], token="next-token")
    page2 = FakePage(["d", "e"])
    search_fn = MagicMock(side_effect=[page1, page2])

    total = MlflowCollectorQueries._count_paginated(search_fn, max_results=3)

    assert total == 5
    assert search_fn.call_count == 2


def test_count_paginated_handles_single_page() -> None:
    """The shared pagination helper works for single-page results too."""
    page = FakePage(["x", "y"])
    search_fn = MagicMock(return_value=page)

    total = MlflowCollectorQueries._count_paginated(search_fn, max_results=100)

    assert total == 2
    assert search_fn.call_count == 1
