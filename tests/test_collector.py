"""Unit tests for MlflowObservabilityCollector."""

import logging
import time
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlflow_exporter.collector import (
    MAX_BACKOFF_SECONDS,
    MlflowObservabilityCollector,
    _backoff_interval,
    _Baseline,
    _ExperimentContribution,
    _ExperimentRef,
    _ExperimentScanResult,
)
from mlflow_exporter.models import MlflowSnapshot
from mlflow_exporter.settings import RUN_STATUSES


class FakePage(list):
    """List with an optional pagination token, mimicking MLflow PagedList."""

    def __init__(self, items: list, token: str | None = None) -> None:
        """Initialise with items and an optional next-page token."""
        super().__init__(items)
        self.token = token


def _make_experiment(
    exp_id: str,
    last_update_time: int,
    lifecycle_stage: str = "active",
) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow Experiment object."""
    return SimpleNamespace(
        experiment_id=exp_id,
        last_update_time=last_update_time,
        lifecycle_stage=lifecycle_stage,
    )


def _make_run(
    status: str,
    experiment_id: str = "exp1",
) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow Run object."""
    return SimpleNamespace(
        info=SimpleNamespace(status=status, experiment_id=experiment_id)
    )


def _make_model_version(stage: str) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow ModelVersion object."""
    return SimpleNamespace(current_stage=stage)


def _empty_client() -> MagicMock:
    """Return a mock MlflowClient that reports an empty MLflow server."""
    client = MagicMock()
    client.search_experiments.return_value = FakePage([])
    client.search_runs.return_value = FakePage([])
    client.search_model_versions.return_value = FakePage([])
    client.search_registered_models.return_value = FakePage([])
    return client


def _make_snapshot(total: int) -> MlflowSnapshot:
    """Return a minimal snapshot for state publication tests."""
    return MlflowSnapshot(
        experiments_total=total,
        experiments_active_total=total,
        experiments_deleted_total=0,
        runs_total=total,
        runs_by_status={status: total for status in RUN_STATUSES},
        registered_models_total=total,
        model_versions_total=total,
        model_versions_by_stage={},
    )


def _make_contribution(
    experiment_id: str,
    lifecycle_stage: str = "active",
    runs_by_status: Mapping[str, int] | None = None,
    last_update_time: int = 0,
) -> _ExperimentContribution:
    """Return an immutable per-experiment contribution for merge tests."""
    return _ExperimentContribution(
        experiment_id=experiment_id,
        last_update_time=last_update_time,
        lifecycle_stage=lifecycle_stage,
        runs_by_status=runs_by_status or {},
    )


def _make_baseline(horizon_ms: int = 500_000) -> _Baseline:
    """Return a reusable baseline for stateful collector tests."""
    contribution = _make_contribution(
        experiment_id="exp-old",
        lifecycle_stage="active",
        runs_by_status={
            "RUNNING": 1,
            "FINISHED": 5,
            "FAILED": 0,
            "KILLED": 0,
        },
        last_update_time=horizon_ms - 1,
    )
    return _Baseline(
        experiments_total=1,
        experiments_by_stage={"active": 1},
        runs_by_status=contribution.runs_by_status,
        experiments_by_id={contribution.experiment_id: contribution},
        registered_models_total=2,
        model_versions_total=7,
        model_versions_by_stage={
            "Production": 3,
            "Staging": 2,
            "None": 2,
            "Archived": 0,
        },
        horizon_ms=horizon_ms,
        built_at=time.monotonic(),
    )


def test_current_snapshot_requires_initialization() -> None:
    """current_snapshot() raises until bootstrap has published a snapshot."""
    collector = MlflowObservabilityCollector(_empty_client())

    with pytest.raises(RuntimeError, match="not initialized"):
        collector.current_snapshot()


def test_initialize_publishes_first_snapshot() -> None:
    """initialize() builds and publishes the first baseline."""
    collector = MlflowObservabilityCollector(_empty_client())

    snapshot = collector.initialize()

    assert collector.current_snapshot() == snapshot


def test_refresh_delta_snapshot_rebuilds_from_latest_baseline() -> None:
    """Delta refreshes always use the currently published baseline."""
    collector = MlflowObservabilityCollector(_empty_client())
    baseline = _make_baseline()
    initial_snapshot = _make_snapshot(total=1)
    collector._publish_state(baseline, initial_snapshot)
    refreshed_snapshot = _make_snapshot(total=9)

    with patch.object(
        collector,
        "_build_snapshot_from_baseline",
        return_value=refreshed_snapshot,
    ) as mock_build:
        snapshot = collector.refresh_delta_snapshot()

    mock_build.assert_called_once_with(baseline)
    assert snapshot == refreshed_snapshot
    assert collector.current_snapshot() == refreshed_snapshot


def test_refresh_delta_snapshot_returns_stale_snapshot_when_locked() -> None:
    """Delta refresh returns the last published snapshot during contention."""
    collector = MlflowObservabilityCollector(_empty_client())
    published_snapshot = _make_snapshot(total=4)
    collector._publish_state(_make_baseline(), published_snapshot)
    collector._refresh_lock.acquire()

    try:
        with patch.object(
            collector, "_build_snapshot_from_baseline"
        ) as mock_build:
            snapshot = collector.refresh_delta_snapshot()
    finally:
        collector._refresh_lock.release()

    mock_build.assert_not_called()
    assert snapshot == published_snapshot


def test_baseline_cycle_replaces_published_state() -> None:
    """A successful baseline cycle publishes a new baseline and snapshot."""
    collector = MlflowObservabilityCollector(_empty_client())
    collector._publish_state(_make_baseline(), _make_snapshot(total=1))
    new_baseline = _make_baseline(horizon_ms=900_000)
    new_snapshot = _make_snapshot(total=5)

    with (
        patch.object(collector, "_build_baseline", return_value=new_baseline),
        patch.object(
            collector,
            "_build_snapshot_from_baseline",
            return_value=new_snapshot,
        ),
    ):
        snapshot = collector._run_baseline_cycle(blocking=True)

    assert snapshot == new_snapshot
    assert collector.current_snapshot() == new_snapshot
    published_state = collector._get_published_state()
    assert published_state is not None
    assert published_state.baseline == new_baseline


def test_baseline_cycle_returns_none_when_delta_refresh_is_running() -> None:
    """Periodic baselines skip a cycle instead of overlapping MLflow I/O."""
    collector = MlflowObservabilityCollector(_empty_client())
    collector._publish_state(_make_baseline(), _make_snapshot(total=1))
    collector._refresh_lock.acquire()

    try:
        snapshot = collector._run_baseline_cycle(blocking=False)
    finally:
        collector._refresh_lock.release()

    assert snapshot is None
    assert collector.current_snapshot() == _make_snapshot(total=1)


def test_start_baseline_worker_is_idempotent() -> None:
    """The baseline worker can only be started once."""
    collector = MlflowObservabilityCollector(_empty_client())

    with patch("mlflow_exporter.collector.threading.Thread") as mock_thread:
        collector.start_baseline_worker()
        collector.start_baseline_worker()

    mock_thread.assert_called_once()


def test_run_baseline_loop_rebuilds_periodically() -> None:
    """The background loop rebuilds until a stop request is observed."""
    collector = MlflowObservabilityCollector(_empty_client())

    with (
        patch.object(
            collector,
            "_wait_for_next_baseline_cycle",
            side_effect=[False, True],
        ),
        patch.object(collector, "_run_baseline_cycle") as mock_cycle,
    ):
        collector._run_baseline_loop(
            on_snapshot=MagicMock(),
            on_failure=MagicMock(),
        )

    mock_cycle.assert_called_once_with(blocking=False)


def test_run_baseline_loop_keeps_previous_snapshot_on_failure() -> None:
    """A failed background baseline leaves the published snapshot intact."""
    collector = MlflowObservabilityCollector(_empty_client())
    published_snapshot = _make_snapshot(total=2)
    collector._publish_state(_make_baseline(), published_snapshot)

    with (
        patch.object(
            collector,
            "_wait_for_next_baseline_cycle",
            side_effect=[False, True],
        ),
        patch.object(
            collector,
            "_run_baseline_cycle",
            side_effect=RuntimeError("boom"),
        ),
    ):
        collector._run_baseline_loop(
            on_snapshot=MagicMock(),
            on_failure=MagicMock(),
        )

    assert collector.current_snapshot() == published_snapshot


def test_run_baseline_loop_reports_success_through_callback() -> None:
    """Successful baseline refreshes are reported through the callback."""
    collector = MlflowObservabilityCollector(_empty_client())
    snapshot = _make_snapshot(total=7)
    on_snapshot = MagicMock()
    on_failure = MagicMock()

    with (
        patch.object(
            collector,
            "_wait_for_next_baseline_cycle",
            side_effect=[False, True],
        ),
        patch.object(
            collector,
            "_run_baseline_cycle",
            return_value=snapshot,
        ),
    ):
        collector._run_baseline_loop(on_snapshot, on_failure)

    on_snapshot.assert_called_once()
    on_failure.assert_not_called()


def test_run_baseline_loop_reports_failure_through_callback() -> None:
    """Failed baseline refreshes are reported through the failure callback."""
    collector = MlflowObservabilityCollector(_empty_client())
    on_snapshot = MagicMock()
    on_failure = MagicMock()

    with (
        patch.object(
            collector,
            "_wait_for_next_baseline_cycle",
            side_effect=[False, True],
        ),
        patch.object(
            collector,
            "_run_baseline_cycle",
            side_effect=RuntimeError("boom"),
        ),
    ):
        collector._run_baseline_loop(on_snapshot, on_failure)

    on_snapshot.assert_not_called()
    on_failure.assert_called_once()


def test_run_delta_refresh_loop_publishes_snapshots_through_callback() -> None:
    """The delta loop reports refreshed snapshots through the success callback."""
    collector = MlflowObservabilityCollector(_empty_client())
    published_snapshot = _make_snapshot(total=3)
    on_snapshot = MagicMock()
    on_failure = MagicMock()

    with (
        patch.object(
            collector,
            "_wait_for_next_delta_cycle",
            side_effect=[False, True],
        ),
        patch.object(
            collector,
            "refresh_delta_snapshot",
            return_value=published_snapshot,
        ),
    ):
        collector.run_delta_refresh_loop(30, on_snapshot, on_failure)

    on_snapshot.assert_called_once()
    on_failure.assert_not_called()


def test_run_delta_refresh_loop_reports_failures_through_callback() -> None:
    """The delta loop reports refresh failures without crashing the state."""
    collector = MlflowObservabilityCollector(_empty_client())
    on_snapshot = MagicMock()
    on_failure = MagicMock()

    with (
        patch.object(
            collector,
            "_wait_for_next_delta_cycle",
            side_effect=[False, True],
        ),
        patch.object(
            collector,
            "refresh_delta_snapshot",
            side_effect=RuntimeError("boom"),
        ),
    ):
        collector.run_delta_refresh_loop(30, on_snapshot, on_failure)

    on_snapshot.assert_not_called()
    on_failure.assert_called_once()


def test_scan_runs_returns_zeroes_for_empty_experiment_list() -> None:
    """_scan_runs_by_experiment() with no IDs returns empty counts."""
    client = _empty_client()
    collector = MlflowObservabilityCollector(client)

    result = collector._scan_runs_by_experiment([])

    assert result == {}
    client.search_runs.assert_not_called()


def test_scan_runs_counts_runs_by_status() -> None:
    """_scan_runs_by_experiment() aggregates run counts per experiment."""
    client = MagicMock()
    client.search_runs.return_value = FakePage(
        [
            _make_run("RUNNING", experiment_id="exp1"),
            _make_run("FINISHED", experiment_id="exp1"),
            _make_run("FINISHED", experiment_id="exp1"),
            _make_run("FAILED", experiment_id="exp1"),
        ]
    )
    collector = MlflowObservabilityCollector(client)

    result = collector._scan_runs_by_experiment(["exp1"])

    assert result["exp1"]["RUNNING"] == 1
    assert result["exp1"]["FINISHED"] == 2
    assert result["exp1"]["FAILED"] == 1
    assert result["exp1"]["KILLED"] == 0


def test_scan_runs_ignores_unknown_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_scan_runs_by_experiment() logs a warning for unknown statuses."""
    client = MagicMock()
    client.search_runs.return_value = FakePage(
        [_make_run("UNKNOWN", experiment_id="exp1")]
    )
    collector = MlflowObservabilityCollector(client)

    with caplog.at_level(logging.WARNING):
        result = collector._scan_runs_by_experiment(["exp1"])

    assert all(value == 0 for value in result["exp1"].values())
    assert "UNKNOWN" in caplog.text


def test_scan_stable_experiments_classifies_by_horizon() -> None:
    """Experiments older than horizon_ms become stable baseline inputs."""
    horizon_ms = 1_000_000
    old_exp = _make_experiment("exp-old", last_update_time=horizon_ms - 1)
    new_exp = _make_experiment("exp-new", last_update_time=horizon_ms + 1)
    client = MagicMock()
    client.search_experiments.return_value = FakePage([old_exp, new_exp])
    collector = MlflowObservabilityCollector(client)

    result = collector._scan_stable_experiments(horizon_ms)

    assert result.experiments == (
        _ExperimentRef(
            experiment_id="exp-old",
            last_update_time=horizon_ms - 1,
            lifecycle_stage="active",
        ),
    )


def test_scan_stable_experiments_preserves_deleted_stage() -> None:
    """Stable experiments preserve their lifecycle stage."""
    horizon_ms = 1_000_000
    deleted_exp = _make_experiment(
        "exp-del",
        last_update_time=horizon_ms - 1,
        lifecycle_stage="deleted",
    )
    client = MagicMock()
    client.search_experiments.return_value = FakePage([deleted_exp])
    collector = MlflowObservabilityCollector(client)

    result = collector._scan_stable_experiments(horizon_ms)

    assert result.experiments == (
        _ExperimentRef(
            experiment_id="exp-del",
            last_update_time=horizon_ms - 1,
            lifecycle_stage="deleted",
        ),
    )


def test_scan_dirty_experiments_applies_horizon_filter_to_api() -> None:
    """_scan_dirty_experiments() passes the horizon filter to the API."""
    horizon_ms = 9_999_999
    fresh_exp = _make_experiment("exp-fresh", last_update_time=horizon_ms + 1)
    client = MagicMock()
    client.search_experiments.return_value = FakePage([fresh_exp])
    collector = MlflowObservabilityCollector(client)

    result = collector._scan_dirty_experiments(horizon_ms)

    called_kwargs = client.search_experiments.call_args.kwargs
    assert str(horizon_ms) in called_kwargs.get("filter_string", "")
    assert result.experiments == (
        _ExperimentRef(
            experiment_id="exp-fresh",
            last_update_time=horizon_ms + 1,
            lifecycle_stage="active",
        ),
    )


def test_scan_model_versions_counts_by_stage() -> None:
    """Model versions are totalled and aggregated per stage."""
    client = MagicMock()
    client.search_model_versions.return_value = FakePage(
        [
            _make_model_version("Production"),
            _make_model_version("Production"),
            _make_model_version("Staging"),
            _make_model_version("None"),
        ]
    )
    collector = MlflowObservabilityCollector(client)

    result = collector._scan_model_versions()

    assert result.total == 4
    assert result.by_stage["Production"] == 2
    assert result.by_stage["Staging"] == 1
    assert result.by_stage["None"] == 1


def test_count_paginated_accumulates_across_multiple_pages() -> None:
    """_count_paginated() sums entries from all pages."""
    page1 = FakePage(["a", "b", "c"], token="next-token")
    page2 = FakePage(["d", "e"])
    search_fn = MagicMock(side_effect=[page1, page2])

    total = MlflowObservabilityCollector._count_paginated(
        search_fn, max_results=3
    )

    assert total == 5
    assert search_fn.call_count == 2


def test_count_paginated_handles_single_page() -> None:
    """_count_paginated() works correctly when there is only one page."""
    page = FakePage(["x", "y"])
    search_fn = MagicMock(return_value=page)

    total = MlflowObservabilityCollector._count_paginated(
        search_fn, max_results=100
    )

    assert total == 2
    assert search_fn.call_count == 1


def test_build_snapshot_from_baseline_merges_baseline_and_delta() -> None:
    """Merged snapshots combine the current baseline with dirty experiments."""
    collector = MlflowObservabilityCollector(_empty_client())

    with (
        patch.object(
            collector,
            "_scan_dirty_experiments",
            return_value=_ExperimentScanResult(
                experiments=(
                    _ExperimentRef(
                        experiment_id="exp-new",
                        last_update_time=10**15,
                        lifecycle_stage="active",
                    ),
                )
            ),
        ),
        patch.object(
            collector,
            "_build_experiment_contributions",
            return_value=(
                _make_contribution(
                    experiment_id="exp-new",
                    lifecycle_stage="active",
                    runs_by_status={"RUNNING": 2},
                    last_update_time=10**15,
                ),
            ),
        ),
    ):
        snapshot = collector._build_snapshot_from_baseline(_make_baseline())

    assert snapshot.experiments_total == 2
    assert snapshot.experiments_active_total == 2
    assert snapshot.experiments_deleted_total == 0
    assert snapshot.runs_by_status["RUNNING"] == 3
    assert snapshot.runs_by_status["FINISHED"] == 5
    assert snapshot.registered_models_total == 2
    assert snapshot.model_versions_total == 7


def test_build_snapshot_from_baseline_replaces_stale_contribution() -> None:
    """Dirty experiments replace their old baseline contribution entirely."""
    collector = MlflowObservabilityCollector(_empty_client())
    baseline = _make_baseline()

    with (
        patch.object(
            collector,
            "_scan_dirty_experiments",
            return_value=_ExperimentScanResult(
                experiments=(
                    _ExperimentRef(
                        experiment_id="exp-old",
                        last_update_time=baseline.horizon_ms + 1,
                        lifecycle_stage="active",
                    ),
                )
            ),
        ),
        patch.object(
            collector,
            "_build_experiment_contributions",
            return_value=(
                _make_contribution(
                    experiment_id="exp-old",
                    lifecycle_stage="active",
                    runs_by_status={"RUNNING": 0, "FINISHED": 6},
                    last_update_time=baseline.horizon_ms + 1,
                ),
            ),
        ),
    ):
        snapshot = collector._build_snapshot_from_baseline(baseline)

    assert snapshot.experiments_total == 1
    assert snapshot.runs_by_status["RUNNING"] == 0
    assert snapshot.runs_by_status["FINISHED"] == 6


def test_published_snapshot_is_immutable() -> None:
    """Published snapshots reject accidental mutation of nested counts."""
    snapshot = _make_snapshot(total=1)

    with pytest.raises(TypeError):
        snapshot.runs_by_status["RUNNING"] = 99


def test_backoff_interval_returns_base_on_zero_failures() -> None:
    """No failures means the original interval is used."""
    assert _backoff_interval(30, 0) == 30


def test_backoff_interval_doubles_on_each_failure() -> None:
    """Consecutive failures successively double the wait interval."""
    assert _backoff_interval(30, 1) == 60
    assert _backoff_interval(30, 2) == 120
