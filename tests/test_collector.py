"""Unit tests for collector runtime coordination."""

import time
from collections.abc import Mapping
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from mlflow_exporter.collector import (
    MAX_BACKOFF_SECONDS,
    MlflowObservabilityCollector,
    _backoff_interval,
    _Baseline,
    _ExperimentBaseline,
    _ExperimentRef,
    _ExperimentScanResult,
    _ModelVersionScanResult,
    _RunCountsByExperimentScanResult,
)
from mlflow_exporter.config.settings import RUN_STATUSES
from mlflow_exporter.models import MlflowSnapshot
from tests.helpers import FakePage


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


def _make_experiment_baseline(
    experiment_id: str,
    lifecycle_stage: str = "active",
    stable_runs_by_status: Mapping[str, int] | None = None,
    last_update_time: int = 0,
) -> _ExperimentBaseline:
    """Return an immutable baseline entry for merge tests."""
    return _ExperimentBaseline(
        experiment_id=experiment_id,
        last_update_time=last_update_time,
        lifecycle_stage=lifecycle_stage,
        stable_runs_by_status=stable_runs_by_status or {},
    )


def _make_baseline(horizon_ms: int = 500_000) -> _Baseline:
    """Return a reusable baseline for stateful collector tests."""
    experiment = _make_experiment_baseline(
        experiment_id="exp-old",
        lifecycle_stage="active",
        stable_runs_by_status={
            "RUNNING": 0,
            "FINISHED": 5,
            "FAILED": 0,
            "KILLED": 0,
        },
        last_update_time=horizon_ms - 1,
    )
    return _Baseline(
        experiments_by_id={experiment.experiment_id: experiment},
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

    with patch(
        "mlflow_exporter.collector.coordinator.threading.Thread"
    ) as mock_thread:
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


def test_build_baseline_composes_query_results_into_immutable_state() -> None:
    """The runtime collector assembles a baseline from query-adapter outputs."""
    collector = MlflowObservabilityCollector(_empty_client())
    experiments = _ExperimentScanResult(
        experiments=(
            _ExperimentRef(
                experiment_id="exp-old",
                last_update_time=100,
                lifecycle_stage="active",
            ),
            _ExperimentRef(
                experiment_id="exp-new",
                last_update_time=200,
                lifecycle_stage="deleted",
            ),
        )
    )
    model_versions = _ModelVersionScanResult(
        total=4,
        by_stage={
            "Production": 2,
            "Staging": 1,
            "None": 1,
            "Archived": 0,
        },
    )

    with (
        patch.object(collector, "_horizon_ms", return_value=1234),
        patch.object(
            collector._queries,
            "scan_all_experiments",
            return_value=experiments,
        ),
        patch.object(
            collector._queries,
            "scan_stable_runs_by_experiment",
            return_value=_RunCountsByExperimentScanResult(
                counts_by_experiment={
                    "exp-old": {
                        "RUNNING": 0,
                        "FINISHED": 5,
                        "FAILED": 1,
                        "KILLED": 0,
                    }
                }
            ),
        ) as mock_stable_runs,
        patch.object(
            collector._queries,
            "scan_model_versions",
            return_value=model_versions,
        ),
        patch.object(
            collector._queries,
            "count_registered_models",
            return_value=2,
        ),
    ):
        baseline = collector._build_baseline()

    assert tuple(mock_stable_runs.call_args.args[0]) == ("exp-old", "exp-new")
    assert mock_stable_runs.call_args.args[1] == 1234
    assert baseline.horizon_ms == 1234
    assert baseline.registered_models_total == 2
    assert baseline.model_versions_total == 4
    assert baseline.experiments_by_id["exp-old"].stable_runs_by_status == {
        "RUNNING": 0,
        "FINISHED": 5,
        "FAILED": 1,
        "KILLED": 0,
    }
    assert baseline.experiments_by_id["exp-new"].lifecycle_stage == "deleted"
    assert all(
        value == 0
        for value in baseline.experiments_by_id[
            "exp-new"
        ].stable_runs_by_status.values()
    )


def test_build_snapshot_from_baseline_merges_dirty_metadata_and_volatile_runs() -> (
    None
):
    """The runtime collector refreshes metadata, then merges the volatile slice."""
    collector = MlflowObservabilityCollector(_empty_client())
    baseline = _make_baseline()
    dirty_experiments = _ExperimentScanResult(
        experiments=(
            _ExperimentRef(
                experiment_id="exp-old",
                last_update_time=baseline.horizon_ms + 1,
                lifecycle_stage="deleted",
            ),
            _ExperimentRef(
                experiment_id="exp-new",
                last_update_time=baseline.horizon_ms + 2,
                lifecycle_stage="active",
            ),
        )
    )

    with (
        patch.object(
            collector._queries,
            "scan_dirty_experiments",
            return_value=dirty_experiments,
        ),
        patch.object(
            collector._queries,
            "scan_volatile_runs_by_experiment",
            return_value=_RunCountsByExperimentScanResult(
                counts_by_experiment={
                    "exp-old": {
                        "RUNNING": 1,
                        "FINISHED": 0,
                        "FAILED": 0,
                        "KILLED": 0,
                    },
                    "exp-new": {
                        "RUNNING": 2,
                        "FINISHED": 0,
                        "FAILED": 0,
                        "KILLED": 0,
                    },
                }
            ),
        ) as mock_scan_volatile,
    ):
        snapshot = collector._build_snapshot_from_baseline(baseline)

    assert tuple(mock_scan_volatile.call_args.args[0]) == (
        "exp-old",
        "exp-new",
    )
    assert mock_scan_volatile.call_args.args[1] == baseline.horizon_ms
    assert snapshot.experiments_total == 2
    assert snapshot.experiments_active_total == 1
    assert snapshot.experiments_deleted_total == 1
    assert snapshot.runs_by_status["RUNNING"] == 3
    assert snapshot.runs_by_status["FINISHED"] == 5
    assert snapshot.registered_models_total == 2
    assert snapshot.model_versions_total == 7


def test_published_snapshot_is_immutable() -> None:
    """Published snapshots reject accidental mutation of nested counts."""
    snapshot = _make_snapshot(total=1)
    mutable_view = cast(dict[str, int], snapshot.runs_by_status)

    with pytest.raises(TypeError):
        mutable_view["RUNNING"] = 99


def test_backoff_interval_returns_base_on_zero_failures() -> None:
    """No failures means the original interval is used."""
    assert _backoff_interval(30, 0) == 30


def test_backoff_interval_doubles_on_each_failure() -> None:
    """Consecutive failures successively double the wait interval."""
    assert _backoff_interval(30, 1) == 60
    assert _backoff_interval(30, 2) == 120
