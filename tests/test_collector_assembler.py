"""Unit tests for collector assembly helpers."""

from collections.abc import Mapping

from mlflow_exporter.collector_assembler import CollectorAssembler
from mlflow_exporter.collector_state import (
    _Baseline,
    _ExperimentBaseline,
    _ExperimentRef,
    _ExperimentScanResult,
    _ModelVersionScanResult,
    _RunCountsByExperimentScanResult,
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
    """Return a reusable baseline for snapshot-assembly tests."""
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
        built_at=123.0,
    )


def test_build_experiment_baselines_captures_all_experiments() -> None:
    """Every experiment gets a baseline entry, even without stable runs yet."""
    experiments = (
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

    baselines = CollectorAssembler.build_experiment_baselines(
        experiments,
        stable_runs_by_experiment=_RunCountsByExperimentScanResult(
            counts_by_experiment={
                "exp-old": {
                    "RUNNING": 0,
                    "FINISHED": 5,
                    "FAILED": 1,
                    "KILLED": 0,
                }
            }
        ),
    )

    assert baselines[0].experiment_id == "exp-old"
    assert baselines[0].stable_runs_by_status["FINISHED"] == 5
    assert baselines[1].experiment_id == "exp-new"
    assert baselines[1].lifecycle_stage == "deleted"
    assert all(
        value == 0 for value in baselines[1].stable_runs_by_status.values()
    )


def test_build_baseline_records_model_registry_and_horizon() -> None:
    """Baseline assembly preserves registry counts, horizon, and build time."""
    experiment_baselines = (
        _make_experiment_baseline(
            experiment_id="exp-old",
            stable_runs_by_status={"FINISHED": 5},
        ),
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

    baseline = CollectorAssembler.build_baseline(
        experiment_baselines=experiment_baselines,
        registered_models_total=2,
        model_versions=model_versions,
        horizon_ms=1234,
        built_at=99.0,
    )

    assert baseline.horizon_ms == 1234
    assert baseline.built_at == 99.0
    assert baseline.registered_models_total == 2
    assert baseline.model_versions_total == 4
    assert (
        baseline.experiments_by_id["exp-old"].stable_runs_by_status["FINISHED"]
        == 5
    )


def test_current_experiments_from_baseline_refreshes_metadata_only() -> None:
    """Dirty experiments keep stable counts while refreshing metadata."""
    baseline = _make_baseline()

    current = CollectorAssembler.current_experiments_from_baseline(
        baseline,
        _ExperimentScanResult(
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
        ),
    )

    assert current["exp-old"].lifecycle_stage == "deleted"
    assert current["exp-old"].stable_runs_by_status["FINISHED"] == 5
    assert all(
        value == 0
        for value in current["exp-new"].stable_runs_by_status.values()
    )


def test_build_snapshot_merges_baseline_and_delta() -> None:
    """Merged snapshots combine stable baseline counts with volatile runs."""
    snapshot = CollectorAssembler.build_snapshot(
        baseline=_make_baseline(),
        current_experiments={
            "exp-old": _make_experiment_baseline(
                experiment_id="exp-old",
                lifecycle_stage="active",
                stable_runs_by_status={"FINISHED": 5},
            ),
            "exp-new": _make_experiment_baseline(
                experiment_id="exp-new",
                lifecycle_stage="active",
                stable_runs_by_status={},
            ),
        },
        volatile_runs_by_experiment=_RunCountsByExperimentScanResult(
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
    )

    assert snapshot.experiments_total == 2
    assert snapshot.experiments_active_total == 2
    assert snapshot.experiments_deleted_total == 0
    assert snapshot.runs_by_status["RUNNING"] == 3
    assert snapshot.runs_by_status["FINISHED"] == 5
    assert snapshot.registered_models_total == 2
    assert snapshot.model_versions_total == 7


def test_build_snapshot_updates_experiment_stage_metadata() -> None:
    """Stage totals reflect refreshed experiment metadata even without new runs."""
    snapshot = CollectorAssembler.build_snapshot(
        baseline=_make_baseline(),
        current_experiments={
            "exp-old": _make_experiment_baseline(
                experiment_id="exp-old",
                lifecycle_stage="deleted",
                stable_runs_by_status={"FINISHED": 5},
            )
        },
        volatile_runs_by_experiment=_RunCountsByExperimentScanResult(
            counts_by_experiment={}
        ),
    )

    assert snapshot.experiments_total == 1
    assert snapshot.experiments_active_total == 0
    assert snapshot.experiments_deleted_total == 1
    assert snapshot.runs_by_status["FINISHED"] == 5
