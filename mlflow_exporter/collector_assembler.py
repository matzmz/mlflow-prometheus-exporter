"""Helper class that assembles collector state from MLflow scan results."""

import time
from collections.abc import Mapping
from typing import Optional

from mlflow_exporter.collector_state import (
    _Baseline,
    _ExperimentBaseline,
    _ExperimentRef,
    _ExperimentScanResult,
    _ModelVersionScanResult,
    _RunCountsByExperimentScanResult,
)
from mlflow_exporter.models import MlflowSnapshot
from mlflow_exporter.settings import RUN_STATUSES


class CollectorAssembler:
    """Transform raw scan results into collector state objects."""

    @staticmethod
    def normalise_run_counts(
        counts: Optional[Mapping[str, int]] = None,
    ) -> dict[str, int]:
        """Return counts initialised for every known run status."""
        normalised = {status: 0 for status in RUN_STATUSES}
        if counts is None:
            return normalised
        for status, count in counts.items():
            if status in normalised:
                normalised[status] = count
        return normalised

    @staticmethod
    def increment_stage_count(
        counts: dict[str, int], lifecycle_stage: str
    ) -> None:
        """Increment the aggregate count for one lifecycle stage."""
        counts[lifecycle_stage] = counts.get(lifecycle_stage, 0) + 1

    @classmethod
    def add_run_counts(
        cls,
        target: dict[str, int],
        counts: Mapping[str, int],
    ) -> None:
        """Add one experiment contribution into aggregate run counts."""
        for status, count in cls.normalise_run_counts(counts).items():
            target[status] = target.get(status, 0) + count

    @classmethod
    def run_counts_for(
        cls,
        result: _RunCountsByExperimentScanResult,
        experiment_id: str,
    ) -> Mapping[str, int]:
        """Return counts for one experiment, defaulting to zeroes."""
        return result.counts_by_experiment.get(
            experiment_id,
            cls.normalise_run_counts(),
        )

    @classmethod
    def merge_run_count_results(
        cls,
        left: _RunCountsByExperimentScanResult,
        right: _RunCountsByExperimentScanResult,
    ) -> _RunCountsByExperimentScanResult:
        """Return a new scan result containing the sum of two scan results."""
        merged_counts = {
            experiment_id: cls.normalise_run_counts(counts)
            for experiment_id, counts in left.counts_by_experiment.items()
        }
        for experiment_id, counts in right.counts_by_experiment.items():
            target = merged_counts.setdefault(
                experiment_id,
                cls.normalise_run_counts(),
            )
            cls.add_run_counts(target, counts)
        return _RunCountsByExperimentScanResult(
            counts_by_experiment=merged_counts
        )

    @classmethod
    def build_experiment_baselines(
        cls,
        experiments: tuple[_ExperimentRef, ...],
        stable_runs_by_experiment: _RunCountsByExperimentScanResult,
    ) -> tuple[_ExperimentBaseline, ...]:
        """Build baseline entries for every known experiment."""
        return tuple(
            _ExperimentBaseline(
                experiment_id=experiment.experiment_id,
                last_update_time=experiment.last_update_time,
                lifecycle_stage=experiment.lifecycle_stage,
                stable_runs_by_status=cls.run_counts_for(
                    stable_runs_by_experiment,
                    experiment.experiment_id,
                ),
            )
            for experiment in experiments
        )

    @staticmethod
    def build_baseline(
        experiment_baselines: tuple[_ExperimentBaseline, ...],
        registered_models_total: int,
        model_versions: _ModelVersionScanResult,
        horizon_ms: int,
        built_at: float | None = None,
    ) -> _Baseline:
        """Build the immutable baseline object published by the collector."""
        return _Baseline(
            experiments_by_id={
                experiment.experiment_id: experiment
                for experiment in experiment_baselines
            },
            registered_models_total=registered_models_total,
            model_versions_total=model_versions.total,
            model_versions_by_stage=model_versions.by_stage,
            horizon_ms=horizon_ms,
            built_at=time.monotonic() if built_at is None else built_at,
        )

    @classmethod
    def current_experiments_from_baseline(
        cls,
        baseline: _Baseline,
        dirty_experiments: _ExperimentScanResult,
    ) -> dict[str, _ExperimentBaseline]:
        """Refresh experiment metadata while preserving stable run counts."""
        current_experiments = dict(baseline.experiments_by_id)
        for experiment in dirty_experiments.experiments:
            previous = current_experiments.get(experiment.experiment_id)
            stable_runs_by_status = (
                previous.stable_runs_by_status
                if previous is not None
                else cls.normalise_run_counts()
            )
            current_experiments[experiment.experiment_id] = (
                _ExperimentBaseline(
                    experiment_id=experiment.experiment_id,
                    last_update_time=experiment.last_update_time,
                    lifecycle_stage=experiment.lifecycle_stage,
                    stable_runs_by_status=stable_runs_by_status,
                )
            )
        return current_experiments

    @classmethod
    def build_snapshot(
        cls,
        baseline: _Baseline,
        current_experiments: Mapping[str, _ExperimentBaseline],
        volatile_runs_by_experiment: _RunCountsByExperimentScanResult,
    ) -> MlflowSnapshot:
        """Merge the stable baseline with the current volatile run slice."""
        experiments_by_stage: dict[str, int] = {}
        runs_by_status = cls.normalise_run_counts()
        for experiment in current_experiments.values():
            cls.increment_stage_count(
                experiments_by_stage,
                experiment.lifecycle_stage,
            )
            cls.add_run_counts(
                runs_by_status, experiment.stable_runs_by_status
            )
            cls.add_run_counts(
                runs_by_status,
                cls.run_counts_for(
                    volatile_runs_by_experiment,
                    experiment.experiment_id,
                ),
            )
        return MlflowSnapshot(
            experiments_total=len(current_experiments),
            experiments_active_total=experiments_by_stage.get("active", 0),
            experiments_deleted_total=experiments_by_stage.get("deleted", 0),
            runs_total=sum(runs_by_status.values()),
            runs_by_status=runs_by_status,
            registered_models_total=baseline.registered_models_total,
            model_versions_total=baseline.model_versions_total,
            model_versions_by_stage=baseline.model_versions_by_stage,
        )
