"""MLflow query adapter used by the collector.

The collector needs to coordinate refresh timing and concurrency. The actual
MLflow pagination and filter details live here so they can be tested in
isolation and changed without touching the runtime control flow.
"""

import logging
from collections.abc import Collection, Iterable
from typing import Any, Optional

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from mlflow_exporter.collector.assembler import CollectorAssembler
from mlflow_exporter.collector.state import (
    _ExperimentRef,
    _ExperimentScanResult,
    _ModelVersionScanResult,
    _RunCountsByExperimentScanResult,
)
from mlflow_exporter.config.settings import (
    EXPERIMENT_PAGE_SIZE,
    MODEL_PAGE_SIZE,
    MODEL_STAGES,
    MODEL_VERSION_PAGE_SIZE,
    RUN_PAGE_SIZE,
    RUN_STATUSES,
)

LOGGER = logging.getLogger(__name__)


def _stable_runs_filter_string(horizon_ms: int) -> str:
    """Return the filter for runs that belong in the stable baseline."""
    return (
        "attributes.status != 'RUNNING' "
        f"AND attributes.end_time <= {horizon_ms}"
    )


def _recent_terminal_runs_filter_string(horizon_ms: int) -> str:
    """Return the filter for terminal runs that still belong to the delta."""
    return (
        "attributes.status != 'RUNNING' "
        f"AND attributes.end_time > {horizon_ms}"
    )


def _running_runs_filter_string() -> str:
    """Return the filter for runs whose status is still mutable."""
    return "attributes.status = 'RUNNING'"


class MlflowCollectorQueries:
    """Read MLflow state using exporter-specific pagination and filters."""

    def __init__(self, client: MlflowClient) -> None:
        """Store the MLflow client used by every collector query."""
        self._client = client

    def scan_all_experiments(self) -> _ExperimentScanResult:
        """Return the full set of experiments visible to the exporter."""
        return self._scan_experiments(filter_string=None)

    def scan_dirty_experiments(self, horizon_ms: int) -> _ExperimentScanResult:
        """Return experiments updated after the current baseline horizon."""
        return self._scan_experiments(
            filter_string=f"last_update_time > {horizon_ms}"
        )

    def scan_stable_runs_by_experiment(
        self,
        experiment_ids: Collection[str] | Iterable[str],
        horizon_ms: int,
    ) -> _RunCountsByExperimentScanResult:
        """Return stable run counts for the given experiments."""
        return self._scan_runs_by_experiment(
            experiment_ids,
            filter_string=_stable_runs_filter_string(horizon_ms),
        )

    def scan_volatile_runs_by_experiment(
        self,
        experiment_ids: Collection[str] | Iterable[str],
        horizon_ms: int,
    ) -> _RunCountsByExperimentScanResult:
        """Return the run slice rebuilt on every delta refresh.

        The volatile slice is the union of:

        * all currently ``RUNNING`` runs
        * all terminal runs whose ``end_time`` is newer than the horizon
        """
        running_runs_by_experiment = self._scan_runs_by_experiment(
            experiment_ids,
            filter_string=_running_runs_filter_string(),
        )
        recent_terminal_runs_by_experiment = self._scan_runs_by_experiment(
            experiment_ids,
            filter_string=_recent_terminal_runs_filter_string(horizon_ms),
        )
        return CollectorAssembler.merge_run_count_results(
            running_runs_by_experiment,
            recent_terminal_runs_by_experiment,
        )

    def scan_model_versions(self) -> _ModelVersionScanResult:
        """Count all model versions and aggregate them by stage."""
        page_token = None
        total = 0
        by_stage: dict[str, int] = {stage: 0 for stage in MODEL_STAGES}
        while True:
            page = self._client.search_model_versions(
                max_results=MODEL_VERSION_PAGE_SIZE,
                page_token=page_token,
            )
            total += len(page)
            for version in page:
                stage = version.current_stage or "None"
                by_stage[stage] = by_stage.get(stage, 0) + 1
            page_token = getattr(page, "token", None)
            if not page_token:
                return _ModelVersionScanResult(total=total, by_stage=by_stage)

    def count_registered_models(self) -> int:
        """Count all registered models exposed by MLflow."""
        return self._count_paginated(
            self._client.search_registered_models,
            max_results=MODEL_PAGE_SIZE,
        )

    def _scan_experiments(
        self, filter_string: Optional[str]
    ) -> _ExperimentScanResult:
        """Scan experiments with an optional server-side filter."""
        experiments: list[_ExperimentRef] = []
        page_token = None
        while True:
            page_kwargs: dict[str, Any] = {
                "view_type": ViewType.ALL,
                "max_results": EXPERIMENT_PAGE_SIZE,
                "page_token": page_token,
            }
            if filter_string is not None:
                page_kwargs["filter_string"] = filter_string
            page = self._client.search_experiments(**page_kwargs)
            for exp in page:
                experiments.append(
                    _ExperimentRef(
                        experiment_id=exp.experiment_id,
                        last_update_time=exp.last_update_time or 0,
                        lifecycle_stage=exp.lifecycle_stage or "active",
                    )
                )
            page_token = getattr(page, "token", None)
            if not page_token:
                return _ExperimentScanResult(experiments=tuple(experiments))

    def _scan_runs_by_experiment(
        self,
        experiment_ids: Collection[str] | Iterable[str],
        filter_string: str = "",
    ) -> _RunCountsByExperimentScanResult:
        """Return run counts grouped by experiment and then by status."""
        experiment_ids = tuple(dict.fromkeys(experiment_ids))
        if not experiment_ids:
            return _RunCountsByExperimentScanResult(counts_by_experiment={})
        counts_by_experiment: dict[str, dict[str, int]] = {
            experiment_id: CollectorAssembler.normalise_run_counts()
            for experiment_id in experiment_ids
        }
        page_token = None
        while True:
            page = self._client.search_runs(
                experiment_ids=sorted(experiment_ids),
                filter_string=filter_string,
                run_view_type=ViewType.ALL,
                max_results=RUN_PAGE_SIZE,
                page_token=page_token,
            )
            for run in page:
                experiment_id = run.info.experiment_id
                if experiment_id not in counts_by_experiment:
                    counts_by_experiment[experiment_id] = (
                        CollectorAssembler.normalise_run_counts()
                    )
                status = run.info.status
                if status in RUN_STATUSES:
                    counts_by_experiment[experiment_id][status] += 1
                else:
                    LOGGER.warning(
                        "Ignoring unknown run status %r in experiment %s",
                        status,
                        experiment_id,
                    )
            page_token = getattr(page, "token", None)
            if not page_token:
                return _RunCountsByExperimentScanResult(
                    counts_by_experiment=counts_by_experiment
                )

    @staticmethod
    def _count_paginated(search_function: Any, **kwargs: Any) -> int:
        """Count all entries returned by a paginated MLflow search function."""
        total = 0
        page_token = None
        while True:
            page = search_function(page_token=page_token, **kwargs)
            total += len(page)
            page_token = getattr(page, "token", None)
            if not page_token:
                return total
