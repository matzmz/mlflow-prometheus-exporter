"""MLflow observability data collector with coordinated baseline+delta refreshes.

Polling strategy
----------------
The collector keeps two refresh responsibilities separate:

* **Baseline**: a complete rebuild of the stable state. It runs once
  during bootstrap and then periodically in a dedicated background thread.
* **Delta refresh**: a lightweight merge against the latest published
  baseline. It runs frequently and only re-scans experiments newer than the
  baseline horizon timestamp.

Only one refresh is allowed to talk to MLflow at a time. If a periodic
refresh collides with another long-running refresh, the exporter keeps
serving the last published snapshot instead of blocking scrapes.
"""

import logging
import threading
import time
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from mlflow_exporter.models import MlflowSnapshot
from mlflow_exporter.settings import (
    BASELINE_INTERVAL_SECONDS,
    EXPERIMENT_PAGE_SIZE,
    HORIZON_DAYS,
    MODEL_PAGE_SIZE,
    MODEL_STAGES,
    MODEL_VERSION_PAGE_SIZE,
    RUN_PAGE_SIZE,
    RUN_STATUSES,
)

LOGGER = logging.getLogger(__name__)
MAX_BACKOFF_SECONDS = 120


def _backoff_interval(base_interval: int, consecutive_failures: int) -> int:
    """Return the wait interval with exponential backoff on failures."""
    if consecutive_failures <= 0:
        return base_interval
    return min(base_interval * (2**consecutive_failures), MAX_BACKOFF_SECONDS)


def _normalise_run_counts(
    counts: Optional[Mapping[str, int]] = None
) -> dict[str, int]:
    """Return counts initialised for every known run status."""
    normalised = {status: 0 for status in RUN_STATUSES}
    if counts is None:
        return normalised
    for status, count in counts.items():
        if status in normalised:
            normalised[status] = count
    return normalised


def _increment_stage_count(
    counts: dict[str, int], lifecycle_stage: str
) -> None:
    """Increment the aggregate count for one lifecycle stage."""
    counts[lifecycle_stage] = counts.get(lifecycle_stage, 0) + 1


def _decrement_stage_count(
    counts: dict[str, int], lifecycle_stage: str
) -> None:
    """Decrement the aggregate count for one lifecycle stage."""
    if lifecycle_stage not in counts:
        return
    next_value = counts[lifecycle_stage] - 1
    if next_value <= 0:
        counts.pop(lifecycle_stage, None)
        return
    counts[lifecycle_stage] = next_value


def _add_run_counts(
    target: dict[str, int], counts: Mapping[str, int]
) -> None:
    """Add one experiment contribution into aggregate run counts."""
    for status, count in _normalise_run_counts(counts).items():
        target[status] = target.get(status, 0) + count


def _subtract_run_counts(
    target: dict[str, int], counts: Mapping[str, int]
) -> None:
    """Remove one experiment contribution from aggregate run counts."""
    for status, count in _normalise_run_counts(counts).items():
        target[status] = target.get(status, 0) - count


@dataclass(frozen=True)
class _ExperimentRef:
    """Experiment metadata used to drive per-experiment recounts."""

    experiment_id: str
    last_update_time: int
    lifecycle_stage: str


@dataclass(frozen=True)
class _ExperimentContribution:
    """Immutable exported contribution of a single experiment."""

    experiment_id: str
    last_update_time: int
    lifecycle_stage: str
    runs_by_status: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze nested counts to protect published state from mutation."""
        object.__setattr__(
            self,
            "runs_by_status",
            MappingProxyType(_normalise_run_counts(self.runs_by_status)),
        )


@dataclass(frozen=True)
class _Baseline:
    """Immutable baseline snapshot published by the collector."""

    experiments_total: int
    experiments_by_stage: Mapping[str, int]
    runs_by_status: Mapping[str, int]
    experiments_by_id: Mapping[str, _ExperimentContribution]
    registered_models_total: int
    model_versions_total: int
    model_versions_by_stage: Mapping[str, int]
    horizon_ms: int
    built_at: float

    def __post_init__(self) -> None:
        """Freeze nested mappings so the baseline can be shared safely."""
        object.__setattr__(
            self,
            "experiments_by_stage",
            MappingProxyType(dict(self.experiments_by_stage)),
        )
        object.__setattr__(
            self,
            "runs_by_status",
            MappingProxyType(_normalise_run_counts(self.runs_by_status)),
        )
        object.__setattr__(
            self,
            "experiments_by_id",
            MappingProxyType(dict(self.experiments_by_id)),
        )
        object.__setattr__(
            self,
            "model_versions_by_stage",
            MappingProxyType(dict(self.model_versions_by_stage)),
        )


@dataclass(frozen=True)
class _ExperimentScanResult:
    """Result of scanning experiments against a time horizon."""

    experiments: tuple[_ExperimentRef, ...]


@dataclass(frozen=True)
class _ModelVersionScanResult:
    """Result of scanning all model versions."""

    total: int
    by_stage: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze aggregated stage counts after construction."""
        object.__setattr__(
            self,
            "by_stage",
            MappingProxyType(dict(self.by_stage)),
        )


@dataclass(frozen=True)
class _PublishedState:
    """Atomically published exporter state."""

    baseline: _Baseline
    snapshot: MlflowSnapshot


class MlflowObservabilityCollector:
    """Collect aggregated observability data from an MLflow tracking server."""

    def __init__(
        self,
        client: MlflowClient,
        baseline_interval_seconds: int = BASELINE_INTERVAL_SECONDS,
        horizon_days: int = HORIZON_DAYS,
    ) -> None:
        """Configure the collector with an MLflow client and refresh policy.

        Parameters:
        client (MlflowClient): Authenticated MLflow tracking client.
        baseline_interval_seconds (int): Seconds between background
            baseline rebuilds.
        horizon_days (int): Age in days above which data is considered
            stable and stored in the published baseline.
        """
        self._client = client
        self._baseline_interval_seconds = baseline_interval_seconds
        self._horizon_days = horizon_days
        self._refresh_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state: Optional[_PublishedState] = None
        self._stop_event = threading.Event()
        self._worker_lock = threading.Lock()
        self._baseline_thread: Optional[threading.Thread] = None

    def initialize(self) -> MlflowSnapshot:
        """Build and publish the first baseline before serving traffic."""
        snapshot = self._run_baseline_cycle(blocking=True)
        if snapshot is None:
            raise RuntimeError("Bootstrap failed to produce a snapshot")
        return snapshot

    def start_baseline_worker(self) -> None:
        """Start the background baseline worker exactly once."""
        self.start_baseline_worker_with_callbacks(
            on_snapshot=lambda _snapshot, _duration_seconds: None,
            on_failure=lambda _duration_seconds: None,
        )

    def start_baseline_worker_with_callbacks(
        self,
        on_snapshot: Callable[[MlflowSnapshot, float], None],
        on_failure: Callable[[float], None],
    ) -> None:
        """Start the background baseline worker exactly once."""
        with self._worker_lock:
            if self._baseline_thread is not None:
                return
            thread = threading.Thread(
                target=self._run_baseline_loop,
                kwargs={
                    "on_snapshot": on_snapshot,
                    "on_failure": on_failure,
                },
                name="baseline",
                daemon=True,
            )
            thread.start()
            self._baseline_thread = thread

    def stop(self) -> None:
        """Request background workers to stop on the next wait boundary."""
        LOGGER.info("Collector stop requested")
        self._stop_event.set()

    def current_snapshot(self) -> MlflowSnapshot:
        """Return the last published snapshot.

        Raises:
        RuntimeError: If the collector has not completed bootstrap yet.
        """
        with self._state_lock:
            if self._state is None:
                raise RuntimeError("Collector is not initialized")
            return self._state.snapshot

    def refresh_delta_snapshot(self) -> MlflowSnapshot:
        """Refresh the delta view, or return stale data during contention."""
        published = self._get_published_state()
        if published is None:
            raise RuntimeError("Collector is not initialized")
        if not self._refresh_lock.acquire(blocking=False):
            LOGGER.debug("Delta refresh skipped: lock busy")
            return published.snapshot
        try:
            published = self._get_published_state()
            if published is None:
                raise RuntimeError("Collector is not initialized")
            snapshot = self._build_snapshot_from_baseline(published.baseline)
            self._publish_state(published.baseline, snapshot)
            return snapshot
        finally:
            self._refresh_lock.release()

    def run_delta_refresh_loop(
        self,
        poll_interval_seconds: int,
        on_snapshot: Callable[[MlflowSnapshot, float], None],
        on_failure: Callable[[float], None],
    ) -> None:
        """Refresh delta state forever and report outcomes through callbacks."""
        consecutive_failures = 0
        while not self._wait_for_next_delta_cycle(
            _backoff_interval(poll_interval_seconds, consecutive_failures)
        ):
            started = time.monotonic()
            try:
                snapshot = self.refresh_delta_snapshot()
                on_snapshot(snapshot, time.monotonic() - started)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                on_failure(time.monotonic() - started)
                LOGGER.exception("MLflow delta refresh failed")

    def _run_baseline_loop(
        self,
        on_snapshot: Callable[[MlflowSnapshot, float], None],
        on_failure: Callable[[float], None],
    ) -> None:
        """Periodically rebuild and publish the baseline."""
        consecutive_failures = 0
        while not self._wait_for_next_baseline_cycle(
            _backoff_interval(
                self._baseline_interval_seconds, consecutive_failures
            )
        ):
            started = time.monotonic()
            try:
                snapshot = self._run_baseline_cycle(blocking=False)
                if snapshot is not None:
                    on_snapshot(snapshot, time.monotonic() - started)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                on_failure(time.monotonic() - started)
                LOGGER.exception("Background baseline refresh failed")

    def _wait_for_next_baseline_cycle(self, interval_seconds: int) -> bool:
        """Wait for the next baseline cycle or a stop request."""
        return self._stop_event.wait(interval_seconds)

    def _wait_for_next_delta_cycle(self, interval_seconds: int) -> bool:
        """Wait for the next delta cycle or a stop request."""
        return self._stop_event.wait(interval_seconds)

    def _run_baseline_cycle(self, blocking: bool) -> Optional[MlflowSnapshot]:
        """Build a baseline and publish a matching merged snapshot."""
        if not self._refresh_lock.acquire(blocking=blocking):
            return None
        try:
            baseline = self._build_baseline()
            snapshot = self._build_snapshot_from_baseline(baseline)
            self._publish_state(baseline, snapshot)
            return snapshot
        finally:
            self._refresh_lock.release()

    def _get_published_state(self) -> Optional[_PublishedState]:
        """Return the latest published state under the state lock."""
        with self._state_lock:
            return self._state

    def _publish_state(
        self, baseline: _Baseline, snapshot: MlflowSnapshot
    ) -> None:
        """Atomically publish a baseline/snapshot pair."""
        with self._state_lock:
            self._state = _PublishedState(baseline=baseline, snapshot=snapshot)

    def _horizon_ms(self) -> int:
        """Return the Unix timestamp in ms below which data is treated as stable.

        Returns:
        int: Current time minus horizon_days, expressed in milliseconds.
        """
        return int((time.time() - self._horizon_days * 86_400) * 1_000)

    def _build_baseline(self) -> _Baseline:
        """Run a full scan and build the stable cache for data older than horizon.

        Returns:
        _Baseline: Immutable snapshot covering all data up to the current
            horizon.
        """
        horizon_ms = self._horizon_ms()
        stable_experiments = self._scan_stable_experiments(horizon_ms)
        stable_contributions = self._build_experiment_contributions(
            stable_experiments.experiments
        )
        experiments_total, experiments_by_stage, runs_by_status = (
            self._aggregate_experiment_contributions(stable_contributions)
        )
        model_versions = self._scan_model_versions()
        rm_total = self._count_paginated(
            self._client.search_registered_models,
            max_results=MODEL_PAGE_SIZE,
        )
        return _Baseline(
            experiments_total=experiments_total,
            experiments_by_stage=experiments_by_stage,
            runs_by_status=runs_by_status,
            experiments_by_id={
                contribution.experiment_id: contribution
                for contribution in stable_contributions
            },
            registered_models_total=rm_total,
            model_versions_total=model_versions.total,
            model_versions_by_stage=model_versions.by_stage,
            horizon_ms=horizon_ms,
            built_at=time.monotonic(),
        )

    def _build_snapshot_from_baseline(
        self, baseline: _Baseline
    ) -> MlflowSnapshot:
        """Merge the published baseline with a lightweight fresh-data scan.

        Returns:
        MlflowSnapshot: Combined snapshot of stable and recently updated
            experiments.
        """
        dirty_experiments = self._scan_dirty_experiments(baseline.horizon_ms)
        dirty_contributions = self._build_experiment_contributions(
            dirty_experiments.experiments
        )
        experiments_total = baseline.experiments_total
        experiments_by_stage = dict(baseline.experiments_by_stage)
        runs_by_status = _normalise_run_counts(baseline.runs_by_status)
        for contribution in dirty_contributions:
            previous = baseline.experiments_by_id.get(contribution.experiment_id)
            if previous is None:
                experiments_total += 1
            else:
                _decrement_stage_count(
                    experiments_by_stage, previous.lifecycle_stage
                )
                _subtract_run_counts(runs_by_status, previous.runs_by_status)
            _increment_stage_count(
                experiments_by_stage, contribution.lifecycle_stage
            )
            _add_run_counts(runs_by_status, contribution.runs_by_status)
        return MlflowSnapshot(
            experiments_total=experiments_total,
            experiments_active_total=experiments_by_stage.get("active", 0),
            experiments_deleted_total=experiments_by_stage.get(
                "deleted", 0
            ),
            runs_total=sum(runs_by_status.values()),
            runs_by_status=runs_by_status,
            registered_models_total=baseline.registered_models_total,
            model_versions_total=baseline.model_versions_total,
            model_versions_by_stage=baseline.model_versions_by_stage,
        )

    def _scan_stable_experiments(
        self, horizon_ms: int
    ) -> _ExperimentScanResult:
        """Scan all experiments and return those stable at the horizon.

        Classifies each experiment in Python by comparing its last update time
        against horizon_ms, avoiding an extra server-side filter for the stable
        side of the baseline.

        Parameters:
        horizon_ms (int): Unix timestamp in ms used as the stable-data cutoff.
        """
        experiments: list[_ExperimentRef] = []
        page_token = None
        while True:
            page = self._client.search_experiments(
                view_type=ViewType.ALL,
                max_results=EXPERIMENT_PAGE_SIZE,
                page_token=page_token,
            )
            for exp in page:
                update_time = exp.last_update_time or 0
                if update_time <= horizon_ms:
                    experiments.append(
                        _ExperimentRef(
                            experiment_id=exp.experiment_id,
                            last_update_time=update_time,
                            lifecycle_stage=exp.lifecycle_stage or "active",
                        )
                    )
            page_token = getattr(page, "token", None)
            if not page_token:
                return _ExperimentScanResult(experiments=tuple(experiments))

    def _scan_dirty_experiments(
        self, horizon_ms: int
    ) -> _ExperimentScanResult:
        """Fetch experiments updated after horizon_ms for a full recount.

        Parameters:
        horizon_ms (int): Unix timestamp in ms; only newer experiments are
            fetched.
        """
        experiments: list[_ExperimentRef] = []
        page_token = None
        while True:
            page = self._client.search_experiments(
                view_type=ViewType.ALL,
                max_results=EXPERIMENT_PAGE_SIZE,
                filter_string=f"last_update_time > {horizon_ms}",
                page_token=page_token,
            )
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

    def _build_experiment_contributions(
        self, experiments: tuple[_ExperimentRef, ...]
    ) -> tuple[_ExperimentContribution, ...]:
        """Build immutable exported contributions for the given experiments."""
        run_counts_by_experiment = self._scan_runs_by_experiment(
            experiment.experiment_id for experiment in experiments
        )
        return tuple(
            _ExperimentContribution(
                experiment_id=experiment.experiment_id,
                last_update_time=experiment.last_update_time,
                lifecycle_stage=experiment.lifecycle_stage,
                runs_by_status=run_counts_by_experiment.get(
                    experiment.experiment_id,
                    _normalise_run_counts(),
                ),
            )
            for experiment in experiments
        )

    def _scan_runs_by_experiment(
        self, experiment_ids: Collection[str] | Iterable[str]
    ) -> dict[str, dict[str, int]]:
        """Return run counts grouped by experiment and then by status.

        Parameters:
        experiment_ids (Collection[str] | Iterable[str]): Experiment IDs to
            scope the run search.

        Returns:
        dict: Per-experiment run counts keyed first by experiment id and then
            by run status.
        """
        experiment_ids = tuple(dict.fromkeys(experiment_ids))
        if not experiment_ids:
            return {}
        by_experiment: dict[str, dict[str, int]] = {
            experiment_id: _normalise_run_counts()
            for experiment_id in experiment_ids
        }
        page_token = None
        while True:
            page = self._client.search_runs(
                experiment_ids=sorted(experiment_ids),
                filter_string="",
                run_view_type=ViewType.ALL,
                max_results=RUN_PAGE_SIZE,
                page_token=page_token,
            )
            for run in page:
                experiment_id = run.info.experiment_id
                if experiment_id not in by_experiment:
                    by_experiment[experiment_id] = _normalise_run_counts()
                status = run.info.status
                if status in RUN_STATUSES:
                    by_experiment[experiment_id][status] += 1
            page_token = getattr(page, "token", None)
            if not page_token:
                return by_experiment

    def _scan_model_versions(self) -> _ModelVersionScanResult:
        """Count all model versions and aggregate by stage."""
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

    @staticmethod
    def _aggregate_experiment_contributions(
        contributions: Iterable[_ExperimentContribution],
    ) -> tuple[int, dict[str, int], dict[str, int]]:
        """Aggregate per-experiment contributions into snapshot totals."""
        experiments_total = 0
        experiments_by_stage: dict[str, int] = {}
        runs_by_status = _normalise_run_counts()
        for contribution in contributions:
            experiments_total += 1
            _increment_stage_count(
                experiments_by_stage, contribution.lifecycle_stage
            )
            _add_run_counts(runs_by_status, contribution.runs_by_status)
        return experiments_total, experiments_by_stage, runs_by_status

    @staticmethod
    def _count_paginated(
        search_function: Callable[..., Sequence[Any]], **kwargs: Any
    ) -> int:
        """Count all entries returned by a paginated MLflow search function.

        Parameters:
        search_function (Callable): An MLflow search function that accepts a
            ``page_token`` keyword argument and returns a paged list with a
            ``.token`` attribute indicating the next page.
        **kwargs: Additional keyword arguments forwarded to ``search_function``
            on every page request (e.g. ``max_results``).

        Returns:
        int: Total number of entries across all pages.
        """
        total = 0
        page_token = None
        while True:
            page = search_function(page_token=page_token, **kwargs)
            total += len(page)
            page_token = getattr(page, "token", None)
            if not page_token:
                return total
