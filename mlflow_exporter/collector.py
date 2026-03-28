"""MLflow observability data collector with coordinated baseline+delta refreshes.

Polling strategy
----------------
The collector keeps two refresh responsibilities separate:

* **Baseline**: a complete rebuild of the stable state. It runs once
  during bootstrap and then periodically in a dedicated background thread.
* **Delta refresh**: a lightweight merge against the latest published
  baseline. It runs frequently and only re-scans data newer than the
  baseline horizon timestamp.

Only one refresh is allowed to talk to MLflow at a time. If a periodic
refresh collides with another long-running refresh, the exporter keeps
serving the last published snapshot instead of blocking scrapes.
"""

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from mlflow_exporter.settings import (
    BASELINE_INTERVAL_SECONDS,
    EXPERIMENT_PAGE_SIZE,
    HORIZON_DAYS,
    MODEL_PAGE_SIZE,
    MODEL_STAGES,
    MODEL_VERSION_PAGE_SIZE,
    RUN_PAGE_SIZE,
    RUN_STATUSES,
    MlflowSnapshot,
)

LOGGER = logging.getLogger(__name__)
MAX_BACKOFF_SECONDS = 120


def _backoff_interval(base_interval: int, consecutive_failures: int) -> int:
    """Return the wait interval with exponential backoff on failures."""
    if consecutive_failures <= 0:
        return base_interval
    return min(base_interval * (2**consecutive_failures), MAX_BACKOFF_SECONDS)


@dataclass(frozen=True)
class _Baseline:
    """Immutable snapshot of MLflow data older than the scan horizon."""

    old_experiment_ids: list[str]
    old_experiments_by_stage: dict[str, int]
    old_runs_by_status: dict[str, int]
    registered_models_total: int
    model_versions_total: int
    model_versions_by_stage: dict[str, int]
    horizon_ms: int
    built_at: float


@dataclass(frozen=True)
class _ExperimentScanResult:
    """Result of scanning experiments against a time horizon."""

    ids: list[str]
    by_stage: dict[str, int]


@dataclass(frozen=True)
class _ModelVersionScanResult:
    """Result of scanning all model versions."""

    total: int
    by_stage: dict[str, int]


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
        _Baseline: Immutable snapshot covering all data up to the current horizon.
        """
        horizon_ms = self._horizon_ms()
        old_experiments = self._scan_old_experiments(horizon_ms)
        old_runs = self._scan_runs(
            old_experiments.ids,
            filter_string=f"attributes.start_time <= {horizon_ms}",
        )
        model_versions = self._scan_model_versions()
        rm_total = self._count_paginated(
            self._client.search_registered_models,
            max_results=MODEL_PAGE_SIZE,
        )
        return _Baseline(
            old_experiment_ids=old_experiments.ids,
            old_experiments_by_stage=old_experiments.by_stage,
            old_runs_by_status=old_runs,
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
        MlflowSnapshot: Combined snapshot of old and recent MLflow data.
        """
        fresh = self._scan_fresh_experiments(baseline.horizon_ms)
        fresh_runs = self._scan_runs(
            baseline.old_experiment_ids + fresh.ids,
            filter_string=f"attributes.start_time > {baseline.horizon_ms}",
        )
        return MlflowSnapshot(
            experiments_total=(
                sum(baseline.old_experiments_by_stage.values())
                + sum(fresh.by_stage.values())
            ),
            experiments_active_total=(
                baseline.old_experiments_by_stage.get("active", 0)
                + fresh.by_stage.get("active", 0)
            ),
            experiments_deleted_total=(
                baseline.old_experiments_by_stage.get("deleted", 0)
                + fresh.by_stage.get("deleted", 0)
            ),
            runs_total=(
                sum(baseline.old_runs_by_status.values())
                + sum(fresh_runs.values())
            ),
            runs_by_status={
                status: baseline.old_runs_by_status.get(status, 0)
                + fresh_runs.get(status, 0)
                for status in RUN_STATUSES
            },
            registered_models_total=baseline.registered_models_total,
            model_versions_total=baseline.model_versions_total,
            model_versions_by_stage=baseline.model_versions_by_stage,
        )

    def _scan_old_experiments(self, horizon_ms: int) -> _ExperimentScanResult:
        """Scan all experiments; return old IDs and old stage counts.

        Classifies each experiment in Python by comparing its
        last_update_time against horizon_ms, avoiding an extra API call.

        Parameters:
        horizon_ms (int): Unix timestamp in ms used as the stable-data cutoff.
        """
        ids: list[str] = []
        by_stage: dict[str, int] = {}
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
                    ids.append(exp.experiment_id)
                    stage = exp.lifecycle_stage or "active"
                    by_stage[stage] = by_stage.get(stage, 0) + 1
            page_token = getattr(page, "token", None)
            if not page_token:
                return _ExperimentScanResult(ids=ids, by_stage=by_stage)

    def _scan_fresh_experiments(
        self, horizon_ms: int
    ) -> _ExperimentScanResult:
        """Fetch only experiments updated after horizon_ms (fast delta scan).

        Parameters:
        horizon_ms (int): Unix timestamp in ms; only newer experiments are fetched.
        """
        ids: list[str] = []
        by_stage: dict[str, int] = {}
        page_token = None
        while True:
            page = self._client.search_experiments(
                view_type=ViewType.ALL,
                max_results=EXPERIMENT_PAGE_SIZE,
                filter_string=f"last_update_time > {horizon_ms}",
                page_token=page_token,
            )
            for exp in page:
                ids.append(exp.experiment_id)
                stage = exp.lifecycle_stage or "active"
                by_stage[stage] = by_stage.get(stage, 0) + 1
            page_token = getattr(page, "token", None)
            if not page_token:
                return _ExperimentScanResult(ids=ids, by_stage=by_stage)

    def _scan_runs(
        self, experiment_ids: list[str], filter_string: str
    ) -> dict[str, int]:
        """Paginate runs matching filter_string and return counts by status.

        Parameters:
        experiment_ids (list): Experiment IDs to scope the run search.
        filter_string (str): MLflow filter expression applied to the run query
            (e.g. ``"attributes.start_time > 1234567890000"``).

        Returns:
        dict: Mapping of each status in RUN_STATUSES to its run count.
        """
        if not experiment_ids:
            return {status: 0 for status in RUN_STATUSES}
        by_status: dict[str, int] = {status: 0 for status in RUN_STATUSES}
        page_token = None
        while True:
            page = self._client.search_runs(
                experiment_ids=list(experiment_ids),
                filter_string=filter_string,
                run_view_type=ViewType.ALL,
                max_results=RUN_PAGE_SIZE,
                page_token=page_token,
            )
            for run in page:
                status = run.info.status
                if status in by_status:
                    by_status[status] += 1
            page_token = getattr(page, "token", None)
            if not page_token:
                return by_status

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
