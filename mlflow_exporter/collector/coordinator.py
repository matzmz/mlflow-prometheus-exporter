"""Runtime coordination for baseline+delta MLflow collection.

The collector owns refresh timing, concurrency control, and publication of the
latest immutable snapshot. Query details and snapshot assembly live in
dedicated modules so this file can stay focused on lifecycle behavior.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Optional

from mlflow.tracking import MlflowClient

from mlflow_exporter.collector.assembler import CollectorAssembler
from mlflow_exporter.collector.queries import MlflowCollectorQueries
from mlflow_exporter.collector.state import (
    _Baseline,
    _PublishedState,
)
from mlflow_exporter.config.settings import (
    BASELINE_INTERVAL_SECONDS,
    HORIZON_DAYS,
)
from mlflow_exporter.models import MlflowSnapshot

LOGGER = logging.getLogger(__name__)
MAX_BACKOFF_SECONDS = 120
__all__ = [
    "MAX_BACKOFF_SECONDS",
    "MlflowObservabilityCollector",
    "_backoff_interval",
]


def _backoff_interval(base_interval: int, consecutive_failures: int) -> int:
    """Return the wait interval with exponential backoff on failures."""
    if consecutive_failures <= 0:
        return base_interval
    return min(base_interval * (2**consecutive_failures), MAX_BACKOFF_SECONDS)


class MlflowObservabilityCollector:
    """Collect aggregated observability data from an MLflow tracking server.

    This class intentionally stays small: it owns lock coordination, refresh
    loops, and the published state. The MLflow API access is delegated to
    ``MlflowCollectorQueries`` and the assembly of immutable state is delegated
    to ``CollectorAssembler``.
    """

    def __init__(
        self,
        client: MlflowClient,
        baseline_interval_seconds: int = BASELINE_INTERVAL_SECONDS,
        horizon_days: int = HORIZON_DAYS,
        queries: Optional[MlflowCollectorQueries] = None,
    ) -> None:
        """Configure the collector with an MLflow client and refresh policy."""
        self._queries = queries or MlflowCollectorQueries(client)
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
        """Build the immutable baseline used by all future delta refreshes."""
        horizon_ms = self._horizon_ms()
        experiments = self._queries.scan_all_experiments()
        stable_runs_by_experiment = (
            self._queries.scan_stable_runs_by_experiment(
                (
                    experiment.experiment_id
                    for experiment in experiments.experiments
                ),
                horizon_ms,
            )
        )
        experiment_baselines = CollectorAssembler.build_experiment_baselines(
            experiments.experiments,
            stable_runs_by_experiment,
        )
        model_versions = self._queries.scan_model_versions()
        registered_models_total = self._queries.count_registered_models()
        return CollectorAssembler.build_baseline(
            experiment_baselines,
            registered_models_total,
            model_versions,
            horizon_ms,
        )

    def _build_snapshot_from_baseline(
        self, baseline: _Baseline
    ) -> MlflowSnapshot:
        """Merge the baseline with the latest volatile MLflow data."""
        dirty_experiments = self._queries.scan_dirty_experiments(
            baseline.horizon_ms
        )
        current_experiments = (
            CollectorAssembler.current_experiments_from_baseline(
                baseline,
                dirty_experiments,
            )
        )
        volatile_runs_by_experiment = (
            self._queries.scan_volatile_runs_by_experiment(
                current_experiments.keys(),
                baseline.horizon_ms,
            )
        )
        return CollectorAssembler.build_snapshot(
            baseline,
            current_experiments,
            volatile_runs_by_experiment,
        )
