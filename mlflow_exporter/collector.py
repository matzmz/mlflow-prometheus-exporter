#!/usr/bin/env python3


"""MLflow observability data collector with time-based caching.

Polling strategy
----------------
The collector separates MLflow data into two zones:

* **Baseline** (old data, ``last_update_time <= horizon_ms``): rebuilt
  once every *cache_ttl_seconds* (default 1 h). These records are
  stable: finished runs do not change status, deleted experiments stay
  deleted.
* **Delta** (fresh data, ``last_update_time > horizon_ms``): re-scanned
  on every poll cycle (default 30 s). In a mature deployment this set
  is tiny compared to the full history.

Model-registry entities (registered models, model versions) have no
timestamp filter in the MLflow API, so they are always taken from the
baseline and refreshed only at cache-rebuild time.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from mlflow_exporter.settings import (
    CACHE_TTL_SECONDS,
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


@dataclass(frozen=True)
class _Baseline:
    """Immutable snapshot of MLflow data older than the scan horizon."""

    old_experiment_ids: list
    old_experiments_by_stage: dict
    old_runs_by_status: dict
    registered_models_total: int
    model_versions_total: int
    model_versions_by_stage: dict
    horizon_ms: int
    built_at: float


class MlflowObservabilityCollector:
    """Collect aggregated observability data from an MLflow tracking server."""

    def __init__(
        self,
        client: MlflowClient,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
        horizon_days: int = HORIZON_DAYS,
    ) -> None:
        """Configure the collector with an MLflow client and cache settings.

        Parameters:
        client (MlflowClient): Authenticated MLflow tracking client.
        cache_ttl_seconds (int): Seconds before the stable baseline is rebuilt
            in the background. Defaults to CACHE_TTL_SECONDS.
        horizon_days (int): Age in days above which data is considered stable
            and cached in the baseline. Defaults to HORIZON_DAYS.
        """
        self._client = client
        self._cache_ttl_seconds = cache_ttl_seconds
        self._horizon_days = horizon_days
        self._baseline: Optional[_Baseline] = None
        self._lock = threading.Lock()
        self._rebuild_in_progress = False

    def collect(self) -> MlflowSnapshot:
        """Return a snapshot, triggering a background baseline rebuild when stale.

        The first call blocks until a full baseline is available. Subsequent
        calls return immediately from the cached baseline; if the baseline is
        stale, a background thread rebuilds it so polls stay non-blocking.

        Returns:
        MlflowSnapshot: Aggregated observability data from the MLflow server.
        """
        if self._baseline is None:
            self._swap_baseline(self._build_baseline())
        elif self._is_stale() and not self._rebuild_in_progress:
            self._start_background_rebuild()
        return self._compute_snapshot()

    def _is_stale(self) -> bool:
        """Return True when the baseline has exceeded its time-to-live."""
        assert self._baseline is not None
        age = time.monotonic() - self._baseline.built_at
        return age > self._cache_ttl_seconds

    def _swap_baseline(self, new_baseline: _Baseline) -> None:
        """Atomically replace the baseline under the write lock."""
        with self._lock:
            self._baseline = new_baseline

    def _start_background_rebuild(self) -> None:
        """Fire a daemon thread to rebuild the baseline without blocking polls."""
        self._rebuild_in_progress = True
        thread = threading.Thread(target=self._rebuild_and_swap, daemon=True)
        thread.start()

    def _rebuild_and_swap(self) -> None:
        """Run a full baseline scan and atomically publish the result."""
        try:
            new_baseline = self._build_baseline()
            self._swap_baseline(new_baseline)
        except Exception:
            LOGGER.exception("Background baseline rebuild failed")
        finally:
            self._rebuild_in_progress = False

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
        old_exp_ids, old_exp_by_stage = self._scan_old_experiments(horizon_ms)
        old_runs = self._scan_runs(
            old_exp_ids,
            filter_string=f"attributes.start_time <= {horizon_ms}",
        )
        mv_total, mv_by_stage = self._scan_model_versions()
        rm_total = self._count_paginated(
            self._client.search_registered_models,
            max_results=MODEL_PAGE_SIZE,
        )
        return _Baseline(
            old_experiment_ids=old_exp_ids,
            old_experiments_by_stage=old_exp_by_stage,
            old_runs_by_status=old_runs,
            registered_models_total=rm_total,
            model_versions_total=mv_total,
            model_versions_by_stage=mv_by_stage,
            horizon_ms=horizon_ms,
            built_at=time.monotonic(),
        )

    def _compute_snapshot(self) -> MlflowSnapshot:
        """Merge the stable baseline with a lightweight fresh-data delta scan.

        Returns:
        MlflowSnapshot: Combined snapshot of old and recent MLflow data.
        """
        with self._lock:
            bl = self._baseline
        assert bl is not None
        fresh_exp_ids, fresh_exp_by_stage = self._scan_fresh_experiments(
            bl.horizon_ms
        )
        fresh_runs = self._scan_runs(
            bl.old_experiment_ids + fresh_exp_ids,
            filter_string=f"attributes.start_time > {bl.horizon_ms}",
        )
        return MlflowSnapshot(
            experiments_total=(
                sum(bl.old_experiments_by_stage.values())
                + sum(fresh_exp_by_stage.values())
            ),
            experiments_active_total=(
                bl.old_experiments_by_stage.get("active", 0)
                + fresh_exp_by_stage.get("active", 0)
            ),
            experiments_deleted_total=(
                bl.old_experiments_by_stage.get("deleted", 0)
                + fresh_exp_by_stage.get("deleted", 0)
            ),
            runs_total=(
                sum(bl.old_runs_by_status.values()) + sum(fresh_runs.values())
            ),
            runs_by_status={
                s: bl.old_runs_by_status.get(s, 0) + fresh_runs.get(s, 0)
                for s in RUN_STATUSES
            },
            registered_models_total=bl.registered_models_total,
            model_versions_total=bl.model_versions_total,
            model_versions_by_stage=bl.model_versions_by_stage,
        )

    def _scan_old_experiments(self, horizon_ms: int) -> tuple:
        """Scan all experiments; return old IDs and old stage counts.

        Classifies each experiment in Python by comparing its
        last_update_time against horizon_ms, avoiding an extra API call.

        Parameters:
        horizon_ms (int): Unix timestamp in ms used as the stable-data cutoff.

        Returns:
        tuple: A pair ``(old_ids, old_by_stage)`` where ``old_ids`` is a list
            of experiment IDs and ``old_by_stage`` maps lifecycle stage to count.
        """
        old_ids: list = []
        old_by_stage: dict = {}
        page_token = None
        while True:
            page = self._client.search_experiments(
                view_type=ViewType.ALL,
                max_results=EXPERIMENT_PAGE_SIZE,
                page_token=page_token,
            )
            for exp in page:
                if exp.last_update_time <= horizon_ms:
                    old_ids.append(exp.experiment_id)
                    stage = exp.lifecycle_stage or "active"
                    old_by_stage[stage] = old_by_stage.get(stage, 0) + 1
            page_token = getattr(page, "token", None)
            if not page_token:
                return old_ids, old_by_stage

    def _scan_fresh_experiments(self, horizon_ms: int) -> tuple:
        """Fetch only experiments updated after horizon_ms (fast delta scan).

        Parameters:
        horizon_ms (int): Unix timestamp in ms; only newer experiments are fetched.

        Returns:
        tuple: A pair ``(ids, by_stage)`` mirroring the shape returned by
            ``_scan_old_experiments`` but for recently updated experiments.
        """
        ids: list = []
        by_stage: dict = {}
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
                return ids, by_stage

    def _scan_runs(self, experiment_ids: list, filter_string: str) -> dict:
        """Paginate runs matching filter_string and return counts by status.

        Parameters:
        experiment_ids (list): Experiment IDs to scope the run search.
        filter_string (str): MLflow filter expression applied to the run query
            (e.g. ``"attributes.start_time > 1234567890000"``).

        Returns:
        dict: Mapping of each status in RUN_STATUSES to its run count.
        """
        if not experiment_ids:
            return {s: 0 for s in RUN_STATUSES}
        by_status: dict = {s: 0 for s in RUN_STATUSES}
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

    def _scan_model_versions(self) -> tuple:
        """Count all model versions and aggregate by stage.

        Returns:
        tuple: A pair ``(total, by_stage)`` where ``total`` is the aggregate
            count and ``by_stage`` maps each MODEL_STAGES entry to its count.
        """
        page_token = None
        total = 0
        by_stage = {stage: 0 for stage in MODEL_STAGES}
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
                return total, by_stage

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
