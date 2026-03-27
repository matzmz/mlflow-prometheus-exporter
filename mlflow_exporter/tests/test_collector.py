#!/usr/bin/env python3


"""Unit tests for MlflowObservabilityCollector."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlflow_exporter.collector import MlflowObservabilityCollector, _Baseline
from mlflow_exporter.settings import RUN_STATUSES, MlflowSnapshot

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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


def _make_run(status: str) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow Run object."""
    return SimpleNamespace(info=SimpleNamespace(status=status))


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


def _advance_monotonic(
    monkeypatch: pytest.MonkeyPatch, seconds: float
) -> None:
    """Replace time.monotonic so it appears to advance by *seconds*."""
    origin = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: origin + seconds)


# ---------------------------------------------------------------------------
# collect() dispatch logic
# ---------------------------------------------------------------------------


def test_collect_builds_baseline_on_first_call() -> None:
    """First collect() call blocks until a valid baseline is built."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=3600
    )

    snapshot = collector.collect()

    assert collector._baseline is not None
    assert isinstance(snapshot, MlflowSnapshot)


def test_collect_uses_cached_baseline_when_fresh() -> None:
    """Subsequent collect() calls reuse the cached baseline."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=3600
    )
    collector.collect()
    baseline_after_first = collector._baseline

    with patch.object(
        collector, "_build_baseline", wraps=collector._build_baseline
    ) as spy:
        collector.collect()

    spy.assert_not_called()
    assert collector._baseline is baseline_after_first


def test_collect_triggers_background_rebuild_when_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """collect() fires a background rebuild when the baseline is stale."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=1
    )
    collector.collect()
    _advance_monotonic(monkeypatch, seconds=10)

    with patch.object(collector, "_start_background_rebuild") as mock_rebuild:
        collector.collect()

    mock_rebuild.assert_called_once()


def test_collect_skips_rebuild_when_already_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """collect() does not spawn a duplicate rebuild when one is running."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=1
    )
    collector.collect()
    _advance_monotonic(monkeypatch, seconds=10)
    collector._rebuild_in_progress = True

    with patch.object(collector, "_start_background_rebuild") as mock_rebuild:
        collector.collect()

    mock_rebuild.assert_not_called()
    collector._rebuild_in_progress = False  # avoid leaking state


# ---------------------------------------------------------------------------
# _is_stale
# ---------------------------------------------------------------------------


def test_is_stale_returns_false_for_fresh_baseline() -> None:
    """A baseline built just now is not considered stale."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=3600
    )
    collector.collect()

    assert not collector._is_stale()


def test_is_stale_returns_true_after_ttl_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baseline whose age exceeds cache_ttl_seconds is stale."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=60
    )
    collector.collect()
    _advance_monotonic(monkeypatch, seconds=120)

    assert collector._is_stale()


# ---------------------------------------------------------------------------
# _rebuild_and_swap
# ---------------------------------------------------------------------------


def test_rebuild_and_swap_clears_flag_on_exception() -> None:
    """_rebuild_and_swap() resets the in-progress flag after an exception."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=3600
    )
    collector._rebuild_in_progress = True

    with patch.object(
        collector,
        "_build_baseline",
        side_effect=RuntimeError("build failed"),
    ):
        collector._rebuild_and_swap()

    assert not collector._rebuild_in_progress


def test_rebuild_and_swap_replaces_baseline_on_success() -> None:
    """_rebuild_and_swap() publishes a new baseline and clears the flag."""
    collector = MlflowObservabilityCollector(
        _empty_client(), cache_ttl_seconds=3600
    )
    collector.collect()
    initial_baseline = collector._baseline
    collector._rebuild_in_progress = True

    collector._rebuild_and_swap()

    assert not collector._rebuild_in_progress
    assert collector._baseline is not initial_baseline


# ---------------------------------------------------------------------------
# _scan_runs
# ---------------------------------------------------------------------------


def test_scan_runs_returns_zeroes_for_empty_experiment_list() -> None:
    """_scan_runs() with no experiment IDs returns zero for all statuses."""
    client = _empty_client()
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    result = collector._scan_runs([], filter_string="")

    assert result == {s: 0 for s in RUN_STATUSES}
    client.search_runs.assert_not_called()


def test_scan_runs_counts_runs_by_status() -> None:
    """_scan_runs() correctly aggregates run counts per status."""
    client = MagicMock()
    client.search_runs.return_value = FakePage(
        [
            _make_run("RUNNING"),
            _make_run("FINISHED"),
            _make_run("FINISHED"),
            _make_run("FAILED"),
        ]
    )
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    result = collector._scan_runs(["exp1"], filter_string="")

    assert result["RUNNING"] == 1
    assert result["FINISHED"] == 2
    assert result["FAILED"] == 1
    assert result["KILLED"] == 0


def test_scan_runs_ignores_unknown_status() -> None:
    """_scan_runs() silently ignores unrecognised run statuses."""
    client = MagicMock()
    client.search_runs.return_value = FakePage([_make_run("UNKNOWN")])
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    result = collector._scan_runs(["exp1"], filter_string="")

    assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# _scan_old_experiments
# ---------------------------------------------------------------------------


def test_scan_old_experiments_classifies_by_horizon() -> None:
    """Experiments older than horizon_ms go into old_ids; newer ones don't."""
    horizon_ms = 1_000_000
    old_exp = _make_experiment("exp-old", last_update_time=horizon_ms - 1)
    new_exp = _make_experiment("exp-new", last_update_time=horizon_ms + 1)
    client = MagicMock()
    client.search_experiments.return_value = FakePage([old_exp, new_exp])
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    old_ids, old_by_stage = collector._scan_old_experiments(horizon_ms)

    assert old_ids == ["exp-old"]
    assert old_by_stage == {"active": 1}


def test_scan_old_experiments_counts_deleted_stage() -> None:
    """Deleted experiments are counted under the 'deleted' key."""
    horizon_ms = 1_000_000
    deleted_exp = _make_experiment(
        "exp-del",
        last_update_time=horizon_ms - 1,
        lifecycle_stage="deleted",
    )
    client = MagicMock()
    client.search_experiments.return_value = FakePage([deleted_exp])
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    old_ids, old_by_stage = collector._scan_old_experiments(horizon_ms)

    assert old_ids == ["exp-del"]
    assert old_by_stage.get("deleted") == 1


# ---------------------------------------------------------------------------
# _scan_fresh_experiments
# ---------------------------------------------------------------------------


def test_scan_fresh_experiments_applies_horizon_filter_to_api() -> None:
    """_scan_fresh_experiments() passes the horizon filter to the API."""
    horizon_ms = 9_999_999
    fresh_exp = _make_experiment("exp-fresh", last_update_time=horizon_ms + 1)
    client = MagicMock()
    client.search_experiments.return_value = FakePage([fresh_exp])
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    ids, _ = collector._scan_fresh_experiments(horizon_ms)

    called_kwargs = client.search_experiments.call_args.kwargs
    assert str(horizon_ms) in called_kwargs.get("filter_string", "")
    assert ids == ["exp-fresh"]


# ---------------------------------------------------------------------------
# _scan_model_versions
# ---------------------------------------------------------------------------


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
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)

    total, by_stage = collector._scan_model_versions()

    assert total == 4
    assert by_stage["Production"] == 2
    assert by_stage["Staging"] == 1
    assert by_stage["None"] == 1


# ---------------------------------------------------------------------------
# _count_paginated
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _compute_snapshot
# ---------------------------------------------------------------------------


def test_compute_snapshot_merges_baseline_and_delta() -> None:
    """Snapshot totals are the sum of stable baseline and fresh delta."""
    client = MagicMock()
    fresh_exp = _make_experiment(
        "exp-new", last_update_time=10**15, lifecycle_stage="active"
    )
    client.search_experiments.return_value = FakePage([fresh_exp])
    client.search_runs.return_value = FakePage(
        [_make_run("RUNNING"), _make_run("RUNNING")]
    )
    collector = MlflowObservabilityCollector(client, cache_ttl_seconds=3600)
    # Inject a pre-built baseline to isolate _compute_snapshot.
    collector._baseline = _Baseline(
        old_experiment_ids=["exp-old"],
        old_experiments_by_stage={"active": 3, "deleted": 1},
        old_runs_by_status={
            "RUNNING": 1,
            "FINISHED": 5,
            "FAILED": 0,
            "KILLED": 0,
        },
        registered_models_total=2,
        model_versions_total=7,
        model_versions_by_stage={
            "Production": 3,
            "Staging": 2,
            "None": 2,
            "Archived": 0,
        },
        horizon_ms=500_000,
        built_at=time.monotonic(),
    )

    snapshot = collector._compute_snapshot()

    # 4 old experiments (3 active + 1 deleted) + 1 fresh active
    assert snapshot.experiments_total == 5
    assert snapshot.experiments_active_total == 4
    assert snapshot.experiments_deleted_total == 1
    # 1 old RUNNING + 2 fresh RUNNING
    assert snapshot.runs_by_status["RUNNING"] == 3
    assert snapshot.runs_by_status["FINISHED"] == 5
    assert snapshot.registered_models_total == 2
    assert snapshot.model_versions_total == 7
