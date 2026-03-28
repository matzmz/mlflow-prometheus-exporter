"""Collector state objects.

This module intentionally contains only dataclasses that describe collector
state. Operational logic such as normalization or aggregation lives elsewhere.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from mlflow_exporter.models import MlflowSnapshot


@dataclass(frozen=True)
class _ExperimentRef:
    """Experiment metadata used to drive per-experiment recounts."""

    experiment_id: str
    last_update_time: int
    lifecycle_stage: str


@dataclass(frozen=True)
class _ExperimentBaseline:
    """Baseline metadata plus stable run counts for a single experiment."""

    experiment_id: str
    last_update_time: int
    lifecycle_stage: str
    stable_runs_by_status: Mapping[str, int]


@dataclass(frozen=True)
class _Baseline:
    """Immutable baseline snapshot published by the collector."""

    experiments_by_id: Mapping[str, _ExperimentBaseline]
    registered_models_total: int
    model_versions_total: int
    model_versions_by_stage: Mapping[str, int]
    horizon_ms: int
    built_at: float


@dataclass(frozen=True)
class _ExperimentScanResult:
    """Result of scanning experiments."""

    experiments: tuple[_ExperimentRef, ...]


@dataclass(frozen=True)
class _ModelVersionScanResult:
    """Result of scanning all model versions."""

    total: int
    by_stage: Mapping[str, int]


@dataclass(frozen=True)
class _RunCountsByExperimentScanResult:
    """Run counts grouped by experiment and then by status."""

    counts_by_experiment: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class _PublishedState:
    """Atomically published exporter state."""

    baseline: _Baseline
    snapshot: MlflowSnapshot
