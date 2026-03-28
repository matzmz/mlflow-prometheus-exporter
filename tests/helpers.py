"""Shared test factory functions for the mlflow_exporter test suite."""

from types import SimpleNamespace

from mlflow_exporter.config.settings import (
    MODEL_STAGES,
    RUN_STATUSES,
    ExporterSettings,
)
from mlflow_exporter.models import MlflowSnapshot


class FakePage(list):
    """List with an optional pagination token, mimicking MLflow PagedList."""

    def __init__(self, items: list, token: str | None = None) -> None:
        """Initialise with items and an optional next-page token."""
        super().__init__(items)
        self.token = token


def make_settings(**overrides: object) -> ExporterSettings:
    """Return runtime settings with test-friendly defaults."""
    defaults: dict = dict(
        port=9999,
        listen_address="0.0.0.0",
        poll_interval_seconds=30,
        baseline_interval_seconds=3600,
        tracking_uri="http://localhost:5000/",
        tracking_username=None,
        tracking_password=None,
        mlflow_request_timeout_seconds=30,
        mlflow_request_max_retries=3,
        log_level="INFO",
        log_format="text",
    )
    defaults.update(overrides)
    return ExporterSettings(**defaults)


def make_snapshot(
    experiments_total: int = 0,
    experiments_active_total: int = 0,
    experiments_deleted_total: int = 0,
    runs_total: int = 0,
    registered_models_total: int = 0,
    model_versions_total: int = 0,
) -> MlflowSnapshot:
    """Return a MlflowSnapshot with sensible test defaults."""
    return MlflowSnapshot(
        experiments_total=experiments_total,
        experiments_active_total=experiments_active_total,
        experiments_deleted_total=experiments_deleted_total,
        runs_total=runs_total,
        runs_by_status={s: 0 for s in RUN_STATUSES},
        registered_models_total=registered_models_total,
        model_versions_total=model_versions_total,
        model_versions_by_stage={s: 0 for s in MODEL_STAGES},
    )


def make_experiment(
    experiment_id: str,
    last_update_time: int,
    lifecycle_stage: str = "active",
) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow Experiment object."""
    return SimpleNamespace(
        experiment_id=experiment_id,
        last_update_time=last_update_time,
        lifecycle_stage=lifecycle_stage,
    )


def make_run(
    status: str,
    experiment_id: str = "exp1",
) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow Run object."""
    return SimpleNamespace(
        info=SimpleNamespace(status=status, experiment_id=experiment_id)
    )


def make_model_version(stage: str) -> SimpleNamespace:
    """Return a minimal stand-in for an MLflow ModelVersion object."""
    return SimpleNamespace(current_stage=stage)
