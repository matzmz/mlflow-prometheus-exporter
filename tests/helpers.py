"""Shared test factory functions for the mlflow_exporter test suite."""

from mlflow_exporter.settings import (
    MODEL_STAGES,
    RUN_STATUSES,
    ExporterSettings,
    MlflowSnapshot,
)


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
