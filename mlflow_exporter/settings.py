#!/usr/bin/env python3


"""Runtime constants and typed configuration data structures."""

from dataclasses import dataclass
from typing import Optional

DEFAULT_EXPORTER_PORT = 8000
DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_TRACKING_URI = "http://localhost:5000/"

EXPERIMENT_PAGE_SIZE = 1_000
MODEL_PAGE_SIZE = 100
MODEL_VERSION_PAGE_SIZE = 10_000
RUN_PAGE_SIZE = 1_000

RUN_STATUSES = ("RUNNING", "FINISHED", "FAILED", "KILLED")
MODEL_STAGES = ("None", "Staging", "Production", "Archived")

# Baseline cache is rebuilt every CACHE_TTL_SECONDS. Between rebuilds
# only data newer than HORIZON_DAYS is re-scanned on every poll.
CACHE_TTL_SECONDS: int = 3_600
HORIZON_DAYS: int = 7


@dataclass(frozen=True)
class ExporterSettings:
    """Runtime configuration for the exporter process."""

    port: int
    poll_interval_seconds: int
    tracking_uri: str
    tracking_username: Optional[str]
    tracking_password: Optional[str]


@dataclass(frozen=True)
class MlflowSnapshot:
    """Aggregated MLflow state exported to Prometheus."""

    experiments_total: int
    experiments_active_total: int
    experiments_deleted_total: int
    runs_total: int
    runs_by_status: dict
    registered_models_total: int
    model_versions_total: int
    model_versions_by_stage: dict
