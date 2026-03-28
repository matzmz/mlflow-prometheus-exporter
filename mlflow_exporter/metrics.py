#!/usr/bin/env python3


"""Prometheus metric definitions and update logic."""

import platform
import time

from prometheus_client import Counter, Gauge, Histogram, Info

import mlflow_exporter
from mlflow_exporter.settings import MlflowSnapshot


class PrometheusMetrics:
    """Encapsulate all Prometheus metrics exposed by the exporter."""

    def __init__(self):
        """Register all Prometheus metrics."""
        self.build_info = Info(
            "mlflow_exporter",
            "MLflow Prometheus exporter build information.",
        )
        self.build_info.info(
            {
                "version": mlflow_exporter.__version__,
                "python_version": platform.python_version(),
            }
        )
        self.experiments_total = Gauge(
            "mlflow_experiments_total",
            "Total number of MLflow experiments.",
        )
        self.experiments_active_total = Gauge(
            "mlflow_experiments_active_total",
            "Number of active MLflow experiments.",
        )
        self.experiments_deleted_total = Gauge(
            "mlflow_experiments_deleted_total",
            "Number of deleted MLflow experiments.",
        )
        self.runs_total = Gauge(
            "mlflow_runs_total",
            "Total number of MLflow runs.",
        )
        self.runs_by_status = Gauge(
            "mlflow_runs_by_status_total",
            "Number of MLflow runs grouped by status.",
            ["status"],
        )
        self.registered_models_total = Gauge(
            "mlflow_registered_models_total",
            "Total number of registered MLflow models.",
        )
        self.model_versions_total = Gauge(
            "mlflow_model_versions_total",
            "Total number of MLflow model versions.",
        )
        self.model_versions_by_stage = Gauge(
            "mlflow_model_versions_by_stage_total",
            "Number of MLflow model versions grouped by stage.",
            ["stage"],
        )
        self.collect_duration_seconds = Histogram(
            "mlflow_exporter_collect_duration_seconds",
            "Time spent collecting metrics from MLflow.",
        )
        self.collect_success = Gauge(
            "mlflow_exporter_last_collect_success",
            "Whether the latest MLflow collection cycle succeeded.",
        )
        self.collect_timestamp_seconds = Gauge(
            "mlflow_exporter_last_collect_timestamp_seconds",
            "Unix timestamp of the latest completed collection cycle.",
        )
        self.collect_errors_total = Counter(
            "mlflow_exporter_collect_errors_total",
            "Total number of failed MLflow collection cycles.",
        )
        self.baseline_duration_seconds = Histogram(
            "mlflow_exporter_baseline_duration_seconds",
            "Time spent rebuilding the MLflow baseline.",
        )
        self.baseline_success = Gauge(
            "mlflow_exporter_last_baseline_success",
            "Whether the latest baseline rebuild succeeded.",
        )
        self.baseline_timestamp_seconds = Gauge(
            "mlflow_exporter_last_baseline_timestamp_seconds",
            "Unix timestamp of the latest completed baseline rebuild.",
        )
        self.baseline_errors_total = Counter(
            "mlflow_exporter_baseline_errors_total",
            "Total number of failed baseline rebuilds.",
        )

    def update_snapshot(self, snapshot: MlflowSnapshot) -> None:
        """Publish a successful MLflow snapshot to Prometheus."""
        self.experiments_total.set(snapshot.experiments_total)
        self.experiments_active_total.set(snapshot.experiments_active_total)
        self.experiments_deleted_total.set(snapshot.experiments_deleted_total)
        self.runs_total.set(snapshot.runs_total)
        for status, count in snapshot.runs_by_status.items():
            self.runs_by_status.labels(status=status).set(count)
        self.registered_models_total.set(snapshot.registered_models_total)
        self.model_versions_total.set(snapshot.model_versions_total)
        for stage, count in snapshot.model_versions_by_stage.items():
            self.model_versions_by_stage.labels(stage=stage).set(count)

    def mark_success(self, duration_seconds: float) -> None:
        """Record a successful collection cycle."""
        self.collect_duration_seconds.observe(duration_seconds)
        self.collect_success.set(1)
        self.collect_timestamp_seconds.set(time.time())

    def mark_failure(self, duration_seconds: float) -> None:
        """Record a failed collection cycle."""
        self.collect_duration_seconds.observe(duration_seconds)
        self.collect_success.set(0)
        self.collect_timestamp_seconds.set(time.time())
        self.collect_errors_total.inc()

    def mark_baseline_success(self, duration_seconds: float) -> None:
        """Record a successful baseline rebuild."""
        self.baseline_duration_seconds.observe(duration_seconds)
        self.baseline_success.set(1)
        self.baseline_timestamp_seconds.set(time.time())

    def mark_baseline_failure(self, duration_seconds: float) -> None:
        """Record a failed baseline rebuild."""
        self.baseline_duration_seconds.observe(duration_seconds)
        self.baseline_success.set(0)
        self.baseline_timestamp_seconds.set(time.time())
        self.baseline_errors_total.inc()
