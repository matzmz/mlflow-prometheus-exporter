

"""Runtime service for the MLflow Prometheus exporter."""

import logging
import time
from typing import Callable

from mlflow_exporter.collector import MlflowObservabilityCollector
from mlflow_exporter.metrics import PrometheusMetrics
from mlflow_exporter.settings import ExporterSettings, MlflowSnapshot

LOGGER = logging.getLogger(__name__)


class ExporterRuntime:
    """Own the running exporter lifecycle after dependencies are composed."""

    def __init__(
        self,
        settings: ExporterSettings,
        collector: MlflowObservabilityCollector,
        metrics: PrometheusMetrics,
        start_http_server: Callable[..., None],
    ) -> None:
        """Store runtime collaborators and infrastructure hooks."""
        self._settings = settings
        self._collector = collector
        self._metrics = metrics
        self._start_http_server = start_http_server

    def run(self) -> None:
        """Bootstrap the collector, expose metrics, and start refresh loops."""
        try:
            LOGGER.info("Bootstrapping initial baseline")
            started = time.monotonic()
            snapshot = self._collector.initialize()
            duration_seconds = time.monotonic() - started
            LOGGER.info(
                "Bootstrap complete in %.2fs: %d experiments, %d runs,"
                " %d models",
                duration_seconds,
                snapshot.experiments_total,
                snapshot.runs_total,
                snapshot.registered_models_total,
            )
            self._publish_snapshot(snapshot, duration_seconds)
            self._metrics.mark_baseline_success(duration_seconds)
            self._collector.start_baseline_worker_with_callbacks(
                on_snapshot=self._publish_baseline_snapshot,
                on_failure=self._metrics.mark_baseline_failure,
            )
            self._start_http_server(
                self._settings.port,
                addr=self._settings.listen_address,
            )
            LOGGER.info(
                "HTTP server listening on %s:%d",
                self._settings.listen_address,
                self._settings.port,
            )
            self._collector.run_delta_refresh_loop(
                poll_interval_seconds=self._settings.poll_interval_seconds,
                on_snapshot=self._publish_snapshot,
                on_failure=self._metrics.mark_failure,
            )
        finally:
            self.stop()

    def stop(self) -> None:
        """Request a coordinated shutdown of collector-owned loops."""
        LOGGER.info("Stopping exporter")
        self._collector.stop()

    def _publish_snapshot(
        self, snapshot: MlflowSnapshot, duration_seconds: float
    ) -> None:
        """Publish a refreshed snapshot and mark the cycle successful."""
        self._metrics.update_snapshot(snapshot)
        self._metrics.mark_success(duration_seconds)

    def _publish_baseline_snapshot(
        self, snapshot: MlflowSnapshot, duration_seconds: float
    ) -> None:
        """Publish a baseline refresh and mark both health signals."""
        self._publish_snapshot(snapshot, duration_seconds)
        self._metrics.mark_baseline_success(duration_seconds)
