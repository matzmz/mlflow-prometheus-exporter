#!/usr/bin/env python3


"""Orchestrate the MLflow Prometheus exporter: compose and start."""

import logging
import signal
from typing import Optional, Sequence

from mlflow_exporter.collector import MlflowObservabilityCollector
from mlflow_exporter.config import configure_mlflow_client, parse_args
from mlflow_exporter.metrics import PrometheusMetrics
from mlflow_exporter.runtime import ExporterRuntime
from mlflow_exporter.server import ExporterServer
from mlflow_exporter.settings import ExporterSettings

LOGGER = logging.getLogger(__name__)


def build_runtime(settings: ExporterSettings) -> ExporterRuntime:
    """Compose runtime dependencies for the exporter service.

    Parameters:
    settings (ExporterSettings): Resolved exporter configuration including
        port, tracking URI, and polling interval.
    """
    client = configure_mlflow_client(settings)
    collector = MlflowObservabilityCollector(
        client,
        baseline_interval_seconds=settings.baseline_interval_seconds,
    )
    metrics = PrometheusMetrics()
    server = ExporterServer()
    runtime = ExporterRuntime(
        settings=settings,
        collector=collector,
        metrics=metrics,
        server=server,
    )
    return runtime


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """Parse configuration and launch the exporter.

    Parameters:
    arguments (Sequence[str] | None): Optional argument list passed through
        to ``parse_args``; defaults to ``sys.argv[1:]`` when ``None``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = parse_args(arguments)
    LOGGER.info(
        "Starting exporter: tracking_uri=%s port=%d" " poll=%ds baseline=%ds",
        settings.tracking_uri,
        settings.port,
        settings.poll_interval_seconds,
        settings.baseline_interval_seconds,
    )
    runtime = build_runtime(settings)

    def request_shutdown(_signum: int, _frame: object) -> None:
        """Ask the runtime to stop gracefully on termination signals."""
        LOGGER.warning("Shutdown requested")
        runtime.stop()

    default_sigint = signal.getsignal(signal.SIGINT)
    default_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        runtime.run()
    finally:
        signal.signal(signal.SIGINT, default_sigint)
        signal.signal(signal.SIGTERM, default_sigterm)


if __name__ == "__main__":
    main()
