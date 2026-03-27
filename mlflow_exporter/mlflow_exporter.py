#!/usr/bin/env python3


"""Orchestrate the MLflow Prometheus exporter: compose, start, loop."""

import logging
import time
from typing import Optional, Sequence

from prometheus_client import start_http_server

from mlflow_exporter.collector import MlflowObservabilityCollector
from mlflow_exporter.config import configure_mlflow_client, parse_args
from mlflow_exporter.metrics import PrometheusMetrics
from mlflow_exporter.settings import ExporterSettings

LOGGER = logging.getLogger(__name__)


def run_exporter(settings: ExporterSettings) -> None:
    """Start the HTTP endpoint and refresh Prometheus metrics forever.

    Parameters:
    settings (ExporterSettings): Resolved exporter configuration including
        port, tracking URI, and polling interval.
    """
    client = configure_mlflow_client(settings)
    collector = MlflowObservabilityCollector(client)
    metrics = PrometheusMetrics()

    start_http_server(settings.port)
    while True:
        started = time.monotonic()
        try:
            snapshot = collector.collect()
            metrics.update_snapshot(snapshot)
            metrics.mark_success(time.monotonic() - started)
        except Exception:
            metrics.mark_failure(time.monotonic() - started)
            LOGGER.exception("MLflow metric collection failed")
        time.sleep(settings.poll_interval_seconds)


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """Parse configuration and launch the exporter.

    Parameters:
    arguments (Sequence[str] | None): Optional argument list passed through
        to ``parse_args``; defaults to ``sys.argv[1:]`` when ``None``.
    """
    logging.basicConfig(level=logging.INFO)
    run_exporter(parse_args(arguments))


if __name__ == "__main__":
    main()
