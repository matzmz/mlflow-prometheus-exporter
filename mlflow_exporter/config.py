#!/usr/bin/env python3


"""CLI argument parsing, environment resolution, and MLflow client setup."""

import argparse
import os
from typing import Optional, Sequence

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_exporter.settings import (
    DEFAULT_EXPORTER_PORT,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TRACKING_URI,
    ExporterSettings,
)


def resolve_tracking_uri() -> str:
    """Return the MLflow tracking URI with explicit env-variable precedence.

    MLFLOW_TRACKING_URI (standard MLflow variable) takes priority over the
    legacy MLFLOW_URL variable used by earlier versions of this exporter.

    Returns:
    str: Resolved tracking URI from environment variables or the built-in
        default.
    """
    return (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("MLFLOW_URL")
        or DEFAULT_TRACKING_URI
    )


def parse_args(arguments: Optional[Sequence[str]] = None) -> ExporterSettings:
    """Build exporter settings from CLI arguments and environment variables.

    Parameters:
    arguments (Sequence[str] | None): Explicit argument list used in tests;
        defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
    ExporterSettings: Resolved configuration for the exporter process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port on which to expose the exporter server.",
        default=int(os.getenv("PORT", DEFAULT_EXPORTER_PORT)),
    )
    parser.add_argument(
        "--mlflowurl",
        "-u",
        type=str,
        help="MLflow tracking URI used for collecting data.",
        default=resolve_tracking_uri(),
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        help="Polling interval for collecting new metrics in seconds.",
        default=int(os.getenv("TIMEOUT", DEFAULT_POLL_INTERVAL_SECONDS)),
    )
    parser.add_argument(
        "--mlflow-username",
        type=str,
        help="Optional username for MLflow basic authentication.",
        default=os.getenv("MLFLOW_TRACKING_USERNAME"),
    )
    parser.add_argument(
        "--mlflow-password",
        type=str,
        help="Optional password or API key for MLflow basic authentication.",
        default=os.getenv("MLFLOW_TRACKING_PASSWORD"),
    )
    parsed = parser.parse_args(arguments)
    return ExporterSettings(
        port=parsed.port,
        poll_interval_seconds=parsed.timeout,
        tracking_uri=parsed.mlflowurl,
        tracking_username=parsed.mlflow_username,
        tracking_password=parsed.mlflow_password,
    )


def configure_mlflow_client(settings: ExporterSettings) -> MlflowClient:
    """Configure MLflow authentication from settings and return a client.

    Writes resolved credentials back to the standard MLflow environment
    variables so that any MLflow SDK call made later in the process
    automatically picks up the correct auth context.

    Parameters:
    settings (ExporterSettings): Resolved exporter configuration including
        tracking URI and optional credentials.

    Returns:
    MlflowClient: Client instance bound to the tracking URI in settings.
    """
    os.environ["MLFLOW_TRACKING_URI"] = settings.tracking_uri
    if settings.tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.tracking_username
    if settings.tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.tracking_password
    mlflow.set_tracking_uri(settings.tracking_uri)
    return MlflowClient(tracking_uri=settings.tracking_uri)
