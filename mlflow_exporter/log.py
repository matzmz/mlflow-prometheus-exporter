"""Logging configuration for the MLflow Prometheus exporter."""

import json
import logging

_TEXT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        return json.dumps(
            {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def configure_logging(level: str, fmt: str) -> None:
    """Set up the root logger with the requested level and format."""
    if fmt == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(level=level, format=_TEXT_FORMAT)
