"""Unit tests for the logging configuration module."""

import json
import logging

from mlflow_exporter.log import configure_logging


def test_configure_logging_text_format(caplog: object) -> None:
    """configure_logging() with text format sets the expected formatter."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    configure_logging("DEBUG", "text")

    assert root.level == logging.DEBUG
    assert any(
        hasattr(h, "formatter")
        and h.formatter
        and "%(levelname)s" in (h.formatter._fmt or "")
        for h in root.handlers
    )
    for h in root.handlers[:]:
        root.removeHandler(h)


def test_configure_logging_json_format() -> None:
    """configure_logging() with json format emits valid JSON records."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    configure_logging("WARNING", "json")

    assert root.level == logging.WARNING
    handler = root.handlers[0]
    formatter = handler.formatter
    assert formatter is not None
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test"
    assert parsed["message"] == "hello world"
    for h in root.handlers[:]:
        root.removeHandler(h)
