"""Unit tests for exporter configuration helpers."""

import os
from unittest.mock import patch

import pytest

from mlflow_exporter.config.cli import (
    configure_mlflow_client,
    parse_args,
    resolve_tracking_uri,
)
from mlflow_exporter.config.settings import (
    DEFAULT_TRACKING_URI,
    ExporterSettings,
)


def test_parse_args_prefers_tracking_uri_environment(monkeypatch):
    """Prefer the standard MLflow tracking URI over the legacy exporter one."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example")
    monkeypatch.setenv("MLFLOW_URL", "https://legacy.example")

    settings = parse_args([])

    assert settings.tracking_uri == "https://tracking.example"


def test_parse_args_reads_optional_tracking_credentials(monkeypatch):
    """Load optional MLflow basic authentication credentials from the env."""
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "name.lastname@example.com")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret-api-key")

    settings = parse_args([])

    assert settings.tracking_username == "name.lastname@example.com"
    assert settings.tracking_password == "secret-api-key"


def test_parse_args_reads_mlflow_http_settings(monkeypatch) -> None:
    """Load MLflow HTTP timeout and retry settings into ExporterSettings."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "45")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")

    settings = parse_args([])

    assert settings.mlflow_request_timeout_seconds == 45
    assert settings.mlflow_request_max_retries == 7


def test_parse_args_loads_values_from_env_file(tmp_path, monkeypatch) -> None:
    """Load exporter settings from a provided .env file."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MLFLOW_TRACKING_URI=https://dotenv.example\nPORT=9100\n",
        encoding="utf-8",
    )

    settings = parse_args(["--env-file", str(env_file)])

    assert settings.tracking_uri == "https://dotenv.example"
    assert settings.port == 9100


def test_parse_args_prefers_shell_environment_over_env_file(
    tmp_path, monkeypatch
) -> None:
    """Keep explicit process env vars ahead of .env file values."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://shell.example")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MLFLOW_TRACKING_URI=https://dotenv.example\n",
        encoding="utf-8",
    )

    settings = parse_args(["--env-file", str(env_file)])

    assert settings.tracking_uri == "https://shell.example"


def test_parse_args_prefers_cli_over_env_file(tmp_path, monkeypatch) -> None:
    """Keep explicit CLI arguments ahead of values loaded from .env."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MLFLOW_TRACKING_URI=https://dotenv.example\n",
        encoding="utf-8",
    )

    settings = parse_args(
        [
            "--env-file",
            str(env_file),
            "--mlflowurl",
            "https://cli.example",
        ]
    )

    assert settings.tracking_uri == "https://cli.example"


def test_configure_mlflow_client_applies_credentials(monkeypatch):
    """Expose resolved MLflow connection settings through standard env vars."""
    settings = ExporterSettings(
        port=8000,
        listen_address="0.0.0.0",
        poll_interval_seconds=30,
        baseline_interval_seconds=3600,
        tracking_uri="https://tracking.example",
        tracking_username="name.lastname@example.com",
        tracking_password="secret-api-key",
        mlflow_request_timeout_seconds=45,
        mlflow_request_max_retries=7,
        log_level="INFO",
        log_format="text",
    )

    with patch(
        "mlflow_exporter.config.cli.mlflow.set_tracking_uri"
    ) as mock_set_uri:
        client = configure_mlflow_client(settings)

    assert client.tracking_uri == "https://tracking.example"
    assert mock_set_uri.call_args.args == ("https://tracking.example",)
    assert os.environ["MLFLOW_TRACKING_URI"] == "https://tracking.example"
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "45"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "7"
    assert (
        os.environ["MLFLOW_TRACKING_USERNAME"] == "name.lastname@example.com"
    )
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "secret-api-key"


def test_resolve_tracking_uri_returns_default_when_no_env(
    monkeypatch,
) -> None:
    """resolve_tracking_uri() falls back to the built-in default."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_URL", raising=False)

    assert resolve_tracking_uri() == DEFAULT_TRACKING_URI


def test_resolve_tracking_uri_falls_back_to_mlflow_url(monkeypatch) -> None:
    """resolve_tracking_uri() uses MLFLOW_URL when MLFLOW_TRACKING_URI absent."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_URL", "https://legacy.example")

    assert resolve_tracking_uri() == "https://legacy.example"


def test_parse_args_accepts_cli_arguments() -> None:
    """parse_args() builds ExporterSettings from explicit CLI arguments."""
    settings = parse_args(
        [
            "--port",
            "9090",
            "--listen-address",
            "127.0.0.1",
            "--mlflowurl",
            "https://remote.example",
            "--timeout",
            "60",
            "--baseline-interval",
            "600",
            "--mlflow-request-timeout",
            "15",
            "--mlflow-request-max-retries",
            "9",
        ]
    )

    assert settings.port == 9090
    assert settings.listen_address == "127.0.0.1"
    assert settings.tracking_uri == "https://remote.example"
    assert settings.poll_interval_seconds == 60
    assert settings.baseline_interval_seconds == 600
    assert settings.mlflow_request_timeout_seconds == 15
    assert settings.mlflow_request_max_retries == 9


def test_configure_mlflow_client_without_credentials(monkeypatch) -> None:
    """configure_mlflow_client() clears stale auth env vars when absent."""
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "stale-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "stale-pass")
    settings = ExporterSettings(
        port=8000,
        listen_address="0.0.0.0",
        poll_interval_seconds=30,
        baseline_interval_seconds=3600,
        tracking_uri="https://tracking.example",
        tracking_username=None,
        tracking_password=None,
        mlflow_request_timeout_seconds=30,
        mlflow_request_max_retries=3,
        log_level="INFO",
        log_format="text",
    )

    with patch("mlflow_exporter.config.cli.mlflow.set_tracking_uri"):
        configure_mlflow_client(settings)

    assert "MLFLOW_TRACKING_USERNAME" not in os.environ
    assert "MLFLOW_TRACKING_PASSWORD" not in os.environ


def test_parse_args_rejects_non_positive_timeout() -> None:
    """parse_args() rejects non-positive poll intervals."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--timeout", "0"])


def test_parse_args_rejects_non_positive_baseline_interval() -> None:
    """parse_args() rejects non-positive baseline intervals."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--baseline-interval", "-1"])


def test_parse_args_rejects_invalid_port() -> None:
    """parse_args() rejects ports outside the valid TCP range."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--port", "70000"])


def test_parse_args_rejects_non_positive_mlflow_request_timeout() -> None:
    """parse_args() rejects non-positive MLflow HTTP timeouts."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--mlflow-request-timeout", "0"])


def test_parse_args_rejects_negative_mlflow_request_max_retries() -> None:
    """parse_args() rejects negative MLflow HTTP retry limits."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--mlflow-request-max-retries", "-1"])


def test_parse_args_accepts_log_level_and_format() -> None:
    """parse_args() accepts --log-level and --log-format CLI arguments."""
    settings = parse_args(["--log-level", "DEBUG", "--log-format", "json"])

    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"


def test_parse_args_normalises_log_level_case() -> None:
    """parse_args() normalises log level to uppercase."""
    settings = parse_args(["--log-level", "warning"])

    assert settings.log_level == "WARNING"


def test_parse_args_normalises_log_format_case() -> None:
    """parse_args() normalises log format to lowercase."""
    settings = parse_args(["--log-format", "JSON"])

    assert settings.log_format == "json"


def test_parse_args_rejects_invalid_log_level() -> None:
    """parse_args() rejects unrecognised log levels."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--log-level", "VERBOSE"])


def test_parse_args_rejects_invalid_log_format() -> None:
    """parse_args() rejects unrecognised log formats."""
    with patch("argparse.ArgumentParser.error", side_effect=ValueError):
        with pytest.raises(ValueError):
            parse_args(["--log-format", "xml"])
