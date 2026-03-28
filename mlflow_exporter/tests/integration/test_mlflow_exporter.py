#!/usr/bin/env python3


"""Integration tests for testing mlflow exporter with local MLflow server."""

import socket
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_delay, wait_fixed

TEST_EXPORTER_TIMEOUT = 20


@retry(stop=stop_after_delay(60), wait=wait_fixed(1))
def _wait_for_mlflow_ready(tracking_url: str) -> None:
    """Poll the MLflow server until it accepts requests."""
    requests.get(f"{tracking_url}health", timeout=2).raise_for_status()


@pytest.fixture
def exporter_server(tmp_path):
    """Deploy local MLflow server with the exporter on different processes.

    Test data is seeded after MLflow is ready but *before* the exporter
    starts, so that the exporter's first baseline snapshot already contains
    the registered model and all runs.
    """
    mlflow_port = _get_free_port()
    exporter_port = _get_free_port()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    tracking_url = f"http://127.0.0.1:{mlflow_port}/"

    mlflow_server_process = subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(mlflow_port),
            "--backend-store-uri",
            _build_sqlite_uri(tmp_path / "mlflow.db"),
            "--default-artifact-root",
            artifact_root.as_uri(),
        ]
    )

    _wait_for_mlflow_ready(tracking_url)

    client = MlflowClient(tracking_uri=tracking_url)
    finished_run = client.create_run("0")
    failed_run = client.create_run("0")
    killed_run = client.create_run("0")
    running_run = client.create_run("0")
    client.set_terminated(finished_run.info.run_id, status="FINISHED")
    client.set_terminated(failed_run.info.run_id, status="FAILED")
    client.set_terminated(killed_run.info.run_id, status="KILLED")
    try:
        client.create_registered_model("model_name")
    except RestException:
        pass

    exporter_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow_exporter.main",
            "--mlflowurl",
            tracking_url,
            "-t",
            str(TEST_EXPORTER_TIMEOUT),
            "-p",
            str(exporter_port),
        ]
    )

    yield {
        "tracking_url": tracking_url,
        "exporter_port": exporter_port,
        "running_run": running_run,
    }

    # Clean up after the processes
    _stop_process(exporter_process)
    _stop_process(mlflow_server_process)


@retry(stop=stop_after_delay(TEST_EXPORTER_TIMEOUT), wait=wait_fixed(1))
def verify_metrics(exporter_port):
    """Try to get metrics every 1 second."""
    response = requests.get(f"http://127.0.0.1:{exporter_port}/metrics")
    response.raise_for_status()  # Raise exception if the request was not successful
    metrics_text = response.text

    assert "mlflow_experiments_total 1.0" in metrics_text
    assert "mlflow_experiments_active_total 1.0" in metrics_text
    assert "mlflow_experiments_deleted_total 0.0" in metrics_text
    assert "mlflow_runs_total 4.0" in metrics_text
    assert 'mlflow_runs_by_status_total{status="RUNNING"} 1.0' in metrics_text
    assert 'mlflow_runs_by_status_total{status="FINISHED"} 1.0' in metrics_text
    assert 'mlflow_runs_by_status_total{status="FAILED"} 1.0' in metrics_text
    assert 'mlflow_runs_by_status_total{status="KILLED"} 1.0' in metrics_text
    assert "mlflow_registered_models_total 1.0" in metrics_text
    assert "mlflow_model_versions_total 0.0" in metrics_text
    assert (
        'mlflow_model_versions_by_stage_total{stage="Production"} 0.0'
        in metrics_text
    )
    assert "mlflow_exporter_last_collect_success 1.0" in metrics_text


def test_exporter_integration(exporter_server):
    """Verify that the exporter exposes the expected MLflow metrics."""
    assert exporter_server["running_run"].info.status == "RUNNING"
    verify_metrics(exporter_server["exporter_port"])


def _build_sqlite_uri(database_path: Path) -> str:
    """Build a SQLite URI compatible with the MLflow server CLI."""
    return f"sqlite:///{database_path}"


def _get_free_port() -> int:
    """Reserve an ephemeral TCP port for a local test server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate a spawned process and wait for its exit."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__])
