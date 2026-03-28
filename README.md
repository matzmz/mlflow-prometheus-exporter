# MLflow Prometheus Exporter

A lightweight Prometheus exporter that scrapes an [MLflow](https://mlflow.org/) tracking server and exposes aggregated observability metrics in the [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/).

## Metrics

### Experiments

| Metric | Type | Description |
|--------|------|-------------|
| `mlflow_experiments_total` | Gauge | Total number of experiments (active + deleted) |
| `mlflow_experiments_active_total` | Gauge | Number of active experiments |
| `mlflow_experiments_deleted_total` | Gauge | Number of deleted experiments |

### Runs

| Metric | Type | Description |
|--------|------|-------------|
| `mlflow_runs_total` | Gauge | Total number of runs across all experiments |
| `mlflow_runs_by_status_total{status}` | Gauge | Run count per status: `RUNNING`, `FINISHED`, `FAILED`, `KILLED` |

### Model Registry

| Metric | Type | Description |
|--------|------|-------------|
| `mlflow_registered_models_total` | Gauge | Total number of registered models |
| `mlflow_model_versions_total` | Gauge | Total number of model versions |
| `mlflow_model_versions_by_stage_total{stage}` | Gauge | Model version count per stage: `None`, `Staging`, `Production`, `Archived` |

### Exporter health

| Metric | Type | Description |
|--------|------|-------------|
| `mlflow_exporter_collect_duration_seconds` | Histogram | Time spent collecting metrics from MLflow |
| `mlflow_exporter_last_collect_success` | Gauge | `1` if the last collection cycle succeeded, `0` otherwise |
| `mlflow_exporter_last_collect_timestamp_seconds` | Gauge | Unix timestamp of the last completed collection cycle |
| `mlflow_exporter_collect_errors_total` | Counter | Total number of failed collection cycles |
| `mlflow_exporter_baseline_duration_seconds` | Histogram | Time spent rebuilding the baseline |
| `mlflow_exporter_last_baseline_success` | Gauge | `1` if the last baseline rebuild succeeded, `0` otherwise |
| `mlflow_exporter_last_baseline_timestamp_seconds` | Gauge | Unix timestamp of the last completed baseline rebuild |
| `mlflow_exporter_baseline_errors_total` | Counter | Total number of failed baseline rebuilds |


## Architecture

The exporter uses a **baseline + delta** caching strategy to keep polling overhead negligible even against large MLflow deployments:

- **Baseline** — a full scan of all data older than `HORIZON_DAYS` (default: 7 days). Built once during startup and then rebuilt every `BASELINE_INTERVAL_SECONDS` (default: 1 h) in a background thread.
- **Delta** — a lightweight re-scan of only data newer than the latest baseline horizon, executed on every poll cycle (default: every 30 s).

The exporter starts serving `/metrics` only after the initial baseline has completed. If a baseline rebuild and delta refresh overlap, the exporter serves the last published snapshot instead of blocking scrapes.

Model-registry data (registered models and model versions) has no timestamp filter in the MLflow API and is therefore taken entirely from the baseline.


## Configuration

All settings can be provided through **CLI arguments** or **environment variables**. CLI arguments take precedence over environment variables.

| CLI argument | Environment variable | Default | Description |
|---|---|---|---|
| `-p` / `--port` | `PORT` | `8000` | Port on which the exporter HTTP server listens |
| `--listen-address` | `LISTEN_ADDRESS` | `0.0.0.0` | Address on which the exporter HTTP server listens |
| `-u` / `--mlflowurl` | `MLFLOW_TRACKING_URI` | `http://localhost:5000/` | MLflow tracking server URI |
| — | `MLFLOW_URL` | — | Legacy fallback URI (used when `MLFLOW_TRACKING_URI` is not set) |
| `-t` / `--timeout` | `TIMEOUT` | `30` | Poll interval in seconds |
| `--baseline-interval` | `BASELINE_INTERVAL_SECONDS` | `3600` | Baseline rebuild interval in seconds |
| `--mlflow-username` | `MLFLOW_TRACKING_USERNAME` | — | Username for MLflow basic authentication |
| `--mlflow-password` | `MLFLOW_TRACKING_PASSWORD` | — | Password or API key for MLflow basic authentication |


## Running

### Docker (recommended)

```shell
docker build -t mlflow-prometheus-exporter .

docker run -d \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow.internal.example:5000/ \
  --name mlflow-exporter \
  mlflow-prometheus-exporter
```

With authentication:

```shell
docker run -d \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=https://mlflow.internal.example \
  -e MLFLOW_TRACKING_USERNAME=name.lastname@example.com \
  -e MLFLOW_TRACKING_PASSWORD=mlflow_api_key \
  --name mlflow-exporter \
  mlflow-prometheus-exporter
```

### Local install

```shell
pip install .
mlflow-exporter --mlflowurl http://localhost:5000/ --port 8000 --timeout 30
```

Verify metrics are being served:

```shell
curl http://localhost:8000/metrics
```

---

## Development

### Requirements

- Python ≥ 3.10
- [tox](https://tox.wiki/)

### Quickstart

```shell
git clone https://github.com/matzmz/mlflow-prometheus-exporter/
cd mlflow-prometheus-exporter
pip install tox

tox -e unit        # run unit tests with coverage
tox -e integration # run integration tests (starts a live MLflow server)
tox -e lint        # codespell + flake8 + isort + black + mypy
tox -e fmt         # auto-format with isort + black
```

### Updating pinned dependencies

After editing any `*.in` file, regenerate the corresponding `*.txt` lock file:

```shell
tox -e update-requirements
```

Commit the updated `*.txt` files alongside the `*.in` changes.

### Project layout

```
mlflow_exporter/
├── __init__.py            # package marker
├── main.py                # composition root and process entrypoint
├── settings.py            # constants and typed dataclasses
├── config.py              # CLI parsing, env resolution, MlflowClient setup
├── collector.py           # MLflow data collection with baseline+delta cache
├── metrics.py             # Prometheus metric definitions and update logic
├── runtime.py             # runtime service coordinating collector + metrics
└── tests/
    ├── test_collector.py
    ├── test_exporter_config.py
    ├── test_metrics.py
    ├── test_orchestrator.py
    ├── test_runtime.py
    └── integration/
        └── test_mlflow_exporter.py
```

## License

[GPL-3.0](LICENSE)
