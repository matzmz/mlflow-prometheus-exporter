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
| `mlflow_exporter_info{version, python_version}` | Info | Build metadata for the running exporter |


## Architecture

The exporter uses a **baseline + delta** caching strategy to keep polling overhead negligible even against large MLflow deployments:

- **Baseline** — a full scan of all experiments plus the **stable** run slice: terminal runs whose `end_time` is older than `HORIZON_DAYS` (default: 7 days). Built once during startup and then rebuilt every `BASELINE_INTERVAL_SECONDS` (default: 1 h) in a background thread.
- **Delta** — a lightweight rebuild of only the **volatile** run slice: all `RUNNING` runs plus terminal runs whose `end_time` is newer than the latest baseline horizon. Experiment metadata is refreshed separately using experiment `last_update_time`. Executed on every poll cycle (default: every 30 s).

The exporter starts an HTTP server immediately, exposing `/healthz` from the start and `/readyz` only after the initial baseline has completed. `/metrics` is always available but returns meaningful data only after bootstrap. If a baseline rebuild and delta refresh overlap, the exporter serves the last published snapshot instead of blocking scrapes.

### Endpoints

| Path | Description |
|------|-------------|
| `/healthz` | Liveness probe — always returns `200 OK` |
| `/readyz` | Readiness probe — returns `200` after bootstrap, `503` before |
| `/metrics` | Prometheus metrics endpoint |

Model-registry data (registered models and model versions) has no timestamp filter in the MLflow API and is therefore taken entirely from the baseline. The periodic baseline rebuild also acts as a repair pass for rare historical run changes that fall outside the volatile run window.


## Configuration

Operational settings can be provided through **CLI arguments** or **environment variables**. You can also preload environment variables from a `.env` file with `--env-file`. Precedence is: **CLI arguments** > **shell environment** > **`.env` file** > built-in defaults. Sensitive authentication values remain environment-variable based.

| CLI argument | Environment variable | Default | Description |
|---|---|---|---|
| `--env-file` | — | — | Path to a `.env` file loaded before resolving the other settings |
| `-p` / `--port` | `PORT` | `8000` | Port on which the exporter HTTP server listens |
| `--listen-address` | `LISTEN_ADDRESS` | `0.0.0.0` | Address on which the exporter HTTP server listens |
| `-u` / `--mlflowurl` | `MLFLOW_TRACKING_URI` | `http://localhost:5000/` | MLflow tracking server URI |
| — | `MLFLOW_URL` | — | Legacy fallback URI (used when `MLFLOW_TRACKING_URI` is not set) |
| `-t` / `--timeout` | `TIMEOUT` | `30` | Poll interval in seconds |
| `--baseline-interval` | `BASELINE_INTERVAL_SECONDS` | `3600` | Baseline rebuild interval in seconds |
| `--mlflow-request-timeout` | `MLFLOW_HTTP_REQUEST_TIMEOUT` | `30` | HTTP request timeout in seconds for MLflow API calls |
| `--mlflow-request-max-retries` | `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `3` | Maximum number of retries for failed MLflow API requests |
| — | `MLFLOW_TRACKING_USERNAME` | — | Username for MLflow basic authentication |
| — | `MLFLOW_TRACKING_PASSWORD` | — | Password or API key for MLflow basic authentication |


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

Using a `.env` file:

```shell
cp .env.example .env
mlflow-exporter --env-file .env
```

Verify metrics are being served:

```shell
curl http://localhost:8000/metrics
```

### Helm chart

A Helm chart for Kubernetes deployments is available in
[`helm/mlflow-prometheus-exporter`](./helm/mlflow-prometheus-exporter).

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
├── __init__.py            # package marker and version
├── collector/
│   ├── __init__.py
│   ├── assembler.py       # helper: normalization, merges, baseline/snapshot assembly
│   ├── manager.py         # refresh runtime: locks, loops, publication
│   ├── queries.py         # MLflow pagination, filters, and query adapter
│   └── state.py           # collector state dataclasses
├── config/
│   ├── __init__.py
│   ├── cli.py             # CLI parsing, env resolution, MlflowClient setup
│   ├── log.py             # logging configuration
│   └── settings.py        # constants and typed dataclasses
├── infra/
│   ├── __init__.py
│   ├── metrics.py         # Prometheus metric definitions and update logic
│   └── server.py          # HTTP server with /healthz, /readyz, /metrics
├── main.py                # composition root and process entrypoint
├── runtime.py             # runtime service coordinating collector + metrics
└── models.py              # shared domain value objects
tests/
├── helpers.py             # shared test factories
├── app/
│   ├── __init__.py
│   ├── test_main.py
│   └── test_runtime.py
├── collector/
│   ├── __init__.py
│   ├── test_assembler.py
│   ├── test_manager.py
│   └── test_queries.py
├── config/
│   ├── __init__.py
│   ├── test_cli.py
│   └── test_log.py
├── infra/
│   ├── __init__.py
│   ├── test_metrics.py
│   └── test_server.py
└── integration/
    └── test_mlflow_exporter.py
```

## License

[GPL-3.0](LICENSE)
