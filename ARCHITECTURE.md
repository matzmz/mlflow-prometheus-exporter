# Architecture

This document describes the runtime architecture of the MLflow Prometheus
Exporter.

## Overview

The exporter is structured around four responsibilities:

- `main.py`: composition root and process entrypoint
- `runtime.py`: application lifecycle and operational coordination
- `collector.py`: MLflow data acquisition, caching, and concurrency control
- `metrics.py`: Prometheus publication and health metrics

The core design goal is to reduce load on MLflow while keeping the exporter
simple and operationally safe:

- a blocking startup builds an initial `baseline`
- a background baseline worker periodically refreshes stable state
- a frequent delta loop refreshes recent state
- a shared lock prevents unmanaged concurrent refreshes
- stale data is acceptable during contention; partial data is not

## Module Responsibilities

### `mlflow_exporter/main.py`

This is the composition root.

It is responsible for:

- parsing runtime settings
- building the runtime and its dependencies
- installing signal handlers
- delegating shutdown requests to `runtime.stop()`

It is intentionally thin and contains no MLflow polling logic.

### `mlflow_exporter/runtime.py`

This is the application service.

It is responsible for:

- running the blocking bootstrap
- publishing the first snapshot to Prometheus
- marking baseline and delta health metrics
- starting the baseline background worker
- starting the HTTP server
- delegating the long-running delta loop to the collector
- stopping collector-owned loops during shutdown

It is the boundary between domain behavior (`collector`) and infrastructure
concerns (`prometheus_client`, process lifecycle).

### `mlflow_exporter/collector.py`

This is the domain core of the exporter.

It is responsible for:

- building a complete `baseline`
- building a `snapshot` from `baseline + delta`
- storing the last published immutable state
- ensuring only one refresh talks to MLflow at a time
- returning stale snapshots during lock contention
- running the baseline worker loop
- running the delta loop

Important internal concepts:

- `_Baseline`: immutable stable state older than the horizon
- `_PublishedState`: atomically published pair of `baseline + snapshot`
- `_refresh_lock`: serializes MLflow I/O between baseline and delta refreshes
- `_state_lock`: protects publication/retrieval of the current state
- `_stop_event`: coordinated stop signal for long-running loops

### `mlflow_exporter/metrics.py`

This module is a Prometheus adapter.

It is responsible for:

- publishing business metrics from `MlflowSnapshot`
- exposing delta collection health
- exposing baseline rebuild health

It does not know how snapshots are computed.

## Runtime Flow

### Startup

1. `main()` parses configuration.
2. `main()` builds `ExporterRuntime`.
3. `runtime.run()` calls `collector.initialize()`.
4. `collector.initialize()` performs a blocking baseline cycle.
5. The first full snapshot is published to Prometheus.
6. The baseline worker starts.
7. The HTTP server starts listening.
8. The collector-owned delta loop starts.

This guarantees that `/metrics` is not exposed before a valid snapshot exists.

### Baseline Refresh

The baseline worker periodically:

1. waits for the baseline interval
2. tries to acquire the shared refresh lock
3. rebuilds the baseline from MLflow
4. computes a merged snapshot from the new baseline
5. atomically publishes the new state
6. reports success or failure through callbacks

If the delta refresh is already using the lock, the baseline cycle skips that
iteration instead of forcing concurrent MLflow traffic.

### Delta Refresh

The delta loop periodically:

1. waits for the poll interval
2. tries to acquire the shared refresh lock
3. if the lock is busy, returns the current published snapshot
4. otherwise, re-computes the recent-data view from the latest baseline
5. atomically publishes the new snapshot
6. reports success or failure through callbacks

This means the exporter prefers coherent stale data over unsafe concurrency.

## Concurrency Model

The architecture uses two locks and one stop event:

- `_refresh_lock`
  - serializes MLflow I/O
  - shared by baseline and delta refreshes
- `_state_lock`
  - protects the published in-memory state
  - held only briefly during read/write of `_PublishedState`
- `_stop_event`
  - stops baseline and delta loops
  - used for graceful shutdown

Operationally:

- baseline and delta are never allowed to query MLflow concurrently
- readers never see partially published state
- the current snapshot can remain stale if a refresh is delayed or skipped

## Data Model

The exporter distinguishes between:

- `baseline`
  - stable historical view, rebuilt periodically
- `snapshot`
  - exported view currently served to Prometheus
- `delta`
  - not stored as a first-class object
  - computed on demand from the latest baseline and merged immediately

This keeps the state model simple:

- one published baseline
- one published snapshot
- no separate mutable delta cache to reconcile later

## Mermaid Diagrams

### Class Diagram

```mermaid
classDiagram
    class ExporterSettings {
        +int port
        +str listen_address
        +int poll_interval_seconds
        +int baseline_interval_seconds
        +str tracking_uri
        +Optional[str] tracking_username
        +Optional[str] tracking_password
    }

    class MlflowSnapshot {
        +int experiments_total
        +int experiments_active_total
        +int experiments_deleted_total
        +int runs_total
        +dict runs_by_status
        +int registered_models_total
        +int model_versions_total
        +dict model_versions_by_stage
    }

    class _Baseline {
        +list old_experiment_ids
        +dict old_experiments_by_stage
        +dict old_runs_by_status
        +int registered_models_total
        +int model_versions_total
        +dict model_versions_by_stage
        +int horizon_ms
        +float built_at
    }

    class _PublishedState {
        +_Baseline baseline
        +MlflowSnapshot snapshot
    }

    class PrometheusMetrics {
        +update_snapshot(snapshot)
        +mark_success(duration_seconds)
        +mark_failure(duration_seconds)
        +mark_baseline_success(duration_seconds)
        +mark_baseline_failure(duration_seconds)
    }

    class MlflowObservabilityCollector {
        -MlflowClient _client
        -Lock _refresh_lock
        -Lock _state_lock
        -Event _stop_event
        -_PublishedState _state
        +initialize()
        +start_baseline_worker_with_callbacks(on_snapshot, on_failure)
        +run_delta_refresh_loop(poll_interval_seconds, on_snapshot, on_failure)
        +stop()
        +current_snapshot()
    }

    class ExporterRuntime {
        -ExporterSettings _settings
        -MlflowObservabilityCollector _collector
        -PrometheusMetrics _metrics
        -Callable _start_http_server
        +run()
        +stop()
    }

    class main_py {
        +build_runtime(settings)
        +main(arguments)
    }

    main_py --> ExporterRuntime : builds
    ExporterRuntime --> MlflowObservabilityCollector : uses
    ExporterRuntime --> PrometheusMetrics : uses
    MlflowObservabilityCollector --> _PublishedState : publishes
    _PublishedState --> _Baseline : contains
    _PublishedState --> MlflowSnapshot : contains
    ExporterRuntime --> ExporterSettings : configured by
```

### State Diagram

```mermaid
stateDiagram-v2
    [*] --> Bootstrapping

    Bootstrapping --> Ready: initial baseline built and snapshot published
    Bootstrapping --> FailedStartup: bootstrap exception

    Ready --> BaselineRefresh: baseline interval elapsed
    Ready --> DeltaRefresh: delta poll interval elapsed
    Ready --> Stopping: signal received

    BaselineRefresh --> Ready: baseline rebuilt and snapshot published
    BaselineRefresh --> Ready: lock busy, cycle skipped
    BaselineRefresh --> Ready: failure logged and metric updated

    DeltaRefresh --> Ready: snapshot refreshed and published
    DeltaRefresh --> Ready: lock busy, stale snapshot kept
    DeltaRefresh --> Ready: failure logged and metric updated

    Stopping --> Stopped: stop_event set, loops exit
    FailedStartup --> [*]
    Stopped --> [*]
```

### Iteration Diagram

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Runtime as ExporterRuntime
    participant Collector as MlflowObservabilityCollector
    participant MLflow as MLflow Server
    participant Metrics as PrometheusMetrics
    participant HTTP as Prometheus HTTP Server

    Main->>Runtime: build_runtime(settings)
    Main->>Runtime: run()

    Runtime->>Collector: initialize()
    Collector->>MLflow: build baseline
    Collector->>MLflow: build snapshot from baseline
    Collector-->>Runtime: initial snapshot
    Runtime->>Metrics: update_snapshot(initial)
    Runtime->>Metrics: mark_success(bootstrap_duration)
    Runtime->>Metrics: mark_baseline_success(bootstrap_duration)
    Runtime->>Collector: start_baseline_worker_with_callbacks(...)
    Runtime->>HTTP: start_http_server(port, addr)
    Runtime->>Collector: run_delta_refresh_loop(...)

    loop Every delta interval
        Collector->>Collector: wait(stop_event, poll_interval)
        Collector->>Collector: try refresh lock
        alt lock acquired
            Collector->>MLflow: scan fresh experiments/runs
            Collector->>Collector: publish new snapshot
            Collector->>Metrics: on_snapshot callback
        else lock busy
            Collector->>Collector: keep stale snapshot
            Collector->>Metrics: on_snapshot callback
        end
    end

    par Every baseline interval
        Collector->>Collector: wait(stop_event, baseline_interval)
        Collector->>Collector: try refresh lock
        alt lock acquired
            Collector->>MLflow: rebuild baseline
            Collector->>MLflow: rebuild snapshot
            Collector->>Collector: publish new baseline+snapshot
            Collector->>Metrics: baseline success callback
        else lock busy
            Collector->>Collector: skip cycle
        end
    end
```

## Current Tradeoffs

The current architecture intentionally prefers simplicity over maximal
incremental sophistication.

Notable tradeoffs:

- stale-but-coherent data is preferred over concurrent refreshes
- baseline and delta share one MLflow I/O lock
- delta is recomputed from the latest baseline instead of stored separately
- correctness around long-lived run status transitions is not yet fully
  optimized; this is the main known architectural limitation

## Production-Oriented Behaviors Already Present

- startup gating before serving `/metrics`
- explicit baseline and delta health metrics
- graceful stop via `runtime.stop()`
- configurable listen address
- Docker healthcheck
- unit, integration, and lint coverage

## Known Architectural Limitation

The main deferred issue is the temporal correctness of run status changes for
runs that started before the baseline horizon and completed later.

