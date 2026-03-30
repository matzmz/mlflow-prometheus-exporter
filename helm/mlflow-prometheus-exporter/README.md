# MLflow Prometheus Exporter Helm Chart

This chart deploys:

- one `Deployment` for the exporter
- one internal `Service` (`ClusterIP` by default)
- one `ConfigMap` for non-sensitive configuration
- one optional `Secret` for MLflow basic auth
- one optional `ScrapeConfig` for Prometheus Operator

## Install

```bash
helm upgrade --install mlflow-exporter ./helm/mlflow-prometheus-exporter \
  --namespace monitoring \
  --create-namespace \
  --set image.repository=ghcr.io/your-org/mlflow-prometheus-exporter \
  --set image.tag=0.1.0 \
  --set mlflow.trackingUri=https://mlflow.internal.example \
  --set mlflow.auth.username=mlflow-user \
  --set mlflow.auth.password=mlflow-password
```

## Internal service

The chart exposes the exporter through an internal Kubernetes `Service`.
By default it is:

- `type: ClusterIP`
- port `8000`
- metrics path `/metrics`

## Prometheus Operator `ScrapeConfig`

The chart can create a `ScrapeConfig` resource with API version
`monitoring.coreos.com/v1alpha1`.

Prometheus Operator documents that `ScrapeConfig` resources are discovered
through the `Prometheus.spec.scrapeConfigSelector.matchLabels` selector, and
optionally through `scrapeConfigNamespaceSelector`.

Official docs:

- https://prometheus-operator.dev/docs/developer/scrapeconfig/
- https://prometheus-operator.dev/docs/api-reference/api/

To make Prometheus pick up the generated `ScrapeConfig`, add matching labels in
`values.yaml`, for example:

```yaml
scrapeConfig:
  enabled: true
  labels:
    prometheus: system-monitoring-prometheus
```

Then ensure your `Prometheus` resource selects the same label:

```yaml
spec:
  scrapeConfigSelector:
    matchLabels:
      prometheus: system-monitoring-prometheus
```

## Authentication

Two modes are supported for MLflow basic auth:

1. Inline values:

```yaml
mlflow:
  auth:
    username: my-user
    password: my-password
```

2. Existing secret:

```yaml
mlflow:
  auth:
    existingSecret: mlflow-exporter-auth
    usernameKey: MLFLOW_TRACKING_USERNAME
    passwordKey: MLFLOW_TRACKING_PASSWORD
```

If `existingSecret` is set, the chart does not create a new `Secret`.
