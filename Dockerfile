FROM python:3.13-trixie AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached on source-only changes.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Install the application package without re-downloading its pinned deps.
COPY pyproject.toml .
COPY mlflow_exporter/ ./mlflow_exporter/
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps .

FROM python:3.13-slim-trixie AS runtime

ARG VERSION
ENV TZ=UTC \
    HOME=/home/appuser \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system appgroup \
    && useradd --system --create-home --home-dir /home/appuser --gid appgroup appuser

COPY --from=builder /opt/venv /opt/venv

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3); sys.exit(0)"]
CMD ["/opt/venv/bin/python", "-m", "mlflow_exporter.main"]
