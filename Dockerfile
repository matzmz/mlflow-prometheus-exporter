FROM python:3.11-trixie AS builder

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

FROM python:3.11-slim-trixie AS runtime

ARG VERSION
ENV TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

EXPOSE 8000
CMD ["/opt/venv/bin/mlflow-exporter"]
