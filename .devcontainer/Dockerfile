FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive

ARG HELM_VERSION=v3.18.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        libffi-dev \
        libssl-dev \
        openssh-client \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        sudo \
    && rm -rf /var/lib/apt/lists/*

RUN arch="$(dpkg --print-architecture)" \
    && case "${arch}" in \
        amd64) helm_arch="amd64" ;; \
        arm64) helm_arch="arm64" ;; \
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-${helm_arch}.tar.gz" -o /tmp/helm.tar.gz \
    && tar -xzf /tmp/helm.tar.gz -C /tmp \
    && install -m 0755 "/tmp/linux-${helm_arch}/helm" /usr/local/bin/helm \
    && rm -rf /tmp/helm.tar.gz "/tmp/linux-${helm_arch}"

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces/mlflow-prometheus-exporter
