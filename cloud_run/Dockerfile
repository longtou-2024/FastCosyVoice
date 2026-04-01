ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.4.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG APP_DIR=/workspace/FastCosyVoice
WORKDIR ${APP_DIR}

ENV DEBIAN_FRONTEND=noninteractive

# ── Layer 1: System dependencies ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        git curl sox libsox-fmt-all ffmpeg \
        apt-transport-https ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Layer 2: Python dependencies ──
COPY cloud_run/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /tmp/requirements.txt

# ── Layer 3a: Rarely changed dependencies ──
COPY third_party/ ${APP_DIR}/third_party/

# ── Layer 3b: Project source ──
COPY cosyvoice/ ${APP_DIR}/cosyvoice/
COPY fastcosyvoice/ ${APP_DIR}/fastcosyvoice/

# ── Layer 4: Cloud Run server files ──
COPY cloud_run/ ${APP_DIR}/cloud_run/
COPY cloud_run/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
