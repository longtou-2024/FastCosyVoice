#!/usr/bin/env bash
set -euo pipefail

TENSORRT_VERSION="10.0.1"
NVIDIA_INDEX="https://pypi.nvidia.com"

echo "==> Running uv sync..."
uv sync

echo "==> Installing TensorRT packages (not compatible with uv build isolation)..."
uv pip install \
    "tensorrt-cu12==${TENSORRT_VERSION}" \
    "tensorrt-cu12-bindings==${TENSORRT_VERSION}" \
    "tensorrt-cu12-libs==${TENSORRT_VERSION}" \
    --extra-index-url "${NVIDIA_INDEX}"

echo "==> Installation complete."
