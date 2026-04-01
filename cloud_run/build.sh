#!/bin/bash
# FastCosyVoice TTS Docker image build script
#
# Usage:
#   ./cloud_run/build.sh
#   IMAGE_TAG=v1.0 ./cloud_run/build.sh
#
# ── Build & Deploy ──
#
# 1) Artifact Registry push:
#    gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://asia-northeast3-docker.pkg.dev
#    docker push asia-northeast3-docker.pkg.dev/PROJECT_ID/tts/fastcosyvoice-tts:latest
#
# 2) Cloud Run deployment settings:
#    - Container image: asia-northeast3-docker.pkg.dev/PROJECT_ID/tts/fastcosyvoice-tts:latest
#    - Port: 8080
#    - CPU: 8, Memory: 32Gi
#    - GPU: 1 x nvidia-l4
#    - Max instances: 3, Min instances: 0 or 1
#    - Request timeout: 300s
#    - CPU always allocated
#
# 3) Environment variables:
#    - GCS_MODEL_PATH=gs://bucket/path/to/project  (or MODEL_ROOT=/gcs/...)
#    - PORT=8080

set -e

IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_NAME="${IMAGE_NAME:-cosy_stream_demo}"
PROJECT_ID="dev-ai-project-357507"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building Docker image ==="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

cd "$REPO_ROOT"
docker build \
    -f cloud_run/Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "=== Build complete ==="
echo "Local image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To tag for Artifact Registry:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} asia-northeast3-docker.pkg.dev/${PROJECT_ID}/tts/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://asia-northeast3-docker.pkg.dev"
echo "  docker push asia-northeast3-docker.pkg.dev/${PROJECT_ID}/tts/${IMAGE_NAME}:${IMAGE_TAG}"
