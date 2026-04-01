#!/bin/bash

PORT="${PORT:-8080}"

echo "============================================"
echo " FastCosyVoice TTS Cloud Run Server"
echo "============================================"
echo " MODEL_ROOT: ${MODEL_ROOT:-not set}"
echo " Port:       ${PORT}"
echo "============================================"

# Fix /dev/shm for multiprocessing
if [ -d /mnt/shm ]; then
    echo "Binding /mnt/shm over /dev/shm..."
    mount --bind /mnt/shm /dev/shm 2>/dev/null && \
        echo "OK: /dev/shm backed by in-memory volume ($(df -h /dev/shm | awk 'NR==2{print $2}'))" || true
fi
echo "/dev/shm size: $(df -h /dev/shm | awk 'NR==2{print $2}')"

# Verify MODEL_ROOT if set
if [ -n "${MODEL_ROOT}" ]; then
    echo "--- MODEL_ROOT contents ---"
    ls -la "${MODEL_ROOT}/" 2>&1 | head -10
    echo "config.yaml: $([ -f "${MODEL_ROOT}/config.yaml" ] && echo OK || echo MISSING)"
fi

# Start FastAPI server (model loading happens in background thread)
exec python3 /workspace/FastCosyVoice/cloud_run/fastapi_tts_server.py \
    --host 0.0.0.0 \
    --port "${PORT}"
