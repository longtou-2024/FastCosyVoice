#!/usr/bin/env bash

exec uv run uvicorn run_fastapi:app --host 0.0.0.0 --port 8000
