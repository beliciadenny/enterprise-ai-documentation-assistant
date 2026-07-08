#!/usr/bin/env bash
# start.sh
# --------
# Starts the FastAPI backend and Streamlit frontend together inside the
# same container. Ollama is expected to run as a separate service/host
# (see README.md and docker-compose.yml).

set -e

echo "Starting FastAPI backend on port ${API_PORT:-8000}..."
uvicorn backend.api:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" &
BACKEND_PID=$!

# Give the backend a moment to come up before the frontend starts polling it.
sleep 3

echo "Starting Streamlit frontend on port 8501..."
streamlit run frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true &
FRONTEND_PID=$!

# If either process exits, stop the container so orchestration tools notice.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
