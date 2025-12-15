#!/bin/bash
set -e

MODEL_PATH="/app/models/t5-recipe-generation"

# Check if model exists locally
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
    echo "Model found at $MODEL_PATH, skipping download..."
else
    echo "Model not found. Downloading..."
    python download_model.py
fi

echo "Starting FastAPI service..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
