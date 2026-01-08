#!/bin/bash
set -e

MODEL_PATH="/app/models/recipenlg"

# Check if model exists locally
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
    echo "Recipe model found at $MODEL_PATH, skipping download..."
else
    echo "Recipe model not found. Downloading mbien/recipenlg..."
    python download_model.py
fi

echo "Starting FastAPI service..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
