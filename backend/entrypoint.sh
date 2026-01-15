#!/bin/bash
set -e

VENV_PATH="/app/venv"
PORT="${PORT:-8000}"

# Check if venv has packages installed (check for uvicorn)
if [ ! -f "$VENV_PATH/bin/uvicorn" ]; then
    echo "Installing dependencies in virtual environment..."
    python -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Using existing shared virtual environment..."
    source $VENV_PATH/bin/activate
fi

MODEL_PATH="/app/models/t5-recipe-generation"

# Check if model exists locally
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
    echo "Model found at $MODEL_PATH, skipping download..."
else
    echo "Model not found. Downloading..."
    python download_model.py
fi

echo "Starting FastAPI service on port $PORT..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT
