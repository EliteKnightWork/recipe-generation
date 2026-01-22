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
LANGUAGE_MODEL_PATH="/app/models/language-model"

# Check if T5 model exists locally
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
    echo "T5 model found at $MODEL_PATH, skipping download..."
else
    echo "T5 model not found. Downloading..."
    python download_model.py
fi

# Download language model if language enhancement is enabled
if [ "${USE_LANGUAGE_ENHANCEMENT:-false}" = "true" ]; then
    if [ -d "$LANGUAGE_MODEL_PATH" ] && [ "$(ls -A $LANGUAGE_MODEL_PATH 2>/dev/null)" ]; then
        echo "Language model found at $LANGUAGE_MODEL_PATH, skipping download..."
    else
        echo "Language model not found. Downloading for language enhancement..."
        python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_NAME = os.environ.get('LLAMA_MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
SAVE_PATH = '$LANGUAGE_MODEL_PATH'

print(f'Downloading {MODEL_NAME}...')
os.makedirs(SAVE_PATH, exist_ok=True)

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_PATH)

print('Downloading model weights...')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.save_pretrained(SAVE_PATH)

print(f'Model saved to {SAVE_PATH}')
"
    fi
fi

echo "Starting FastAPI service on port $PORT..."
exec uvicorn main:app --host 0.0.0.0 --port $PORT
