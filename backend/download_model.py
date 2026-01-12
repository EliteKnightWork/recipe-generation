"""
Script to download T5 Recipe Generation model for offline use.
Run this script once to download all model files.
"""

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "flax-community/t5-recipe-generation"
SAVE_PATH = "./models/t5-recipe-generation"

def download_model():
    print(f"Downloading model: {MODEL_NAME}")
    print("This may take a few minutes depending on your internet connection...")

    # Create directory if not exists
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Download tokenizer
    print("\n[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Tokenizer saved successfully!")

    # Download model
    print("\n[2/2] Downloading model weights...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_PATH)
    print("Model saved successfully!")

    # List downloaded files
    print(f"\nDownloaded files in {SAVE_PATH}:")
    for file in os.listdir(SAVE_PATH):
        file_path = os.path.join(SAVE_PATH, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.2f} MB)")

    total_size = sum(
        os.path.getsize(os.path.join(SAVE_PATH, f))
        for f in os.listdir(SAVE_PATH)
    ) / (1024 * 1024)
    print(f"\nTotal size: {total_size:.2f} MB")
    print(f"\nModel ready for offline use at: {SAVE_PATH}")

if __name__ == "__main__":
    download_model()
