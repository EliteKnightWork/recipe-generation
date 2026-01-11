import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from accuracy import AccuracyPipeline, AccuracyConfig

app = FastAPI(
    title="Recipe Generation API",
    description="Generate high-quality recipes with accuracy improvements",
    version="2.0.0"
)

# Configuration
LOCAL_MODEL_PATH = "./models/recipenlg"
ONLINE_MODEL_NAME = "mbien/recipenlg"

# Use local model if available, otherwise download from HuggingFace
USE_LOCAL = os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH)
MODEL_PATH = LOCAL_MODEL_PATH if USE_LOCAL else ONLINE_MODEL_NAME

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Loading model from: {'local' if USE_LOCAL else 'online'} ({MODEL_PATH})")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=USE_LOCAL
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=USE_LOCAL
).to(device)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded successfully!")

# Special tokens for recipenlg model
SPECIAL_TOKENS = {
    "recipe_start": "<RECIPE_START>",
    "recipe_end": "<RECIPE_END>",
    "title_start": "<TITLE_START>",
    "title_end": "<TITLE_END>",
    "ingr_start": "<INGR_START>",
    "ingr_end": "<INGR_END>",
    "instr_start": "<INSTR_START>",
    "instr_end": "<INSTR_END>",
    "input_start": "<INPUT_START>",
    "input_end": "<INPUT_END>",
    "next_ingr": "<NEXT_INGR>",
    "next_instr": "<NEXT_INSTR>",
}

# Accuracy configuration
ACCURACY_CONFIG = AccuracyConfig(
    enable_validation=True,
    enable_scoring=True,
    enable_beam_search=True,
    enable_annealing=True,
    enable_constraints=True,
    enable_quality_checks=True,
    num_candidates=10,
    return_top_n=3,
    min_ingredient_coverage=0.3,
    min_directions=2,
    min_quality_score=0.3,
    num_beams=10,
    num_beam_groups=5,
    diversity_penalty=1.5,
    initial_temperature=0.7,
    max_temperature=1.2,
    temperature_step=0.15,
    quality_threshold=0.6,
)

# Initialize accuracy pipeline with special tokens for recipenlg
accuracy_pipeline = AccuracyPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,
    config=ACCURACY_CONFIG,
    special_tokens=SPECIAL_TOKENS
)

print("Accuracy pipeline initialized!")
print(f"  - Beam search: {ACCURACY_CONFIG.enable_beam_search}")
print(f"  - Candidates: {ACCURACY_CONFIG.num_candidates}")
print(f"  - Return top: {ACCURACY_CONFIG.return_top_n}")


class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    directions: List[str]


@app.post("/generate_recipes", response_model=List[Recipe])
def generate_recipes(items: List[str]):
    """
    Generate high-quality recipes for given ingredients.

    Uses accuracy improvements:
    - Beam search with diversity
    - Temperature annealing
    - Ingredient validation
    - Recipe scoring and re-ranking
    - Constrained generation
    - Quality checks
    """
    recipes = accuracy_pipeline.generate(items)
    return recipes
