import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

app = FastAPI()

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

# Generation parameters (tuned for best quality)
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.8,
    "num_return_sequences": 2,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.encode(SPECIAL_TOKENS["recipe_end"])[0] if SPECIAL_TOKENS["recipe_end"] in tokenizer.get_vocab() else tokenizer.eos_token_id,
}


def format_input(ingredients: List[str]) -> str:
    """Format ingredients into model input format."""
    ingredients_str = ", ".join(ingredients)
    return f"{SPECIAL_TOKENS['input_start']} {ingredients_str} {SPECIAL_TOKENS['input_end']} {SPECIAL_TOKENS['ingr_start']}"


def generation_function(ingredients: List[str]):
    """Generate recipes from ingredients."""
    prompt = format_input(ingredients)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        **generation_kwargs
    )

    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    return generated_texts

def parse_generated_recipes(recipe_list):
    """Parse generated recipes from model output."""
    parsed_recipes = []

    for recipe_text in recipe_list:
        recipe_obj = {}

        # Extract title
        title_match = re.search(
            rf"{re.escape(SPECIAL_TOKENS['title_start'])}(.+?){re.escape(SPECIAL_TOKENS['title_end'])}",
            recipe_text,
            re.DOTALL
        )
        if title_match:
            recipe_obj["title"] = title_match.group(1).strip()
        else:
            recipe_obj["title"] = "Untitled Recipe"

        # Extract ingredients
        ingr_match = re.search(
            rf"{re.escape(SPECIAL_TOKENS['ingr_start'])}(.+?){re.escape(SPECIAL_TOKENS['ingr_end'])}",
            recipe_text,
            re.DOTALL
        )
        if ingr_match:
            ingr_text = ingr_match.group(1).strip()
            ingredients = re.split(rf"{re.escape(SPECIAL_TOKENS['next_ingr'])}", ingr_text)
            recipe_obj["ingredients"] = [ing.strip() for ing in ingredients if ing.strip()]
        else:
            recipe_obj["ingredients"] = []

        # Extract instructions/directions
        instr_match = re.search(
            rf"{re.escape(SPECIAL_TOKENS['instr_start'])}(.+?)(?:{re.escape(SPECIAL_TOKENS['instr_end'])}|{re.escape(SPECIAL_TOKENS['recipe_end'])}|$)",
            recipe_text,
            re.DOTALL
        )
        if instr_match:
            instr_text = instr_match.group(1).strip()
            directions = re.split(rf"{re.escape(SPECIAL_TOKENS['next_instr'])}", instr_text)
            recipe_obj["directions"] = [d.strip() for d in directions if d.strip()]
        else:
            recipe_obj["directions"] = []

        # Only add if we have some content
        if recipe_obj["ingredients"] or recipe_obj["directions"]:
            parsed_recipes.append(recipe_obj)

    return parsed_recipes

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    directions: List[str]

@app.post("/generate_recipes", response_model=List[Recipe])
def generate_recipes(items: List[str]):
    generated = generation_function(items)
    parsed_recipes = parse_generated_recipes(generated)
    return parsed_recipes
