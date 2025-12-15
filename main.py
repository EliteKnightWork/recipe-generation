import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

app = FastAPI()

# Configuration
LOCAL_MODEL_PATH = "./models/t5-recipe-generation"
ONLINE_MODEL_NAME = "flax-community/t5-recipe-generation"

# Use local model if available, otherwise download from HuggingFace
USE_LOCAL = os.path.exists(LOCAL_MODEL_PATH)
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
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    local_files_only=USE_LOCAL
).to(device)

print(f"Model loaded successfully!")

prefix = "items: "
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95,
    "num_return_sequences": 2
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")

    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)

        for k, v in tokens_map.items():
            text = text.replace(k, v)

        new_texts.append(text)
    return new_texts

def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move tensors to device (GPU if available)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(output_ids, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe

def parse_generated_recipes(recipe_list):
    parsed_recipes = []
    for recipe in recipe_list:
        recipe_obj = {}
        title_match = re.search(r" title: (.+?)\n", recipe)
        ingredients_match = re.search(r" ingredients: (.+?)\n directions:", recipe)
        directions_match = re.search(r" directions: (.+)", recipe)

        if title_match:
            recipe_obj["title"] = title_match.group(1).strip()
        if ingredients_match:
            ingredients = ingredients_match.group(1).strip().split('--')
            recipe_obj["ingredients"] = [ingredient.strip() for ingredient in ingredients]
        if directions_match:
            directions = directions_match.group(1).strip().split('--')
            recipe_obj["directions"] = [direction.strip() for direction in directions]

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
