"""
Recipe Generation API
FastAPI backend with optimized T5 model inference.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import utility modules
from utils.config import GenerationConfig, GenerationStrategy
from utils.preprocessing import InputPreprocessor
from utils.postprocessing import OutputParser
from utils.scoring import RecipeScorer
from utils.generation import RecipeGenerator, get_preset, GENERATION_PRESETS


# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="Recipe Generation API",
    description="Generate recipes from ingredients using T5 model",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Model Configuration
# ============================================================================

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

# Initialize the recipe generator with default config
generator = RecipeGenerator(
    model=model,
    tokenizer=tokenizer,
    device=device,
    generation_config=get_preset("default"),
    preprocessor=InputPreprocessor(),
    parser=OutputParser(),
    scorer=RecipeScorer()
)


# ============================================================================
# Request/Response Models
# ============================================================================

class RecipeScore(BaseModel):
    """Score breakdown for a recipe."""
    overall_score: float
    completeness_score: float
    ingredient_coverage_score: float
    coherence_score: float
    length_score: float
    details: Dict[str, Any] = {}


class Recipe(BaseModel):
    """Recipe with optional score."""
    title: str
    ingredients: List[str]
    directions: List[str]
    score: Optional[RecipeScore] = None


class GenerationResponse(BaseModel):
    """Response model for recipe generation."""
    success: bool
    recipes: List[Recipe]
    processed_ingredients: List[str]
    original_ingredients: List[str]
    warnings: Optional[List[str]] = None
    preprocessing_details: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": MODEL_PATH,
        "device": str(device),
        "version": "2.0.0"
    }


@app.post("/generate_recipes", response_model=GenerationResponse)
def generate_recipes(items: List[str]):
    """
    Generate recipes from ingredients with full optimization pipeline.

    Pipeline:
    1. Preprocess ingredients (normalize, validate, deduplicate)
    2. Generate recipes with optimized parameters
    3. Parse and validate output
    4. Score recipes for quality
    5. Return best recipes with scores

    Args:
        items: List of ingredient strings

    Returns:
        GenerationResponse with recipes, scores, and metadata
    """
    if not items:
        raise HTTPException(status_code=400, detail="No ingredients provided")

    # Step 1: Preprocess ingredients
    processed_ingredients, validation_results = generator.preprocessor.preprocess(items)

    if not processed_ingredients:
        raise HTTPException(
            status_code=400,
            detail="No valid ingredients after preprocessing"
        )

    # Collect warnings and preprocessing details
    warnings = []
    preprocessing_details = []
    for v in validation_results:
        detail = {
            "original": v.original,
            "normalized": v.normalized,
            "is_valid": v.is_valid,
            "category": v.category,
        }
        if v.warnings:
            detail["warnings"] = v.warnings
            warnings.extend(v.warnings)
        preprocessing_details.append(detail)

    # Step 2-4: Generate, parse, and score recipes
    result = generator.generate(
        ingredients=processed_ingredients,
        config=generator.generation_config,
        return_scores=True,
        select_best=True,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Generation failed")
        )

    # Step 5: Build response with recipes and scores
    recipes_with_scores = []
    recipes_data = result.get("recipes", [])
    scores_data = result.get("scores", [])

    for i, recipe_dict in enumerate(recipes_data):
        recipe = Recipe(
            title=recipe_dict.get("title", ""),
            ingredients=recipe_dict.get("ingredients", []),
            directions=recipe_dict.get("directions", []),
        )

        # Attach score if available
        if i < len(scores_data):
            score_dict = scores_data[i]
            recipe.score = RecipeScore(
                overall_score=score_dict.get("overall_score", 0),
                completeness_score=score_dict.get("completeness_score", 0),
                ingredient_coverage_score=score_dict.get("ingredient_coverage_score", 0),
                coherence_score=score_dict.get("coherence_score", 0),
                length_score=score_dict.get("length_score", 0),
                details=score_dict.get("details", {}),
            )

        recipes_with_scores.append(recipe)

    # Add any generation warnings
    if result.get("warnings"):
        warnings.extend(result["warnings"])

    return GenerationResponse(
        success=True,
        recipes=recipes_with_scores,
        processed_ingredients=processed_ingredients,
        original_ingredients=items,
        warnings=warnings if warnings else None,
        preprocessing_details=preprocessing_details,
    )
