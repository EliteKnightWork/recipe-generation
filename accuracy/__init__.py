"""
Accuracy improvement module for recipe generation.

This module provides:
- Ingredient validation against a food database
- Recipe scoring and re-ranking
- Constrained decoding
- Post-processing quality checks
- Temperature annealing
- Beam search with diversity
- Coherence scoring

Usage:
    from accuracy import AccuracyPipeline

    pipeline = AccuracyPipeline(model, tokenizer, device, config)
    recipes = pipeline.generate(["chicken", "garlic", "lemon"])
"""

from typing import List, Dict, Optional
import torch

from .ingredients_db import (
    get_valid_ingredients,
    get_cooking_units,
    get_cooking_verbs,
    normalize_ingredient,
    VALID_INGREDIENTS,
    COOKING_VERBS,
)
from .validator import (
    IngredientValidator,
    QualityChecker,
    validate_recipe_constraints,
)
from .scorer import (
    RecipeScorer,
    CoherenceScorer,
)
from .generator import (
    GenerationConfig,
    EnhancedGenerator,
    TemperatureAnnealing,
    ConstrainedGenerator,
    parse_recipenlg_recipes,
)


class AccuracyConfig:
    """Configuration for the accuracy pipeline."""

    def __init__(
        self,
        # Feature toggles
        enable_validation: bool = True,
        enable_scoring: bool = True,
        enable_beam_search: bool = True,
        enable_annealing: bool = True,
        enable_constraints: bool = True,
        enable_quality_checks: bool = True,
        # Generation settings
        num_candidates: int = 10,
        return_top_n: int = 2,
        # Constraint settings
        min_ingredient_coverage: float = 0.5,
        min_directions: int = 3,
        # Quality thresholds
        min_quality_score: float = 0.5,
        # Beam search settings
        num_beams: int = 10,
        num_beam_groups: int = 5,
        diversity_penalty: float = 0.5,
        # Temperature annealing settings
        initial_temperature: float = 0.7,
        max_temperature: float = 1.2,
        temperature_step: float = 0.15,
        quality_threshold: float = 0.6,
    ):
        self.enable_validation = enable_validation
        self.enable_scoring = enable_scoring
        self.enable_beam_search = enable_beam_search
        self.enable_annealing = enable_annealing
        self.enable_constraints = enable_constraints
        self.enable_quality_checks = enable_quality_checks
        self.num_candidates = num_candidates
        self.return_top_n = return_top_n
        self.min_ingredient_coverage = min_ingredient_coverage
        self.min_directions = min_directions
        self.min_quality_score = min_quality_score
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.initial_temperature = initial_temperature
        self.max_temperature = max_temperature
        self.temperature_step = temperature_step
        self.quality_threshold = quality_threshold


class AccuracyPipeline:
    """
    Main pipeline for generating high-accuracy recipes.

    Combines all accuracy features:
    - Beam search with diversity for coherent generation
    - Temperature annealing for quality optimization
    - Ingredient validation
    - Recipe scoring and re-ranking
    - Constrained generation
    - Quality checks and cleaning

    Designed for RecipeNLG (CausalLM/GPT-2) model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        config: AccuracyConfig = None,
        special_tokens: Dict[str, str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or AccuracyConfig()
        self.special_tokens = special_tokens

        # Initialize components
        self.validator = IngredientValidator()
        self.quality_checker = QualityChecker()
        self.scorer = RecipeScorer()
        self.coherence_scorer = CoherenceScorer()

        # Create parser for RecipeNLG
        self.parser = lambda recipes: parse_recipenlg_recipes(recipes, special_tokens)

        # Initialize generator with config
        gen_config = GenerationConfig(
            num_candidates=self.config.num_candidates,
            use_beam_search=self.config.enable_beam_search,
            num_beams=self.config.num_beams,
            num_beam_groups=self.config.num_beam_groups,
            diversity_penalty=self.config.diversity_penalty,
            enable_annealing=self.config.enable_annealing,
            initial_temp=self.config.initial_temperature,
            max_temp=self.config.max_temperature,
            temp_step=self.config.temperature_step,
            quality_threshold=self.config.quality_threshold,
        )
        self.generator = EnhancedGenerator(
            model, tokenizer, device, gen_config,
            special_tokens=special_tokens
        )

        # Initialize constrained generator
        self.constrained_generator = ConstrainedGenerator(
            generator=self.generator,
            validator=self.validator,
            parser=self.parser,
            min_coverage=self.config.min_ingredient_coverage,
            min_directions=self.config.min_directions,
        )

        # Initialize temperature annealing
        self.annealing = TemperatureAnnealing(
            generator=self.generator,
            scorer=self.scorer,
            parser=self.parser,
            initial_temp=self.config.initial_temperature,
            max_temp=self.config.max_temperature,
            temp_step=self.config.temperature_step,
            quality_threshold=self.config.quality_threshold,
        )

    def generate(self, input_items: List[str]) -> List[Dict]:
        """
        Generate high-quality recipes for given ingredients.

        Args:
            input_items: List of ingredient strings

        Returns:
            List of recipe dictionaries with title, ingredients, directions
        """
        # Normalize input items
        normalized_inputs = [item.strip().lower() for item in input_items if item.strip()]

        # Step 1: Generate candidates
        if self.config.enable_constraints:
            recipes = self.constrained_generator.generate_with_constraints(normalized_inputs)
        elif self.config.enable_annealing and not self.config.enable_beam_search:
            recipes, _ = self.annealing.generate_with_annealing(normalized_inputs)
        else:
            raw_recipes = self.generator.generate(normalized_inputs)
            recipes = parse_generated_recipes(raw_recipes)

        # Step 2: Quality checks and cleaning
        if self.config.enable_quality_checks:
            cleaned_recipes = []
            for recipe in recipes:
                cleaned = self.quality_checker.clean_recipe(recipe)
                is_valid, issues = self.quality_checker.check_recipe(cleaned)
                # Include recipe even if not perfect, scoring will rank it
                cleaned_recipes.append(cleaned)
            recipes = cleaned_recipes

        # Step 3: Validation
        if self.config.enable_validation:
            validated_recipes = []
            for recipe in recipes:
                # Validate ingredients
                ingredients = recipe.get("ingredients", [])
                valid_ings, invalid_ings = self.validator.validate_ingredients_list(ingredients)

                # Check coverage
                coverage = self.validator.check_input_coverage(normalized_inputs, ingredients)

                # Add validation info to recipe (for debugging)
                recipe["_validation"] = {
                    "valid_ingredients": len(valid_ings),
                    "invalid_ingredients": len(invalid_ings),
                    "input_coverage": coverage
                }
                validated_recipes.append(recipe)
            recipes = validated_recipes

        # Step 4: Score and rank
        if self.config.enable_scoring and recipes:
            ranked = self.scorer.rank_recipes(
                recipes,
                normalized_inputs,
                top_n=self.config.return_top_n
            )
            # Extract just the recipes (not scores)
            recipes = [r[0] for r in ranked]

            # Add scores to recipes (for debugging)
            for i, (recipe, score, details) in enumerate(ranked):
                if i < len(recipes):
                    recipes[i]["_score"] = score
                    recipes[i]["_score_details"] = details

        # Step 5: Final filtering by quality threshold
        if self.config.enable_scoring:
            recipes = [
                r for r in recipes
                if r.get("_score", 0) >= self.config.min_quality_score
            ]

        # Ensure we return at least something
        if not recipes and len(input_items) > 0:
            # Fallback: generate without constraints
            raw_recipes = self.generator.generate(normalized_inputs)
            recipes = parse_generated_recipes(raw_recipes)
            if recipes:
                recipes = [self.quality_checker.clean_recipe(r) for r in recipes[:self.config.return_top_n]]

        # Remove internal fields before returning
        for recipe in recipes:
            recipe.pop("_validation", None)
            recipe.pop("_score", None)
            recipe.pop("_score_details", None)

        return recipes[:self.config.return_top_n]

    def generate_with_details(self, input_items: List[str]) -> Dict:
        """
        Generate recipes with detailed scoring and validation info.

        Args:
            input_items: List of ingredient strings

        Returns:
            Dictionary with recipes and detailed analysis
        """
        normalized_inputs = [item.strip().lower() for item in input_items if item.strip()]

        # Generate candidates
        raw_recipes = self.generator.generate(normalized_inputs)
        all_recipes = parse_generated_recipes(raw_recipes)

        # Clean all recipes
        all_recipes = [self.quality_checker.clean_recipe(r) for r in all_recipes]

        # Score all recipes
        scored_recipes = []
        for recipe in all_recipes:
            score, details = self.scorer.score_recipe(recipe, normalized_inputs)
            coherence = self.coherence_scorer.score_coherence(recipe)
            is_valid, issues = self.quality_checker.check_recipe(recipe)
            coverage = self.validator.check_input_coverage(
                normalized_inputs, recipe.get("ingredients", [])
            )

            scored_recipes.append({
                "recipe": recipe,
                "total_score": score,
                "score_breakdown": details,
                "coherence_score": coherence,
                "quality_valid": is_valid,
                "quality_issues": issues,
                "input_coverage": coverage
            })

        # Sort by score
        scored_recipes.sort(key=lambda x: x["total_score"], reverse=True)

        return {
            "top_recipes": [sr["recipe"] for sr in scored_recipes[:self.config.return_top_n]],
            "all_candidates": scored_recipes,
            "input_items": normalized_inputs,
            "config": {
                "num_candidates": self.config.num_candidates,
                "beam_search": self.config.enable_beam_search,
                "min_coverage": self.config.min_ingredient_coverage
            }
        }


__all__ = [
    "AccuracyPipeline",
    "AccuracyConfig",
    "IngredientValidator",
    "QualityChecker",
    "RecipeScorer",
    "CoherenceScorer",
    "EnhancedGenerator",
    "GenerationConfig",
    "TemperatureAnnealing",
    "ConstrainedGenerator",
    "parse_generated_recipes",
    "validate_recipe_constraints",
    "get_valid_ingredients",
    "get_cooking_units",
    "get_cooking_verbs",
    "normalize_ingredient",
]
