"""
Generation module for recipe generation.
Handles model inference with optimized parameters and multi-generation strategies.
"""

import torch
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from .config import GenerationConfig, GenerationStrategy
from .preprocessing import InputPreprocessor
from .postprocessing import OutputParser, ParsedRecipe
from .scoring import RecipeScorer, RecipeScore


class RecipeGenerator:
    """
    High-level recipe generator with optimization and scoring.
    Wraps the T5 model with preprocessing, generation, and postprocessing.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        generation_config: Optional[GenerationConfig] = None,
        preprocessor: Optional[InputPreprocessor] = None,
        parser: Optional[OutputParser] = None,
        scorer: Optional[RecipeScorer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize components with defaults if not provided
        self.generation_config = generation_config or GenerationConfig()
        self.preprocessor = preprocessor or InputPreprocessor()
        self.parser = parser or OutputParser()
        self.scorer = scorer or RecipeScorer()

        # Model input configuration
        self.prefix = "items: "
        self.max_input_length = 256

        # Get special tokens for postprocessing
        self.special_tokens = tokenizer.all_special_tokens

    def generate(
        self,
        ingredients: List[str],
        config: Optional[GenerationConfig] = None,
        return_scores: bool = True,
        select_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate recipes from ingredients.

        Args:
            ingredients: List of ingredient strings
            config: Optional override generation config
            return_scores: Whether to include scores in response
            select_best: Whether to select best recipe when generating multiple

        Returns:
            Dict with recipes, scores, and metadata
        """
        config = config or self.generation_config

        # Step 1: Preprocess ingredients
        processed_ingredients, validation_results = self.preprocessor.preprocess(ingredients)

        if not processed_ingredients:
            return {
                "success": False,
                "error": "No valid ingredients after preprocessing",
                "validation": [v.__dict__ for v in validation_results],
                "recipes": [],
            }

        # Step 2: Generate recipes (possibly multiple times for best selection)
        all_recipes = []
        all_scores = []

        num_generations = config.num_generations if select_best else 1

        for _ in range(num_generations):
            # Generate
            generated_texts = self._generate_raw(processed_ingredients, config)

            # Parse
            parsed_recipes = self.parser.parse_batch(generated_texts, self.special_tokens)

            # Filter valid recipes
            valid_recipes = [r for r in parsed_recipes if r.parse_success]

            # Score
            if return_scores or select_best:
                for recipe in valid_recipes:
                    score = self.scorer.score_recipe(
                        recipe.to_dict(),
                        processed_ingredients
                    )
                    all_recipes.append(recipe)
                    all_scores.append(score)
            else:
                all_recipes.extend(valid_recipes)

        # Step 3: Select best or return all
        if select_best and all_scores:
            # Sort by score and take top N
            sorted_pairs = sorted(
                zip(all_recipes, all_scores),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            num_to_return = config.num_return_sequences
            selected_recipes = [r for r, _ in sorted_pairs[:num_to_return]]
            selected_scores = [s for _, s in sorted_pairs[:num_to_return]]
        else:
            selected_recipes = all_recipes[:config.num_return_sequences]
            selected_scores = all_scores[:config.num_return_sequences] if all_scores else []

        # Step 4: Build response
        response = {
            "success": True,
            "recipes": [r.to_dict() for r in selected_recipes],
            "processed_ingredients": processed_ingredients,
            "original_ingredients": ingredients,
        }

        if return_scores and selected_scores:
            response["scores"] = [s.to_dict() for s in selected_scores]

        if validation_results:
            warnings = []
            for v in validation_results:
                if v.warnings:
                    warnings.extend(v.warnings)
            if warnings:
                response["warnings"] = warnings

        return response

    def generate_simple(self, ingredients: List[str]) -> List[Dict]:
        """
        Simple generation interface for backward compatibility.
        Returns just the list of recipe dicts.
        """
        result = self.generate(
            ingredients,
            return_scores=False,
            select_best=False,
        )
        return result.get("recipes", [])

    def _generate_raw(
        self,
        ingredients: List[str],
        config: GenerationConfig
    ) -> List[str]:
        """
        Raw generation from model.

        Args:
            ingredients: Preprocessed ingredients
            config: Generation configuration

        Returns:
            List of generated text strings
        """
        # Format input
        input_text = self.prefix + ", ".join(ingredients)

        # Tokenize
        inputs = self.tokenizer(
            [input_text],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Generate
        generation_kwargs = config.to_generation_kwargs()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )

        # Decode
        generated_texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False
        )

        return generated_texts

    def update_config(self, **kwargs) -> None:
        """Update generation config parameters."""
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)

    def set_strategy(self, strategy: str) -> None:
        """Set generation strategy by name."""
        strategy_map = {
            "sampling": GenerationStrategy.SAMPLING,
            "beam_search": GenerationStrategy.BEAM_SEARCH,
            "nucleus": GenerationStrategy.NUCLEUS,
            "contrastive": GenerationStrategy.CONTRASTIVE,
        }
        if strategy.lower() in strategy_map:
            self.generation_config.strategy = strategy_map[strategy.lower()]

    def get_config(self) -> Dict:
        """Get current generation config as dict."""
        return {
            "strategy": self.generation_config.strategy.value,
            "temperature": self.generation_config.temperature,
            "top_k": self.generation_config.top_k,
            "top_p": self.generation_config.top_p,
            "num_beams": self.generation_config.num_beams,
            "repetition_penalty": self.generation_config.repetition_penalty,
            "max_length": self.generation_config.max_length,
            "min_length": self.generation_config.min_length,
            "num_return_sequences": self.generation_config.num_return_sequences,
        }


# Preset configurations for different use cases
GENERATION_PRESETS = {
    "default": GenerationConfig(
        strategy=GenerationStrategy.NUCLEUS,
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
    ),
    "creative": GenerationConfig(
        strategy=GenerationStrategy.SAMPLING,
        temperature=1.0,
        top_k=100,
        top_p=0.95,
        repetition_penalty=1.1,
    ),
    "focused": GenerationConfig(
        strategy=GenerationStrategy.NUCLEUS,
        temperature=0.6,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.3,
    ),
    "deterministic": GenerationConfig(
        strategy=GenerationStrategy.BEAM_SEARCH,
        num_beams=4,
        length_penalty=1.0,
        repetition_penalty=1.2,
    ),
    "best_quality": GenerationConfig(
        strategy=GenerationStrategy.NUCLEUS,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        num_generations=3,  # Generate 3 times, pick best
        num_return_sequences=4,  # Generate 4 per run
    ),
}


def get_preset(name: str) -> GenerationConfig:
    """Get a generation preset by name."""
    return GENERATION_PRESETS.get(name, GENERATION_PRESETS["default"])
