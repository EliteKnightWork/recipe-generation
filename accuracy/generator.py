"""
Enhanced recipe generation with beam search, temperature annealing, and constraints.
Designed for RecipeNLG (CausalLM/GPT-2) model.
"""

import re
from typing import List, Dict, Tuple
import torch


class GenerationConfig:
    """Configuration for recipe generation."""

    def __init__(
        self,
        max_length: int = 512,
        min_length: int = 64,
        num_candidates: int = 10,
        use_beam_search: bool = True,
        num_beams: int = 10,
        num_beam_groups: int = 5,
        diversity_penalty: float = 0.5,
        no_repeat_ngram_size: int = 3,
        # Sampling parameters (used when beam search disabled or as fallback)
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        # Temperature annealing
        enable_annealing: bool = True,
        initial_temp: float = 0.7,
        max_temp: float = 1.2,
        temp_step: float = 0.15,
        quality_threshold: float = 0.6,
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.num_candidates = num_candidates
        self.use_beam_search = use_beam_search
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.enable_annealing = enable_annealing
        self.initial_temp = initial_temp
        self.max_temp = max_temp
        self.temp_step = temp_step
        self.quality_threshold = quality_threshold


class EnhancedGenerator:
    """Enhanced recipe generator with beam search and temperature annealing for RecipeNLG."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        config: GenerationConfig = None,
        special_tokens: Dict[str, str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or GenerationConfig()
        self.special_tokens = special_tokens or {}

    def generate(self, input_items: List[str]) -> List[str]:
        """
        Generate recipes using beam search (default) or sampling.

        Args:
            input_items: List of ingredient strings

        Returns:
            List of generated recipe strings
        """
        if self.config.use_beam_search:
            return self._generate_beam_search(input_items)
        else:
            return self._generate_sampling(input_items)

    def _format_input(self, input_items: List[str]) -> str:
        """Format input for RecipeNLG model."""
        ingredients_str = ", ".join(input_items)
        return f"{self.special_tokens['input_start']} {ingredients_str} {self.special_tokens['input_end']} {self.special_tokens['ingr_start']}"

    def _generate_beam_search(self, input_items: List[str]) -> List[str]:
        """Generate using diverse beam search."""
        inputs = self._prepare_inputs(input_items)

        generation_kwargs = {
            "max_length": self.config.max_length,
            "min_length": self.config.min_length,
            "num_beams": self.config.num_beams,
            "num_beam_groups": self.config.num_beam_groups,
            "diversity_penalty": self.config.diversity_penalty,
            "num_return_sequences": self.config.num_candidates,
            "early_stopping": True,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add eos_token_id
        if "recipe_end" in self.special_tokens:
            recipe_end = self.special_tokens["recipe_end"]
            if recipe_end in self.tokenizer.get_vocab():
                generation_kwargs["eos_token_id"] = self.tokenizer.encode(recipe_end)[0]

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )

        return self._postprocess(output_ids)

    def _generate_sampling(self, input_items: List[str], temperature: float = None) -> List[str]:
        """Generate using sampling with optional temperature."""
        inputs = self._prepare_inputs(input_items)
        temp = temperature or self.config.temperature

        generation_kwargs = {
            "max_length": self.config.max_length,
            "min_length": self.config.min_length,
            "do_sample": True,
            "temperature": temp,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "num_return_sequences": self.config.num_candidates,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add eos_token_id
        if "recipe_end" in self.special_tokens:
            recipe_end = self.special_tokens["recipe_end"]
            if recipe_end in self.tokenizer.get_vocab():
                generation_kwargs["eos_token_id"] = self.tokenizer.encode(recipe_end)[0]

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )

        return self._postprocess(output_ids)

    def _prepare_inputs(self, input_items: List[str]) -> Dict:
        """Prepare tokenized inputs."""
        _inputs = input_items if isinstance(input_items, list) else [input_items]
        text_input = self._format_input(_inputs)

        inputs = self.tokenizer(
            text_input,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device)
        }

    def _postprocess(self, output_ids: torch.Tensor) -> List[str]:
        """Post-process generated token IDs to strings."""
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        # Remove <|endoftext|> tokens (GPT-2 EOS token)
        decoded = [text.replace("<|endoftext|>", "") for text in decoded]
        return decoded


class TemperatureAnnealing:
    """Temperature annealing for improved generation quality."""

    def __init__(
        self,
        generator: EnhancedGenerator,
        scorer,  # RecipeScorer instance
        parser,  # Recipe parser function
        initial_temp: float = 0.7,
        max_temp: float = 1.2,
        temp_step: float = 0.15,
        quality_threshold: float = 0.6,
        max_attempts: int = 5
    ):
        self.generator = generator
        self.scorer = scorer
        self.parser = parser
        self.initial_temp = initial_temp
        self.max_temp = max_temp
        self.temp_step = temp_step
        self.quality_threshold = quality_threshold
        self.max_attempts = max_attempts

    def generate_with_annealing(
        self,
        input_items: List[str]
    ) -> Tuple[List[Dict], float]:
        """
        Generate recipes with temperature annealing.
        Increases temperature if quality is below threshold.

        Args:
            input_items: List of input ingredients

        Returns:
            Tuple of (best_recipes, best_score)
        """
        temp = self.initial_temp
        best_recipes = []
        best_score = 0.0
        attempts = 0

        while temp <= self.max_temp and attempts < self.max_attempts:
            # Generate with current temperature
            raw_recipes = self.generator._generate_sampling(input_items, temperature=temp)
            parsed_recipes = self.parser(raw_recipes)

            # Score recipes
            for recipe in parsed_recipes:
                if recipe.get("title") and recipe.get("ingredients") and recipe.get("directions"):
                    score, _ = self.scorer.score_recipe(recipe, input_items)
                    if score > best_score:
                        best_score = score
                        best_recipes = parsed_recipes

            # Check if quality threshold met
            if best_score >= self.quality_threshold:
                break

            # Increase temperature for more diversity
            temp += self.temp_step
            attempts += 1

        return best_recipes, best_score


class ConstrainedGenerator:
    """Generator with constraint checking and regeneration."""

    def __init__(
        self,
        generator: EnhancedGenerator,
        validator,  # IngredientValidator instance
        parser,  # Recipe parser function
        min_coverage: float = 0.5,
        min_directions: int = 3,
        max_regenerations: int = 3
    ):
        self.generator = generator
        self.validator = validator
        self.parser = parser
        self.min_coverage = min_coverage
        self.min_directions = min_directions
        self.max_regenerations = max_regenerations

    def generate_with_constraints(
        self,
        input_items: List[str]
    ) -> List[Dict]:
        """
        Generate recipes that meet constraints.
        Regenerates with different parameters if constraints not met.

        Args:
            input_items: List of input ingredients

        Returns:
            List of recipes meeting constraints
        """
        all_valid_recipes = []
        attempts = 0

        while attempts < self.max_regenerations:
            # Generate recipes
            raw_recipes = self.generator.generate(input_items)
            parsed_recipes = self.parser(raw_recipes)

            # Check each recipe against constraints
            for recipe in parsed_recipes:
                if self._check_constraints(recipe, input_items):
                    all_valid_recipes.append(recipe)

            # If we have enough valid recipes, return
            if len(all_valid_recipes) >= 2:
                return all_valid_recipes

            # Try with sampling instead of beam search for more diversity
            if attempts == 0 and self.generator.config.use_beam_search:
                self.generator.config.use_beam_search = False
            elif attempts == 1:
                # Increase temperature
                self.generator.config.temperature = min(
                    self.generator.config.temperature + 0.2,
                    1.2
                )

            attempts += 1

        # Return what we have, even if incomplete
        return all_valid_recipes if all_valid_recipes else parsed_recipes

    def _check_constraints(self, recipe: Dict, input_items: List[str]) -> bool:
        """Check if recipe meets all constraints."""
        # Must have title
        if not recipe.get("title", "").strip():
            return False

        # Must have minimum directions
        directions = recipe.get("directions", [])
        if len([d for d in directions if d and d.strip()]) < self.min_directions:
            return False

        # Must meet ingredient coverage
        ingredients = recipe.get("ingredients", [])
        coverage = self.validator.check_input_coverage(input_items, ingredients)
        if coverage < self.min_coverage:
            return False

        return True


def parse_recipenlg_recipes(recipe_list: List[str], special_tokens: Dict[str, str]) -> List[Dict]:
    """
    Parse recipes generated by RecipeNLG model.

    Args:
        recipe_list: List of raw recipe strings
        special_tokens: Dictionary of special tokens

    Returns:
        List of parsed recipe dictionaries
    """
    parsed_recipes = []

    for recipe_text in recipe_list:
        recipe_obj = {}

        # Extract title
        title_match = re.search(
            rf"{re.escape(special_tokens['title_start'])}(.+?){re.escape(special_tokens['title_end'])}",
            recipe_text,
            re.DOTALL
        )
        if title_match:
            recipe_obj["title"] = title_match.group(1).strip()
        else:
            recipe_obj["title"] = "Untitled Recipe"

        # Extract ingredients
        ingr_match = re.search(
            rf"{re.escape(special_tokens['ingr_start'])}(.+?){re.escape(special_tokens['ingr_end'])}",
            recipe_text,
            re.DOTALL
        )
        if ingr_match:
            ingr_text = ingr_match.group(1).strip()
            if "next_ingr" in special_tokens:
                ingredients = re.split(rf"{re.escape(special_tokens['next_ingr'])}", ingr_text)
            else:
                ingredients = ingr_text.split('\n')
            recipe_obj["ingredients"] = [ing.strip() for ing in ingredients if ing.strip()]
        else:
            recipe_obj["ingredients"] = []

        # Extract instructions/directions
        instr_end_pattern = special_tokens.get('instr_end', '')
        recipe_end_pattern = special_tokens.get('recipe_end', '')

        end_pattern = f"(?:{re.escape(instr_end_pattern)}|{re.escape(recipe_end_pattern)}|$)" if instr_end_pattern or recipe_end_pattern else "$"

        instr_match = re.search(
            rf"{re.escape(special_tokens['instr_start'])}(.+?){end_pattern}",
            recipe_text,
            re.DOTALL
        )
        if instr_match:
            instr_text = instr_match.group(1).strip()
            if "next_instr" in special_tokens:
                directions = re.split(rf"{re.escape(special_tokens['next_instr'])}", instr_text)
            else:
                directions = instr_text.split('\n')
            recipe_obj["directions"] = [d.strip() for d in directions if d.strip()]
        else:
            recipe_obj["directions"] = []

        # Only add if we have some content
        if recipe_obj["ingredients"] or recipe_obj["directions"]:
            parsed_recipes.append(recipe_obj)

    return parsed_recipes
