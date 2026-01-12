"""
Configuration classes for recipe generation.
Centralized configuration for generation parameters and scoring.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class GenerationStrategy(Enum):
    """Available generation strategies."""
    SAMPLING = "sampling"  # Random sampling with temperature
    BEAM_SEARCH = "beam_search"  # Deterministic beam search
    NUCLEUS = "nucleus"  # Top-p (nucleus) sampling
    CONTRASTIVE = "contrastive"  # Contrastive search


@dataclass
class GenerationConfig:
    """
    Configuration for text generation parameters.

    Optimized defaults based on recipe generation task.
    """
    # Length constraints
    max_length: int = 512
    min_length: int = 64

    # Generation strategy
    strategy: GenerationStrategy = GenerationStrategy.NUCLEUS

    # Sampling parameters (used when do_sample=True)
    temperature: float = 0.8  # Lower = more focused, Higher = more creative
    top_k: int = 50  # Limits to top K tokens
    top_p: float = 0.92  # Nucleus sampling threshold

    # Beam search parameters (used when do_sample=False)
    num_beams: int = 4
    length_penalty: float = 1.0  # >1 encourages longer, <1 encourages shorter
    early_stopping: bool = True

    # Repetition control
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.2  # Penalize repeated tokens

    # Output control
    num_return_sequences: int = 2

    # Multi-generation for best selection
    num_generations: int = 1  # Generate multiple times and pick best

    def to_generation_kwargs(self) -> dict:
        """Convert config to HuggingFace generation kwargs."""
        kwargs = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": self.num_return_sequences,
        }

        if self.strategy == GenerationStrategy.BEAM_SEARCH:
            kwargs.update({
                "do_sample": False,
                "num_beams": self.num_beams,
                "length_penalty": self.length_penalty,
                "early_stopping": self.early_stopping,
            })
        elif self.strategy == GenerationStrategy.SAMPLING:
            kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_k": self.top_k,
            })
        elif self.strategy == GenerationStrategy.NUCLEUS:
            kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
            })
        elif self.strategy == GenerationStrategy.CONTRASTIVE:
            kwargs.update({
                "do_sample": False,
                "penalty_alpha": 0.6,
                "top_k": 4,
            })

        return kwargs


@dataclass
class ScoringConfig:
    """Configuration for recipe scoring."""
    # Weights for different scoring components
    completeness_weight: float = 0.3
    ingredient_coverage_weight: float = 0.3
    coherence_weight: float = 0.2
    length_weight: float = 0.2

    # Thresholds
    min_ingredients: int = 2
    min_directions: int = 2
    min_title_length: int = 3
    max_title_length: int = 100

    # Ideal ranges for length scoring
    ideal_ingredients_range: tuple = (3, 15)
    ideal_directions_range: tuple = (3, 12)


@dataclass
class PreprocessingConfig:
    """Configuration for input preprocessing."""
    lowercase: bool = True
    strip_whitespace: bool = True
    remove_duplicates: bool = True
    min_ingredient_length: int = 2
    max_ingredient_length: int = 50
    max_ingredients: int = 20

    # Common ingredient normalizations
    normalize_units: bool = True
    expand_abbreviations: bool = True
