"""
Scoring module for recipe generation.
Implements various metrics to evaluate recipe quality.
"""

import re
from typing import List, Dict, Optional, Set
from collections import Counter
from dataclasses import dataclass, field
import math


@dataclass
class RecipeScore:
    """Comprehensive score for a generated recipe."""
    overall_score: float = 0.0
    completeness_score: float = 0.0
    ingredient_coverage_score: float = 0.0
    coherence_score: float = 0.0
    length_score: float = 0.0
    details: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "completeness_score": round(self.completeness_score, 3),
            "ingredient_coverage_score": round(self.ingredient_coverage_score, 3),
            "coherence_score": round(self.coherence_score, 3),
            "length_score": round(self.length_score, 3),
            "details": self.details,
        }


class RecipeScorer:
    """
    Scorer for evaluating generated recipe quality.
    Combines multiple metrics for comprehensive evaluation.
    """

    def __init__(
        self,
        completeness_weight: float = 0.3,
        ingredient_coverage_weight: float = 0.3,
        coherence_weight: float = 0.2,
        length_weight: float = 0.2,
        ideal_ingredients_range: tuple = (3, 15),
        ideal_directions_range: tuple = (3, 12),
    ):
        self.completeness_weight = completeness_weight
        self.ingredient_coverage_weight = ingredient_coverage_weight
        self.coherence_weight = coherence_weight
        self.length_weight = length_weight
        self.ideal_ingredients_range = ideal_ingredients_range
        self.ideal_directions_range = ideal_directions_range

        # Normalize weights
        total = (completeness_weight + ingredient_coverage_weight +
                coherence_weight + length_weight)
        self.completeness_weight /= total
        self.ingredient_coverage_weight /= total
        self.coherence_weight /= total
        self.length_weight /= total

    def score_recipe(
        self,
        recipe: Dict,
        input_ingredients: List[str],
    ) -> RecipeScore:
        """
        Score a single recipe.

        Args:
            recipe: Dict with title, ingredients, directions
            input_ingredients: Original input ingredients

        Returns:
            RecipeScore with detailed breakdown
        """
        score = RecipeScore()

        # Calculate individual scores
        score.completeness_score = self._calculate_completeness(recipe)
        score.ingredient_coverage_score = self._calculate_ingredient_coverage(
            recipe.get("ingredients", []),
            input_ingredients
        )
        score.coherence_score = self._calculate_coherence(recipe)
        score.length_score = self._calculate_length_score(recipe)

        # Calculate overall score
        score.overall_score = (
            self.completeness_weight * score.completeness_score +
            self.ingredient_coverage_weight * score.ingredient_coverage_score +
            self.coherence_weight * score.coherence_score +
            self.length_weight * score.length_score
        )

        # Add details
        score.details = {
            "has_title": bool(recipe.get("title")),
            "num_ingredients": len(recipe.get("ingredients", [])),
            "num_directions": len(recipe.get("directions", [])),
            "input_ingredients_used": self._count_ingredients_used(
                recipe.get("ingredients", []),
                input_ingredients
            ),
            "total_input_ingredients": len(input_ingredients),
        }

        return score

    def score_batch(
        self,
        recipes: List[Dict],
        input_ingredients: List[str],
    ) -> List[RecipeScore]:
        """Score multiple recipes."""
        return [
            self.score_recipe(recipe, input_ingredients)
            for recipe in recipes
        ]

    def select_best(
        self,
        recipes: List[Dict],
        input_ingredients: List[str],
    ) -> tuple:
        """
        Select the best recipe from a list.

        Returns:
            Tuple of (best_recipe, best_score, all_scores)
        """
        if not recipes:
            return None, None, []

        scores = self.score_batch(recipes, input_ingredients)
        best_idx = max(range(len(scores)), key=lambda i: scores[i].overall_score)

        return recipes[best_idx], scores[best_idx], scores

    def _calculate_completeness(self, recipe: Dict) -> float:
        """Calculate completeness score (0-1)."""
        score = 0.0

        # Title component (30%)
        title = recipe.get("title", "")
        if title:
            # Bonus for reasonable title length
            if 5 <= len(title) <= 50:
                score += 0.3
            else:
                score += 0.15

        # Ingredients component (35%)
        ingredients = recipe.get("ingredients", [])
        if ingredients:
            min_ing, max_ing = self.ideal_ingredients_range
            if min_ing <= len(ingredients) <= max_ing:
                score += 0.35
            elif len(ingredients) > 0:
                score += 0.2

        # Directions component (35%)
        directions = recipe.get("directions", [])
        if directions:
            min_dir, max_dir = self.ideal_directions_range
            if min_dir <= len(directions) <= max_dir:
                score += 0.35
            elif len(directions) > 0:
                score += 0.2

        return score

    def _calculate_ingredient_coverage(
        self,
        recipe_ingredients: List[str],
        input_ingredients: List[str]
    ) -> float:
        """Calculate how well the recipe uses input ingredients (0-1)."""
        if not input_ingredients:
            return 1.0

        if not recipe_ingredients:
            return 0.0

        # Normalize ingredients for comparison
        input_set = self._normalize_ingredients(input_ingredients)
        recipe_text = ' '.join(recipe_ingredients).lower()

        # Count how many input ingredients appear in recipe
        matches = 0
        for ingredient in input_set:
            # Check if ingredient or its parts appear in recipe
            if self._ingredient_in_text(ingredient, recipe_text):
                matches += 1

        coverage = matches / len(input_set)

        # Bonus for using most/all ingredients
        if coverage >= 0.8:
            coverage = min(1.0, coverage * 1.1)

        return coverage

    def _calculate_coherence(self, recipe: Dict) -> float:
        """Calculate coherence score based on text quality (0-1)."""
        score = 0.0

        # Check title coherence
        title = recipe.get("title", "")
        if title:
            # Title should be readable (not random characters)
            if re.match(r'^[A-Z][a-zA-Z\s\-\']+$', title):
                score += 0.3
            elif re.match(r'^[A-Za-z]', title):
                score += 0.2

        # Check ingredients coherence
        ingredients = recipe.get("ingredients", [])
        if ingredients:
            valid_ingredients = sum(
                1 for ing in ingredients
                if self._is_valid_ingredient_text(ing)
            )
            score += 0.35 * (valid_ingredients / max(len(ingredients), 1))

        # Check directions coherence
        directions = recipe.get("directions", [])
        if directions:
            valid_directions = sum(
                1 for dir in directions
                if self._is_valid_direction_text(dir)
            )
            score += 0.35 * (valid_directions / max(len(directions), 1))

        return score

    def _calculate_length_score(self, recipe: Dict) -> float:
        """Calculate score based on appropriate lengths (0-1)."""
        score = 0.0

        # Ingredients length score
        ingredients = recipe.get("ingredients", [])
        min_ing, max_ing = self.ideal_ingredients_range
        if min_ing <= len(ingredients) <= max_ing:
            score += 0.5
        elif ingredients:
            # Partial score for having ingredients
            dist = min(
                abs(len(ingredients) - min_ing),
                abs(len(ingredients) - max_ing)
            )
            score += 0.5 * max(0, 1 - dist / max_ing)

        # Directions length score
        directions = recipe.get("directions", [])
        min_dir, max_dir = self.ideal_directions_range
        if min_dir <= len(directions) <= max_dir:
            score += 0.5
        elif directions:
            dist = min(
                abs(len(directions) - min_dir),
                abs(len(directions) - max_dir)
            )
            score += 0.5 * max(0, 1 - dist / max_dir)

        return score

    def _normalize_ingredients(self, ingredients: List[str]) -> Set[str]:
        """Normalize ingredients for comparison."""
        normalized = set()
        for ing in ingredients:
            # Extract main ingredient word (remove quantities, modifiers)
            words = ing.lower().split()
            # Keep meaningful words
            for word in words:
                if len(word) > 2 and not word.isdigit():
                    normalized.add(word)
        return normalized

    def _ingredient_in_text(self, ingredient: str, text: str) -> bool:
        """Check if an ingredient appears in text."""
        # Direct match
        if ingredient in text:
            return True

        # Check for common variations
        variations = [
            ingredient,
            ingredient + 's',  # plural
            ingredient + 'es',  # plural
            ingredient[:-1] if ingredient.endswith('s') else ingredient,  # singular
        ]

        return any(var in text for var in variations)

    def _is_valid_ingredient_text(self, text: str) -> bool:
        """Check if ingredient text is valid/readable."""
        if len(text) < 2:
            return False
        # Should contain at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False
        # Should not be mostly special characters
        letter_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        return letter_ratio > 0.6

    def _is_valid_direction_text(self, text: str) -> bool:
        """Check if direction text is valid/readable."""
        if len(text) < 5:
            return False
        # Should start with a letter or number
        if not re.match(r'^[A-Za-z0-9]', text):
            return False
        # Should contain spaces (multiple words)
        return ' ' in text

    def _count_ingredients_used(
        self,
        recipe_ingredients: List[str],
        input_ingredients: List[str]
    ) -> int:
        """Count how many input ingredients were used."""
        if not input_ingredients or not recipe_ingredients:
            return 0

        input_set = self._normalize_ingredients(input_ingredients)
        recipe_text = ' '.join(recipe_ingredients).lower()

        return sum(
            1 for ing in input_set
            if self._ingredient_in_text(ing, recipe_text)
        )


def calculate_bleu_score(reference: str, candidate: str, n: int = 4) -> float:
    """
    Calculate BLEU score between reference and candidate text.
    Simplified implementation for recipe evaluation.
    """
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        return Counter(
            tuple(words[i:i+n])
            for i in range(len(words) - n + 1)
        )

    if not reference or not candidate:
        return 0.0

    # Calculate precision for each n-gram level
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = get_ngrams(reference, i)
        cand_ngrams = get_ngrams(candidate, i)

        if not cand_ngrams:
            precisions.append(0.0)
            continue

        # Count matches
        matches = sum(
            min(cand_ngrams[ng], ref_ngrams[ng])
            for ng in cand_ngrams
        )
        total = sum(cand_ngrams.values())

        precisions.append(matches / total if total > 0 else 0.0)

    # Geometric mean of precisions
    if 0 in precisions:
        return 0.0

    log_precision = sum(math.log(p) for p in precisions) / len(precisions)

    # Brevity penalty
    ref_len = len(reference.split())
    cand_len = len(candidate.split())
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0

    return bp * math.exp(log_precision)
