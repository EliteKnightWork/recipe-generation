"""
Ingredient validation and quality checking for generated recipes.
"""

import re
from typing import List, Tuple, Dict, Set
from .ingredients_db import (
    get_valid_ingredients,
    get_cooking_units,
    normalize_ingredient,
    COOKING_VERBS
)


class IngredientValidator:
    """Validates ingredients against a known food database."""

    def __init__(self):
        self.valid_ingredients = get_valid_ingredients()
        self.cooking_units = get_cooking_units()

    def validate_ingredient(self, ingredient: str) -> bool:
        """
        Check if an ingredient is valid (exists in food database).

        Args:
            ingredient: The ingredient string to validate

        Returns:
            True if ingredient is valid, False otherwise
        """
        normalized = normalize_ingredient(ingredient)

        # Direct match
        if normalized in self.valid_ingredients:
            return True

        # Check if any valid ingredient is contained in the normalized string
        for valid in self.valid_ingredients:
            if valid in normalized or normalized in valid:
                return True

        # Check individual words
        words = normalized.split()
        for word in words:
            if word in self.valid_ingredients:
                return True

        return False

    def validate_ingredients_list(self, ingredients: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of ingredients.

        Args:
            ingredients: List of ingredient strings

        Returns:
            Tuple of (valid_ingredients, invalid_ingredients)
        """
        valid = []
        invalid = []

        for ing in ingredients:
            if self.validate_ingredient(ing):
                valid.append(ing)
            else:
                invalid.append(ing)

        return valid, invalid

    def check_input_coverage(self, input_items: List[str], recipe_ingredients: List[str]) -> float:
        """
        Check what percentage of input ingredients appear in the recipe.

        Args:
            input_items: Original input ingredients from user
            recipe_ingredients: Ingredients in the generated recipe

        Returns:
            Float between 0 and 1 representing coverage percentage
        """
        if not input_items:
            return 1.0

        # Normalize all ingredients for comparison
        normalized_inputs = {normalize_ingredient(item) for item in input_items}
        normalized_recipe = ' '.join(normalize_ingredient(ing) for ing in recipe_ingredients)

        matches = 0
        for inp in normalized_inputs:
            # Check if input ingredient appears in recipe ingredients
            if inp and (inp in normalized_recipe or any(inp in normalize_ingredient(r) for r in recipe_ingredients)):
                matches += 1

        return matches / len(normalized_inputs) if normalized_inputs else 1.0


class QualityChecker:
    """Checks and improves recipe quality."""

    def __init__(self):
        self.cooking_verbs = COOKING_VERBS
        self.min_ingredients = 2
        self.min_directions = 2
        self.min_title_length = 3

    def check_recipe(self, recipe: Dict) -> Tuple[bool, List[str]]:
        """
        Check recipe for quality issues.

        Args:
            recipe: Recipe dictionary with title, ingredients, directions

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check title
        title = recipe.get("title", "")
        if not title or len(title.strip()) < self.min_title_length:
            issues.append("Missing or too short title")

        # Check ingredients
        ingredients = recipe.get("ingredients", [])
        if len(ingredients) < self.min_ingredients:
            issues.append(f"Too few ingredients (minimum {self.min_ingredients})")

        # Check for empty ingredients
        empty_ingredients = [i for i in ingredients if not i or not i.strip()]
        if empty_ingredients:
            issues.append(f"Found {len(empty_ingredients)} empty ingredient(s)")

        # Check directions
        directions = recipe.get("directions", [])
        if len(directions) < self.min_directions:
            issues.append(f"Too few directions (minimum {self.min_directions})")

        # Check for empty directions
        empty_directions = [d for d in directions if not d or not d.strip()]
        if empty_directions:
            issues.append(f"Found {len(empty_directions)} empty direction(s)")

        # Check for duplicate directions
        unique_directions = set(d.lower().strip() for d in directions if d)
        if len(unique_directions) < len([d for d in directions if d]):
            issues.append("Found duplicate directions")

        # Check for incomplete sentences in directions
        for i, direction in enumerate(directions):
            if direction and len(direction.strip()) < 10:
                issues.append(f"Direction {i+1} may be incomplete (too short)")

        # Check if directions reference ingredients
        if ingredients and directions:
            directions_text = ' '.join(directions).lower()
            referenced = 0
            for ing in ingredients:
                normalized = normalize_ingredient(ing)
                if normalized and len(normalized) > 2:
                    if normalized in directions_text:
                        referenced += 1
            if referenced < len(ingredients) * 0.3:
                issues.append("Directions don't reference enough ingredients")

        return (len(issues) == 0, issues)

    def clean_recipe(self, recipe: Dict) -> Dict:
        """
        Clean and fix common recipe issues.

        Args:
            recipe: Recipe dictionary

        Returns:
            Cleaned recipe dictionary
        """
        cleaned = {}

        # Clean title
        title = recipe.get("title", "Untitled Recipe")
        cleaned["title"] = title.strip() if title else "Untitled Recipe"

        # Clean ingredients - remove empty and duplicates while preserving order
        ingredients = recipe.get("ingredients", [])
        seen = set()
        cleaned_ingredients = []
        for ing in ingredients:
            if ing and ing.strip():
                normalized = ing.strip().lower()
                if normalized not in seen:
                    seen.add(normalized)
                    cleaned_ingredients.append(ing.strip())
        cleaned["ingredients"] = cleaned_ingredients

        # Clean directions - remove empty and fix formatting
        directions = recipe.get("directions", [])
        cleaned_directions = []
        seen_directions = set()
        for direction in directions:
            if direction and direction.strip():
                clean_dir = direction.strip()
                # Capitalize first letter
                if clean_dir and clean_dir[0].islower():
                    clean_dir = clean_dir[0].upper() + clean_dir[1:]
                # Add period if missing
                if clean_dir and clean_dir[-1] not in '.!?':
                    clean_dir += '.'
                # Check for duplicates
                normalized_dir = clean_dir.lower()
                if normalized_dir not in seen_directions:
                    seen_directions.add(normalized_dir)
                    cleaned_directions.append(clean_dir)
        cleaned["directions"] = cleaned_directions

        return cleaned

    def check_hallucinated_content(self, recipe: Dict) -> List[str]:
        """
        Detect potentially hallucinated content in recipe.

        Args:
            recipe: Recipe dictionary

        Returns:
            List of suspected hallucinations
        """
        hallucinations = []

        # Check for suspicious patterns in ingredients
        suspicious_patterns = [
            r'\d+\s*(cups?|tbsp|tsp|oz)\s+of\s+(nothing|none|air)',
            r'(\d+\s+){3,}',  # Multiple consecutive numbers
            r'[^\w\s]{3,}',  # Multiple consecutive special characters
        ]

        ingredients = recipe.get("ingredients", [])
        for ing in ingredients:
            for pattern in suspicious_patterns:
                if re.search(pattern, ing, re.IGNORECASE):
                    hallucinations.append(f"Suspicious ingredient: {ing}")
                    break

        # Check for nonsensical directions
        directions = recipe.get("directions", [])
        for i, direction in enumerate(directions):
            # Check for very repetitive text
            words = direction.lower().split()
            if len(words) > 5:
                unique_words = set(words)
                if len(unique_words) < len(words) * 0.3:
                    hallucinations.append(f"Direction {i+1} appears repetitive")

        return hallucinations


def validate_recipe_constraints(
    recipe: Dict,
    input_items: List[str],
    min_coverage: float = 0.5,
    min_directions: int = 3
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate that a recipe meets all constraints.

    Args:
        recipe: Recipe dictionary
        input_items: Original input ingredients
        min_coverage: Minimum percentage of input ingredients that must appear
        min_directions: Minimum number of directions required

    Returns:
        Tuple of (passes_constraints, constraint_results)
    """
    validator = IngredientValidator()
    checker = QualityChecker()

    results = {
        "has_title": bool(recipe.get("title", "").strip()),
        "ingredient_count": len(recipe.get("ingredients", [])),
        "direction_count": len(recipe.get("directions", [])),
        "input_coverage": validator.check_input_coverage(
            input_items, recipe.get("ingredients", [])
        ),
        "quality_issues": checker.check_recipe(recipe)[1],
        "hallucinations": checker.check_hallucinated_content(recipe)
    }

    passes = (
        results["has_title"] and
        results["direction_count"] >= min_directions and
        results["input_coverage"] >= min_coverage and
        len(results["quality_issues"]) == 0 and
        len(results["hallucinations"]) == 0
    )

    return passes, results
