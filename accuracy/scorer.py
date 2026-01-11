"""
Recipe scoring and coherence evaluation.
"""

from typing import List, Dict, Tuple
from .validator import IngredientValidator, QualityChecker
from .ingredients_db import (
    COOKING_VERBS,
    ENDING_VERBS,
    STARTING_VERBS,
    normalize_ingredient
)


class CoherenceScorer:
    """Scores the logical coherence of recipe instructions."""

    def __init__(self):
        self.cooking_verbs = COOKING_VERBS
        self.ending_verbs = ENDING_VERBS
        self.starting_verbs = STARTING_VERBS

    def score_coherence(self, recipe: Dict) -> float:
        """
        Score the overall coherence of a recipe.

        Args:
            recipe: Recipe dictionary with title, ingredients, directions

        Returns:
            Float between 0 and 1 representing coherence score
        """
        scores = []

        directions = recipe.get("directions", [])
        ingredients = recipe.get("ingredients", [])

        if not directions:
            return 0.0

        # Score 1: Instructions start with cooking verbs
        verb_score = self._score_cooking_verbs(directions)
        scores.append(verb_score)

        # Score 2: Logical order (prep -> cook -> serve)
        order_score = self._score_logical_order(directions)
        scores.append(order_score)

        # Score 3: Ingredients referenced in directions
        reference_score = self._score_ingredient_references(ingredients, directions)
        scores.append(reference_score)

        # Score 4: Instruction length consistency
        length_score = self._score_length_consistency(directions)
        scores.append(length_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_cooking_verbs(self, directions: List[str]) -> float:
        """Score based on cooking verbs at start of instructions."""
        if not directions:
            return 0.0

        verb_starts = 0
        for direction in directions:
            words = direction.lower().split()
            if words and words[0] in self.cooking_verbs:
                verb_starts += 1

        return verb_starts / len(directions)

    def _score_logical_order(self, directions: List[str]) -> float:
        """Score based on logical cooking order."""
        if len(directions) < 2:
            return 0.5  # Can't evaluate order with single step

        score = 1.0
        penalties = 0
        total_checks = 0

        directions_lower = [d.lower() for d in directions]

        # Check 1: "Preheat" should come in first half
        for i, d in enumerate(directions_lower):
            if 'preheat' in d:
                total_checks += 1
                if i > len(directions) // 2:
                    penalties += 1
                break

        # Check 2: "Serve" or "enjoy" should come near the end
        for verb in ['serve', 'enjoy', 'plate']:
            for i, d in enumerate(directions_lower):
                if verb in d:
                    total_checks += 1
                    if i < len(directions) * 0.6:
                        penalties += 1
                    break

        # Check 3: Prep verbs (chop, dice, mix ingredients) before cooking verbs
        prep_verbs = {'chop', 'dice', 'mince', 'slice', 'mix', 'combine', 'prepare', 'gather'}
        cook_verbs = {'bake', 'fry', 'boil', 'roast', 'grill', 'sauté', 'saute'}

        first_prep_idx = None
        first_cook_idx = None

        for i, d in enumerate(directions_lower):
            words = set(d.split())
            if first_prep_idx is None and words & prep_verbs:
                first_prep_idx = i
            if first_cook_idx is None and words & cook_verbs:
                first_cook_idx = i

        if first_prep_idx is not None and first_cook_idx is not None:
            total_checks += 1
            if first_prep_idx > first_cook_idx:
                penalties += 1

        if total_checks == 0:
            return 0.7  # Default score if no checks applicable

        return max(0.0, 1.0 - (penalties / total_checks))

    def _score_ingredient_references(self, ingredients: List[str], directions: List[str]) -> float:
        """Score based on how well directions reference ingredients."""
        if not ingredients or not directions:
            return 0.5

        directions_text = ' '.join(directions).lower()
        referenced = 0

        for ing in ingredients:
            normalized = normalize_ingredient(ing)
            if normalized and len(normalized) > 2:
                # Check if ingredient or key words appear in directions
                if normalized in directions_text:
                    referenced += 1
                else:
                    # Check individual significant words
                    words = [w for w in normalized.split() if len(w) > 3]
                    if any(w in directions_text for w in words):
                        referenced += 0.5

        return min(1.0, referenced / len(ingredients))

    def _score_length_consistency(self, directions: List[str]) -> float:
        """Score based on consistency of instruction lengths."""
        if len(directions) < 2:
            return 1.0

        lengths = [len(d.split()) for d in directions if d]
        if not lengths:
            return 0.0

        avg_length = sum(lengths) / len(lengths)
        if avg_length == 0:
            return 0.0

        # Calculate coefficient of variation
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / avg_length

        # Lower CV means more consistent, convert to score
        # CV < 0.5 is good, CV > 1.5 is poor
        return max(0.0, min(1.0, 1.0 - (cv - 0.5)))


class RecipeScorer:
    """Scores and ranks recipes based on multiple quality criteria."""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scorer with optional custom weights.

        Args:
            weights: Dictionary of score weights. Keys are:
                - completeness
                - ingredient_coverage
                - instruction_quality
                - coherence
        """
        self.weights = weights or {
            "completeness": 0.20,
            "ingredient_coverage": 0.30,
            "instruction_quality": 0.25,
            "coherence": 0.25
        }
        self.validator = IngredientValidator()
        self.quality_checker = QualityChecker()
        self.coherence_scorer = CoherenceScorer()

    def score_recipe(self, recipe: Dict, input_items: List[str]) -> Tuple[float, Dict[str, float]]:
        """
        Score a recipe based on multiple criteria.

        Args:
            recipe: Recipe dictionary with title, ingredients, directions
            input_items: Original input ingredients from user

        Returns:
            Tuple of (total_score, individual_scores_dict)
        """
        scores = {}

        # Completeness score
        scores["completeness"] = self._score_completeness(recipe)

        # Ingredient coverage score
        scores["ingredient_coverage"] = self._score_coverage(recipe, input_items)

        # Instruction quality score
        scores["instruction_quality"] = self._score_instructions(recipe)

        # Coherence score
        scores["coherence"] = self.coherence_scorer.score_coherence(recipe)

        # Calculate weighted total
        total = sum(
            scores[key] * self.weights.get(key, 0.25)
            for key in scores
        )

        return total, scores

    def _score_completeness(self, recipe: Dict) -> float:
        """Score based on recipe completeness."""
        score = 0.0

        # Has title (0.3)
        if recipe.get("title", "").strip():
            score += 0.3

        # Has ingredients (0.35)
        ingredients = recipe.get("ingredients", [])
        if ingredients:
            # More ingredients = better, up to a point
            ing_count = len([i for i in ingredients if i and i.strip()])
            score += min(0.35, 0.35 * (ing_count / 5))

        # Has directions (0.35)
        directions = recipe.get("directions", [])
        if directions:
            # More directions = better, up to a point
            dir_count = len([d for d in directions if d and d.strip()])
            score += min(0.35, 0.35 * (dir_count / 5))

        return score

    def _score_coverage(self, recipe: Dict, input_items: List[str]) -> float:
        """Score based on input ingredient coverage."""
        if not input_items:
            return 1.0

        ingredients = recipe.get("ingredients", [])
        return self.validator.check_input_coverage(input_items, ingredients)

    def _score_instructions(self, recipe: Dict) -> float:
        """Score based on instruction quality."""
        directions = recipe.get("directions", [])
        if not directions:
            return 0.0

        scores = []

        # Clean directions
        clean_directions = [d for d in directions if d and d.strip()]
        if not clean_directions:
            return 0.0

        # 1. Average instruction length (prefer 5-20 words)
        lengths = [len(d.split()) for d in clean_directions]
        avg_length = sum(lengths) / len(lengths)
        if 5 <= avg_length <= 20:
            scores.append(1.0)
        elif 3 <= avg_length <= 30:
            scores.append(0.7)
        else:
            scores.append(0.3)

        # 2. Number of instructions (prefer 4-10)
        num_instructions = len(clean_directions)
        if 4 <= num_instructions <= 10:
            scores.append(1.0)
        elif 2 <= num_instructions <= 15:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # 3. No duplicate instructions
        unique = set(d.lower().strip() for d in clean_directions)
        if len(unique) == len(clean_directions):
            scores.append(1.0)
        else:
            scores.append(0.5)

        # 4. Instructions have proper endings
        proper_endings = sum(1 for d in clean_directions if d.strip()[-1] in '.!?')
        scores.append(proper_endings / len(clean_directions))

        return sum(scores) / len(scores)

    def rank_recipes(
        self,
        recipes: List[Dict],
        input_items: List[str],
        top_n: int = 2
    ) -> List[Tuple[Dict, float, Dict[str, float]]]:
        """
        Rank recipes by score and return top N diverse recipes.

        Args:
            recipes: List of recipe dictionaries
            input_items: Original input ingredients
            top_n: Number of top recipes to return

        Returns:
            List of tuples: (recipe, total_score, individual_scores)
        """
        scored_recipes = []

        for recipe in recipes:
            total_score, individual_scores = self.score_recipe(recipe, input_items)
            scored_recipes.append((recipe, total_score, individual_scores))

        # Sort by total score descending
        scored_recipes.sort(key=lambda x: x[1], reverse=True)

        # Select diverse recipes (filter out similar ones)
        selected = []
        for recipe, score, details in scored_recipes:
            if not self._is_similar_to_selected(recipe, [r[0] for r in selected]):
                selected.append((recipe, score, details))
                if len(selected) >= top_n:
                    break

        return selected

    def _is_similar_to_selected(self, recipe: Dict, selected: List[Dict], threshold: float = 0.4) -> bool:
        """Check if recipe is too similar to already selected recipes."""
        for existing in selected:
            similarity = self._calculate_similarity(recipe, existing)
            if similarity > threshold:
                return True
        return False

    def _calculate_similarity(self, recipe1: Dict, recipe2: Dict) -> float:
        """Calculate similarity between two recipes (0-1)."""
        scores = []

        # Title similarity
        title1 = recipe1.get("title", "").lower().strip()
        title2 = recipe2.get("title", "").lower().strip()
        if title1 and title2:
            title_words1 = set(title1.split())
            title_words2 = set(title2.split())
            if title_words1 or title_words2:
                title_sim = len(title_words1 & title_words2) / max(len(title_words1 | title_words2), 1)
                scores.append(title_sim)

        # Ingredients similarity
        ing1 = set(i.lower().strip() for i in recipe1.get("ingredients", []) if i)
        ing2 = set(i.lower().strip() for i in recipe2.get("ingredients", []) if i)
        if ing1 or ing2:
            ing_sim = len(ing1 & ing2) / max(len(ing1 | ing2), 1)
            scores.append(ing_sim)

        # Directions similarity (compare first few words of each step)
        dir1 = [' '.join(d.lower().split()[:5]) for d in recipe1.get("directions", []) if d]
        dir2 = [' '.join(d.lower().split()[:5]) for d in recipe2.get("directions", []) if d]
        if dir1 and dir2:
            dir_set1 = set(dir1)
            dir_set2 = set(dir2)
            dir_sim = len(dir_set1 & dir_set2) / max(len(dir_set1 | dir_set2), 1)
            scores.append(dir_sim)

        return sum(scores) / len(scores) if scores else 0.0

    def filter_by_threshold(
        self,
        recipes: List[Dict],
        input_items: List[str],
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Filter recipes that meet minimum score threshold.

        Args:
            recipes: List of recipe dictionaries
            input_items: Original input ingredients
            min_score: Minimum total score to pass

        Returns:
            List of recipes meeting the threshold
        """
        passing = []

        for recipe in recipes:
            total_score, _ = self.score_recipe(recipe, input_items)
            if total_score >= min_score:
                passing.append(recipe)

        return passing
