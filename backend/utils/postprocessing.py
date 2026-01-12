"""
Output postprocessing module for recipe generation.
Handles parsing, cleaning, and validation of generated recipes.
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ParsedRecipe:
    """Structured representation of a parsed recipe."""
    title: str = ""
    ingredients: List[str] = field(default_factory=list)
    directions: List[str] = field(default_factory=list)
    raw_text: str = ""
    parse_success: bool = False
    parse_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "title": self.title,
            "ingredients": self.ingredients,
            "directions": self.directions,
        }

    def is_complete(self) -> bool:
        """Check if recipe has all required components."""
        return bool(self.title and self.ingredients and self.directions)


class OutputParser:
    """
    Parser for generated recipe text.
    Handles extraction, cleaning, and validation of recipe components.
    """

    # Token mappings for postprocessing
    TOKENS_MAP = {
        "<sep>": "--",
        "<section>": "\n",
    }

    # Regex patterns for extraction
    PATTERNS = {
        "title": [
            r"title:\s*(.+?)(?:\n|ingredients:)",
            r"^(.+?)\n\s*ingredients:",
            r"title:\s*(.+?)$",
        ],
        "ingredients": [
            r"ingredients:\s*(.+?)(?:\n\s*directions:|\ndirections:)",
            r"ingredients:\s*(.+?)directions:",
            r"ingredients:\s*(.+?)$",
        ],
        "directions": [
            r"directions:\s*(.+?)$",
            r"directions:\s*(.+)",
        ],
    }

    def __init__(
        self,
        min_title_length: int = 3,
        max_title_length: int = 100,
        min_ingredients: int = 1,
        min_directions: int = 1,
        clean_empty_items: bool = True,
        fix_common_errors: bool = True,
    ):
        self.min_title_length = min_title_length
        self.max_title_length = max_title_length
        self.min_ingredients = min_ingredients
        self.min_directions = min_directions
        self.clean_empty_items = clean_empty_items
        self.fix_common_errors = fix_common_errors

    def parse_batch(self, generated_texts: List[str], special_tokens: List[str]) -> List[ParsedRecipe]:
        """Parse a batch of generated recipe texts."""
        return [self.parse_single(text, special_tokens) for text in generated_texts]

    def parse_single(self, text: str, special_tokens: List[str]) -> ParsedRecipe:
        """Parse a single generated recipe text."""
        recipe = ParsedRecipe(raw_text=text)

        # Step 1: Remove special tokens and apply mappings
        cleaned_text = self._clean_special_tokens(text, special_tokens)

        # Step 2: Apply common error fixes
        if self.fix_common_errors:
            cleaned_text = self._fix_common_errors(cleaned_text)

        # Step 3: Extract components
        recipe.title = self._extract_title(cleaned_text, recipe)
        recipe.ingredients = self._extract_ingredients(cleaned_text, recipe)
        recipe.directions = self._extract_directions(cleaned_text, recipe)

        # Step 4: Validate and clean
        self._validate_and_clean(recipe)

        recipe.parse_success = recipe.is_complete()

        return recipe

    def _clean_special_tokens(self, text: str, special_tokens: List[str]) -> str:
        """Remove special tokens and apply token mappings."""
        # Apply token mappings first
        for token, replacement in self.TOKENS_MAP.items():
            text = text.replace(token, replacement)

        # Remove remaining special tokens
        for token in special_tokens:
            text = text.replace(token, "")

        return text.strip()

    def _fix_common_errors(self, text: str) -> str:
        """Fix common generation errors."""
        # Fix missing spaces after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)

        # Fix missing colons in section headers
        text = re.sub(r'\btitle\s+(?!:)', 'title: ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bingredients\s+(?!:)', 'ingredients: ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdirections\s+(?!:)', 'directions: ', text, flags=re.IGNORECASE)

        # Normalize newlines
        text = re.sub(r'\n\s*\n', '\n', text)

        return text.strip()

    def _extract_title(self, text: str, recipe: ParsedRecipe) -> str:
        """Extract recipe title."""
        for pattern in self.PATTERNS["title"]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                title = self._clean_title(title)
                if self._validate_title(title):
                    return title

        recipe.parse_warnings.append("Could not extract title")
        return ""

    def _extract_ingredients(self, text: str, recipe: ParsedRecipe) -> List[str]:
        """Extract recipe ingredients."""
        for pattern in self.PATTERNS["ingredients"]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1).strip()
                ingredients = self._parse_list(ingredients_text)
                if len(ingredients) >= self.min_ingredients:
                    return ingredients

        recipe.parse_warnings.append("Could not extract ingredients")
        return []

    def _extract_directions(self, text: str, recipe: ParsedRecipe) -> List[str]:
        """Extract recipe directions."""
        for pattern in self.PATTERNS["directions"]:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                directions_text = match.group(1).strip()
                directions = self._parse_list(directions_text)
                if len(directions) >= self.min_directions:
                    return directions

        recipe.parse_warnings.append("Could not extract directions")
        return []

    def _parse_list(self, text: str) -> List[str]:
        """Parse a delimited list of items."""
        # Split by delimiter
        items = text.split('--')

        # Clean each item
        cleaned = []
        for item in items:
            item = item.strip()
            # Remove leading numbers/bullets
            item = re.sub(r'^[\d\.\)\-\*]+\s*', '', item)
            item = item.strip()

            if self.clean_empty_items and not item:
                continue

            if len(item) > 1:  # Skip single characters
                cleaned.append(item)

        return cleaned

    def _clean_title(self, title: str) -> str:
        """Clean extracted title."""
        # Remove leading/trailing punctuation
        title = title.strip('.,!?:;-_')
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        return title

    def _validate_title(self, title: str) -> bool:
        """Validate title meets requirements."""
        if len(title) < self.min_title_length:
            return False
        if len(title) > self.max_title_length:
            return False
        return True

    def _validate_and_clean(self, recipe: ParsedRecipe) -> None:
        """Validate and clean all recipe components."""
        # Clean ingredients
        recipe.ingredients = [
            ing for ing in recipe.ingredients
            if len(ing.strip()) > 1
        ]

        # Clean directions
        recipe.directions = [
            dir for dir in recipe.directions
            if len(dir.strip()) > 2
        ]

        # Truncate title if too long
        if len(recipe.title) > self.max_title_length:
            recipe.title = recipe.title[:self.max_title_length].rsplit(' ', 1)[0]
            recipe.parse_warnings.append("Title truncated")

        # Validate completeness
        if len(recipe.ingredients) < self.min_ingredients:
            recipe.parse_warnings.append(f"Too few ingredients ({len(recipe.ingredients)})")

        if len(recipe.directions) < self.min_directions:
            recipe.parse_warnings.append(f"Too few directions ({len(recipe.directions)})")
