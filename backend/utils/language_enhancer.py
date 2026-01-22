"""
Language Enhancement Module
Uses TinyLlama/TinyLlama-1.1B-Chat-v1.0 to enhance recipe language quality.
"""

import os
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageEnhancer:
    """Enhances recipe text using a language model for more professional output."""

    DEFAULT_MODEL = os.environ.get("LLAMA_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    LOCAL_MODEL_PATH = "/app/models/language-model"

    ENHANCEMENT_PROMPT = """You are a professional culinary writer. Rewrite the following recipe with elegant, professional cookbook-style language.

Rules:
- Keep all the same steps and ingredients
- Make descriptions more vivid and precise
- Use proper culinary terminology
- Add helpful cooking tips where appropriate
- Maintain the same structure (title, ingredients, directions)
- Do not add new ingredients or steps
- Keep it concise but refined

Original Recipe:
Title: {title}

Ingredients:
{ingredients}

Directions:
{directions}

---
Rewrite the recipe below in the exact same format (Title:, Ingredients:, Directions:):
"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        load_in_4bit: bool = True,
    ):
        """
        Initialize the language enhancer.

        Args:
            model_path: Path to model (local or HuggingFace hub)
            device: Torch device to use
            load_in_4bit: Whether to load in 4-bit quantization (saves VRAM)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.load_in_4bit = load_in_4bit

        # Determine model path
        import os
        if model_path:
            self.model_path = model_path
        elif os.path.exists(self.LOCAL_MODEL_PATH):
            self.model_path = self.LOCAL_MODEL_PATH
        else:
            self.model_path = self.DEFAULT_MODEL

        print(f"Language enhancer initialized with model: {self.model_path}")

    def load_model(self):
        """Load the model and tokenizer."""
        if self.is_loaded:
            return

        print(f"Loading language model from: {self.model_path}")

        # Configure quantization if requested
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if self.load_in_4bit and self.device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("Using 4-bit quantization")
            except ImportError:
                print("bitsandbytes not available, loading in fp16")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **model_kwargs
        )

        self.is_loaded = True
        print("Language model loaded successfully!")

    def enhance_recipe(
        self,
        title: str,
        ingredients: List[str],
        directions: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Enhance a single recipe's language.

        Args:
            title: Recipe title
            ingredients: List of ingredients
            directions: List of directions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dict with enhanced title, ingredients, directions
        """
        if not self.is_loaded:
            self.load_model()

        # Format the prompt
        ingredients_text = "\n".join(f"- {ing}" for ing in ingredients)
        directions_text = "\n".join(f"{i+1}. {d}" for i, d in enumerate(directions))

        prompt = self.ENHANCEMENT_PROMPT.format(
            title=title,
            ingredients=ingredients_text,
            directions=directions_text,
        )

        # Format for Llama chat
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output (only new tokens)
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse the enhanced recipe
        enhanced = self._parse_enhanced_output(generated, title, ingredients, directions)

        return enhanced

    def _parse_enhanced_output(
        self,
        output: str,
        original_title: str,
        original_ingredients: List[str],
        original_directions: List[str],
    ) -> Dict[str, Any]:
        """Parse the enhanced output back into structured format."""

        result = {
            "title": original_title,
            "ingredients": original_ingredients,
            "directions": original_directions,
            "enhanced": False,
        }

        try:
            lines = output.strip().split("\n")

            # Extract title
            for line in lines:
                if line.lower().startswith("title:"):
                    result["title"] = line.split(":", 1)[1].strip()
                    break

            # Extract ingredients section
            ingredients_start = None
            directions_start = None

            for i, line in enumerate(lines):
                if "ingredients:" in line.lower():
                    ingredients_start = i + 1
                elif "directions:" in line.lower():
                    directions_start = i + 1
                    break

            # Parse ingredients
            if ingredients_start is not None:
                end = directions_start - 1 if directions_start else len(lines)
                ingredients = []
                for line in lines[ingredients_start:end]:
                    line = line.strip()
                    if line and not line.lower().startswith("directions"):
                        # Remove bullet points or dashes
                        line = line.lstrip("-•* ").strip()
                        if line:
                            ingredients.append(line)
                if ingredients:
                    result["ingredients"] = ingredients

            # Parse directions
            if directions_start is not None:
                directions = []
                for line in lines[directions_start:]:
                    line = line.strip()
                    if line:
                        # Remove numbering
                        import re
                        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                        if line:
                            directions.append(line)
                if directions:
                    result["directions"] = directions

            result["enhanced"] = True

        except Exception as e:
            print(f"Error parsing enhanced output: {e}")
            # Keep original values on parse error

        return result

    def enhance_recipes(
        self,
        recipes: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Enhance multiple recipes.

        Args:
            recipes: List of recipe dicts with title, ingredients, directions
            **kwargs: Additional args passed to enhance_recipe

        Returns:
            List of enhanced recipe dicts
        """
        enhanced_recipes = []

        for recipe in recipes:
            enhanced = self.enhance_recipe(
                title=recipe.get("title", ""),
                ingredients=recipe.get("ingredients", []),
                directions=recipe.get("directions", []),
                **kwargs
            )

            # Preserve score if present
            if "score" in recipe:
                enhanced["score"] = recipe["score"]

            enhanced_recipes.append(enhanced)

        return enhanced_recipes
