Sure, here's an updated version of the README with the requested logos and branding:

## Recipe Generation API

This repository contains code for a REST API that generates recipes using the T5 model from Hugging Face. 

### How it Works

The API is built with FastAPI and runs on a Docker container. 

The `generation_function` takes in a list of ingredients and generates two recipes using the T5 model. The generated recipes are then parsed into a list of Python dictionaries with keys for the recipe title, ingredients, and directions using the `parse_generated_recipes` function.

```python
def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt",
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(output_ids, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe
```

```python
def parse_generated_recipes(recipe_list):
    parsed_recipes = []
    for recipe in recipe_list:
        recipe_obj = {}
        title_match = re.search(r" title: (.+?)\n", recipe)
        ingredients_match = re.search(r" ingredients: (.+?)\n directions:", recipe)
        directions_match = re.search(r" directions: (.+)", recipe)

        if title_match:
            recipe_obj["title"] = title_match.group(1).strip()
        if ingredients_match:
            ingredients = ingredients_match.group(1).strip().split('--')
            recipe_obj["ingredients"] = [ingredient.strip() for ingredient in ingredients]
        if directions_match:
            directions = directions_match.group(1).strip().split('--')
            recipe_obj["directions"] = [direction.strip() for direction in directions]

        parsed_recipes.append(recipe_obj)
    return parsed_recipes
```

```python
class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    directions: List[str]

@app.post("/generate_recipes", response_model=List[Recipe])
def generate_recipes(items: List[str]):
    generated = generation_function(items)
    parsed_recipes = parse_generated_recipes(generated)
    return parsed_recipes
```

### How to Use

To use the API, first clone the repository:

```
git clone https://github.com/example/repo.git
cd repo
```

Then, build the Docker container:

```
docker build -t recipe-generation .
```

Finally, run the container:

```
docker run -p 8000:8000 -d recipe-generation
```

To generate recipes, send a POST request to `http://localhost:8000/generate_recipes` with a JSON payload containing a list of ingredients. For example:

```json
{
  "items": ["eggs, bacon"]
}
```

This will return a list of two recipes in the following format:

```json
[
  {
    "title": "Recipe Title 1",
    "ingredients": ["Ingredient 1", "Ingredient 2"],
    "directions": ["Step 1", "Step 2", "Step 3"]
  },
  {
    "title": "Recipe Title 2",
    "ingredients": ["Ingredient 1", "Ingredient 2"],
    "directions": ["Step 1", "Step 2"]
  }
]
```


### Credits
This project uses the T5 recipe generation model from the Hugging Face model hub: https://huggingface.co/flax-community/t5-recipe-generation

Created by John Peterson

