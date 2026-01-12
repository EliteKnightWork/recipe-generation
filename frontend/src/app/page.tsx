'use client';

import { useState, FormEvent, KeyboardEvent } from 'react';

interface Recipe {
  title: string;
  ingredients: string[];
  directions: string[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [ingredients, setIngredients] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addIngredient = () => {
    const trimmed = inputValue.trim();
    if (trimmed && !ingredients.includes(trimmed.toLowerCase())) {
      setIngredients([...ingredients, trimmed.toLowerCase()]);
      setInputValue('');
    }
  };

  const removeIngredient = (index: number) => {
    setIngredients(ingredients.filter((_, i) => i !== index));
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addIngredient();
    }
  };

  const generateRecipes = async (e: FormEvent) => {
    e.preventDefault();
    if (ingredients.length === 0) {
      setError('Please add at least one ingredient');
      return;
    }

    setLoading(true);
    setError(null);
    setRecipes([]);

    try {
      const response = await fetch(`${API_URL}/generate_recipes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ingredients),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate recipes: ${response.statusText}`);
      }

      const data: Recipe[] = await response.json();
      setRecipes(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setIngredients([]);
    setRecipes([]);
    setError(null);
  };

  return (
    <main className="container">
      <header className="header">
        <h1>Recipe Generator</h1>
        <p>Enter your ingredients and let AI create delicious recipes for you</p>
      </header>

      <section className="input-section">
        <h2>Your Ingredients</h2>
        <form onSubmit={generateRecipes}>
          <div className="ingredients-input">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type an ingredient and press Enter..."
              disabled={loading}
            />
            <button type="button" className="btn btn-secondary" onClick={addIngredient}>
              Add
            </button>
          </div>

          {ingredients.length > 0 && (
            <div className="tags" style={{ marginBottom: '1rem' }}>
              {ingredients.map((ing, index) => (
                <span key={index} className="tag">
                  {ing}
                  <button type="button" onClick={() => removeIngredient(index)}>
                    &times;
                  </button>
                </span>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', gap: '1rem' }}>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Generating...' : 'Generate Recipes'}
            </button>
            {ingredients.length > 0 && (
              <button type="button" className="btn btn-secondary" onClick={clearAll}>
                Clear All
              </button>
            )}
          </div>
        </form>
      </section>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Generating recipes...</p>
        </div>
      )}

      {!loading && recipes.length > 0 && (
        <section className="recipes-section">
          {recipes.map((recipe, index) => (
            <article key={index} className="recipe-card">
              <h3>{recipe.title}</h3>

              <div className="recipe-section ingredients">
                <h4>Ingredients</h4>
                <ul>
                  {recipe.ingredients.map((ing, i) => (
                    <li key={i}>{ing}</li>
                  ))}
                </ul>
              </div>

              <div className="recipe-section directions">
                <h4>Directions</h4>
                <ul>
                  {recipe.directions.map((dir, i) => (
                    <li key={i}>{dir}</li>
                  ))}
                </ul>
              </div>
            </article>
          ))}
        </section>
      )}

      {!loading && recipes.length === 0 && ingredients.length === 0 && (
        <div className="empty-state">
          <p>Add some ingredients to get started!</p>
        </div>
      )}
    </main>
  );
}
