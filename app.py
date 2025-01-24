from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from scipy.sparse import load_npz
from rapidfuzz import process
import random

app = Flask(__name__)
CORS(app)

# Load datasets
raw_interactions = pd.read_csv('RAW_interactions.csv')  # Update paths as needed
raw_recipes = pd.read_csv('RAW_recipes.csv')

# Preprocess datasets
recipes = raw_recipes[['name', 'id', 'tags']]
recipes.columns = ['recipe_name', 'recipe_code', 'tags']
cosine_sim_top_k = load_npz('cosine_sim_top_k.npz')  # Precomputed similarity matrix

# Map recipe names to indices
recipe_idx = dict(zip(recipes['recipe_name'], list(recipes.index)))

# Function to find a recipe by title using RapidFuzz
def recipe_finder(title):
    all_titles = recipes['recipe_name'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# Endpoint to fetch recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title', '')
    n_recommendations = 20  # Fetch 20 recommendations initially

    if not title:
        return jsonify({"error": "No title provided"}), 400

    # Find the closest matching recipe
    closest_recipe = recipe_finder(title)

    if closest_recipe not in recipe_idx:
        return jsonify({"error": f"Recipe '{title}' not found."}), 404

    idx = recipe_idx[closest_recipe]

    # Get similarity scores and top recommendations
    sim_scores = cosine_sim_top_k[idx].toarray().flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]

    # Get indices of similar recipes
    similar_recipes = [recipes.iloc[i[0]] for i in sim_scores]

    # Randomly select 10 from the top 20 recommendations
    random_recommendations = random.sample(similar_recipes, 10)

    def capitalize_recipe_name(name):
        words = name.split()
        capitalized_words = [word.capitalize() if not word.isdigit() else word for word in words]
        return " ".join(capitalized_words)

    # Update recommendations formatting:
    recommendations = [
        {
            "recipe_name": capitalize_recipe_name(row['recipe_name']),
            "tags": row['tags'] if not pd.isna(row['tags']) else "No tags available"
        }
        for row in random_recommendations
    ]

    closest_recipe = capitalize_recipe_name(closest_recipe)


    return jsonify({
        "input_recipe": closest_recipe,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
