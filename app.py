import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from scipy.sparse import load_npz
from rapidfuzz import process
import random
from kaggle.api.kaggle_api_extended import KaggleApi

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Kaggle dataset details
DATASET_NAME = "shuyangli94/food-com-recipes-and-user-interactions"
FILE_NAME = "RAW_recipes.csv"
DATA_PATH = "data"  # Folder for datasets

# Ensure the Kaggle dataset is downloaded
def ensure_dataset():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    dataset_file = os.path.join(DATA_PATH, FILE_NAME)
    if not os.path.exists(dataset_file):
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(DATASET_NAME, FILE_NAME, path=DATA_PATH)
        import zipfile
        with zipfile.ZipFile(os.path.join(DATA_PATH, FILE_NAME + ".zip"), 'r') as zip_ref:
            zip_ref.extractall(DATA_PATH)
        os.remove(os.path.join(DATA_PATH, FILE_NAME + ".zip"))
    return dataset_file

# Load datasets
dataset_file = ensure_dataset()
raw_recipes = pd.read_csv(dataset_file)

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

    # Ensure title is a string
    if not isinstance(title, str) or not title.strip():
        return jsonify({"error": "Invalid or missing title"}), 400

    title = title.strip()  # Remove any surrounding whitespace
    n_recommendations = 20  # Fetch 20 recommendations initially

    try:
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

        # Format recommendations:
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
