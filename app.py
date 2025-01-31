import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from scipy.sparse import load_npz
from rapidfuzz import process
import random
import json
from openai import OpenAI
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import ast

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI API Client
client = OpenAI(api_key=API_KEY)
# Kaggle dataset details
DATASET_NAME = "shuyangli94/food-com-recipes-and-user-interactions"
FILE_NAME = "RAW_recipes.csv"
DATA_PATH = "data"

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
recipes = pd.read_csv(dataset_file)

# Preprocess datasets
recipes.columns = ['recipe_name', 'recipe_code', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients','n_ingredients']
cosine_sim_top_k = load_npz('cosine_sim_top_k.npz')  # Precomputed similarity matrix

# Map recipe names to indices
recipe_idx = dict(zip(recipes['recipe_name'], list(recipes.index)))

# Function to find a recipe by title using RapidFuzz
def recipe_finder(title):
    all_titles = recipes['recipe_name'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# Function to refine recommendations using GPT
def filter_recommendations_with_gpt(input_recipe, recommendations):
    prompt = f"""
    Based on the input recipe '{input_recipe}', please select the 10 most relevant recipes from the list below:
    
    {', '.join(recommendations)}

    Return only a JSON array in this format: ["recipe1", "recipe2", ..., "recipe10"].
    Do not include any other text.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a recipe recommendation expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # Parse the response to get a list of recommended recipes
    try:
        gpt_recommendations = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        gpt_recommendations = recommendations[:10]  # Fallback to top 10 if parsing fails

    return gpt_recommendations

# ---------------------------------------------------------------------
# 1) METADATA ENDPOINT
# ---------------------------------------------------------------------
@app.route('/metadata', methods=['GET'])
def metadata():
    """
    Returns local dataset info:
      - recipe_name, minutes, macros, steps, and now ingredients
    """
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({"error": "Invalid or missing title"}), 400

    try:
        # Find closest recipe
        closest_recipe_name = recipe_finder(title)
        if closest_recipe_name not in recipe_idx:
            return jsonify({"error": f"Recipe '{title}' not found"}), 404

        idx = recipe_idx[closest_recipe_name]
        row = recipes.iloc[idx]

        # minutes -> plain int
        minutes_to_make = int(row['minutes'])

        # parse macros from 'nutrition' => 7 values
        # [calories, total_fat, sugar, sodium, protein, sat_fat, carbs]
        macro_list = ast.literal_eval(row['nutrition']) if isinstance(row['nutrition'], str) else []
        if len(macro_list) < 7:
            macros_dict = {"error": "Not enough nutrition info"}
        else:
            macros_dict = {
                "calories": float(macro_list[0]),
                "fat": f"{macro_list[1]}%",
                "sugar": f"{macro_list[2]}%",
                "sodium": f"{macro_list[3]}%",
                "protein": f"{macro_list[4]}%",
                "sat_fat": f"{macro_list[5]}%",
                "carbs": f"{macro_list[6]}%"
            }

        # parse steps
        steps_str = row['steps']
        try:
            steps_list = ast.literal_eval(steps_str)
            if not isinstance(steps_list, list):
                steps_list = [str(steps_str)]
        except:
            steps_list = [str(steps_str)]

        # parse ingredients
        ing_str = row['ingredients']
        try:
            ingredients_list = ast.literal_eval(ing_str)
            if not isinstance(ingredients_list, list):
                ingredients_list = [str(ing_str)]
        except:
            ingredients_list = [str(ing_str)]

        return jsonify({
            "recipe_name": capitalize_recipe_name(closest_recipe_name),
            "minutes": minutes_to_make,
            "macros": macros_dict,
            "steps": steps_list,
            "ingredients": ingredients_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------
# 2) COVER + RECOMMENDATIONS ENDPOINT
# ---------------------------------------------------------------------
@app.route('/cover_recs', methods=['GET'])
def cover_recs():
    """
    Returns:
      - cover_image (DALL·E 3 generated)
      - recommendations (GPT refined)
    This can take longer => front-end shows skeleton loaders 
    until these arrive.
    """
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({"error": "Invalid or missing title"}), 400

    try:
        closest_recipe_name = recipe_finder(title)
        if closest_recipe_name not in recipe_idx:
            return jsonify({"error": f"Recipe '{title}' not found"}), 404

        idx = recipe_idx[closest_recipe_name]

        # 1) Generate DALL·E
        cover_image_url = generate_dalle_image(closest_recipe_name)

        # 2) Similar recipes => GPT refine
        n_recommendations = 20
        sim_scores = cosine_sim_top_k[idx].toarray().flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        similar_rows = [recipes.iloc[i[0]] for i in sim_scores]

        formatted_titles = [
            capitalize_recipe_name(r['recipe_name']) 
            for r in similar_rows
        ]
        top10_gpt = filter_recommendations_with_gpt(title, formatted_titles)

        # Build final recs
        recs_out = []
        for rec_name in top10_gpt:
            match_df = recipes[recipes['recipe_name'] == rec_name.lower()]
            if not match_df.empty:
                row = match_df.iloc[0]
                recs_out.append({
                    "recipe_name": rec_name,
                    "tags": row['tags'] if pd.notna(row['tags']) else "No tags available"
                })
            else:
                recs_out.append({
                    "recipe_name": rec_name,
                    "tags": "Recipe information not found"
                })

        return jsonify({
            "cover_image": cover_image_url,
            "recommendations": recs_out
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_dalle_image(recipe_name):
    """
    Calls DALL·E 3 for a pastel clipart of the given recipe_name.
    Adjust to your model, prompt, size, or remove if no DALL·E 3 access.
    """
    try:
        prompt = f"Create a pastel clipart of a top-down view of a recipe called \"{recipe_name}\" in a 21:9 aspect ratio. No text."
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as ex:
        print("DALL·E 3 error:", ex)
        # fallback placeholder
        return "https://via.placeholder.com/1024?text=No+DALL-E+Access"

# Function to capitalize recipe names
def capitalize_recipe_name(name):
    words = name.split()
    capitalized_words = [word.capitalize() if not word.isdigit() else word for word in words]
    return " ".join(capitalized_words)
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """
    Returns top 5 closest matches for the given title (rapidfuzz).
    Faster, no GPT overhead—just returns possible recipe names.
    """
    title = request.args.get('title', '').strip()
    if not title:
        # If empty, return an empty list
        return jsonify([])
    
    # We'll search across all recipe titles
    all_titles = recipes['recipe_name'].tolist()
    # RapidFuzz: get top 5 matches (process.extract returns list of (match, score, index))
    matches = process.extract(title, all_titles, limit=5)

    # Just return the recipe names (i.e. match[0])
    results = [m[0] for m in matches]
    return jsonify(results)
if __name__ == '__main__':
    app.run(debug=True)
