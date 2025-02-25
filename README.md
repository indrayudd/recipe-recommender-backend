# Backend: Recipe Recommender (Flask + GPT + Kaggle Dataset)

## Overview

This repository hosts the Flask-based backend for the AI-powered Recipe Recommender. It leverages [Food.com's 230k recipe dataset from Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions), integrates GPT for natural language processing and recipe personalization, and can optionally generate cover images via DALL·E.

## Features

- **Food.com’s 230k Recipe Dataset**: Downloads and uses a large-scale dataset via Kaggle for metadata and step-by-step instructions.
- **GPT-Driven Recipe Enhancements**: Generates personalized recipe remixes, step clarifications, and more.
- **DALL·E Cover Generation (optional)**: Prompt-engineered to create dynamic cover images for each recipe.
- **RESTful Endpoints**: Exposes metadata, recommendations, and remix endpoints for seamless frontend consumption.

## Requirements

- Python 3.7+
- Kaggle API credentials set up (for initial dataset download)
- OpenAI API Key (for GPT and DALL·E usage, if desired)

## Environment Variables

Create a `.env` file or set these environment variables:

```ini
OPENAI_API_KEY=your-openai-key
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
```

(The Kaggle credentials are only needed if the dataset is not already downloaded.)

## Setup

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Verify the server is running at `http://127.0.0.1:5000`.

## API Endpoints

```http
GET /autocomplete?title=<query>
```
- Returns an array of up to 5 suggestions matching `<query>`.

```http
GET /metadata?title=<recipe>
```
- Returns JSON containing recipe name, ingredients, macros, steps, etc.

```http
GET /cover_recs?title=<recipe>
```
- Returns the DALL·E cover image (if GPT key is provided) and top recommendations.

```http
POST /remix
```
- Accepts JSON with `original` (recipe metadata) and `newIngredients`, then returns a GPT-generated remix.
