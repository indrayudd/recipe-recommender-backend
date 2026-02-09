import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


DATASET_NAME = "shuyangli94/food-com-recipes-and-user-interactions"
FILE_NAME = "RAW_recipes.csv"
DATASET_DIR = "data"
DEFAULT_DATASET_PATH = os.path.join(DATASET_DIR, FILE_NAME)
DEFAULT_MATRIX_PATH = "cosine_sim_top_k.npz"
DEFAULT_REPORT_PATH = os.path.join("eval", "results", "latest.json")


@dataclass
class EvalSummary:
    dataset_path: str
    matrix_path: str
    total_recipes: int
    matrix_shape: tuple
    sample_size: int
    top_k: int
    average_recommendations: float
    unique_recommendations: int
    missing_recommendations: int
    timestamp_utc: str
    sample_examples: list


def load_recipes(dataset_path: str) -> pd.DataFrame:
    recipes = pd.read_csv(dataset_path)
    recipes.columns = [
        "recipe_name",
        "recipe_code",
        "minutes",
        "contributor_id",
        "submitted",
        "tags",
        "nutrition",
        "n_steps",
        "steps",
        "description",
        "ingredients",
        "n_ingredients",
    ]
    return recipes


def ensure_dataset(dataset_path: str) -> str:
    dataset_dir = os.path.dirname(dataset_path) or "."
    os.makedirs(dataset_dir, exist_ok=True)
    if os.path.exists(dataset_path):
        return dataset_path

    print("Dataset not found locally. Downloading from Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(DATASET_NAME, FILE_NAME, path=dataset_dir)
    import zipfile

    zip_path = os.path.join(dataset_dir, FILE_NAME + ".zip")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    os.remove(zip_path)
    return dataset_path


def compute_top_k(sim_row: np.ndarray, top_k: int, skip_index: int) -> list:
    top_k = min(top_k, sim_row.size - 1)
    sorted_indices = np.argsort(sim_row)[::-1]
    filtered = [idx for idx in sorted_indices if idx != skip_index]
    return filtered[:top_k]


def evaluate_rankings(
    recipes: pd.DataFrame,
    similarity_matrix,
    sample_size: int,
    top_k: int,
    seed: int,
    dataset_path: str,
    matrix_path: str,
) -> EvalSummary:
    rng = np.random.default_rng(seed)
    total_recipes = len(recipes)
    sample_size = min(sample_size, total_recipes)
    sampled_indices = rng.choice(total_recipes, size=sample_size, replace=False)

    all_recommendations = []
    sample_examples = []
    missing_recommendations = 0

    for idx in sampled_indices:
        sim_row = similarity_matrix[idx].toarray().flatten()
        top_indices = compute_top_k(sim_row, top_k, idx)
        if not top_indices:
            missing_recommendations += 1
            continue
        all_recommendations.extend(top_indices)

        if len(sample_examples) < 5:
            input_name = recipes.iloc[idx]["recipe_name"]
            rec_names = recipes.iloc[top_indices]["recipe_name"].tolist()
            sample_examples.append(
                {
                    "input_recipe": input_name,
                    "recommendations": rec_names,
                }
            )

    average_recommendations = 0.0
    if sample_size - missing_recommendations > 0:
        average_recommendations = len(all_recommendations) / (sample_size - missing_recommendations)

    return EvalSummary(
        dataset_path=dataset_path,
        matrix_path=matrix_path,
        total_recipes=total_recipes,
        matrix_shape=tuple(similarity_matrix.shape),
        sample_size=sample_size,
        top_k=top_k,
        average_recommendations=average_recommendations,
        unique_recommendations=len(set(all_recommendations)),
        missing_recommendations=missing_recommendations,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        sample_examples=sample_examples,
    )


def write_report(report_path: str, summary: EvalSummary) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(asdict(summary), report_file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluator for recipe recommendation rankings.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Path to RAW_recipes.csv")
    parser.add_argument("--matrix-path", default=DEFAULT_MATRIX_PATH, help="Path to cosine_sim_top_k.npz")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of recipes to sample")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to inspect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--report-path",
        nargs="?",
        const=DEFAULT_REPORT_PATH,
        default=None,
        help=f"Optional JSON report path (default: {DEFAULT_REPORT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = ensure_dataset(args.dataset_path)
    if not os.path.exists(args.matrix_path):
        raise FileNotFoundError(f"Similarity matrix not found at {args.matrix_path}")

    recipes = load_recipes(dataset_path)
    similarity_matrix = load_npz(args.matrix_path)

    if similarity_matrix.shape[0] != len(recipes):
        raise ValueError(
            "Mismatch between similarity matrix rows "
            f"({similarity_matrix.shape[0]}) and recipes ({len(recipes)})."
        )

    summary = evaluate_rankings(
        recipes,
        similarity_matrix,
        args.sample_size,
        args.top_k,
        args.seed,
        dataset_path,
        args.matrix_path,
    )

    print("Offline evaluation summary:")
    for key, value in asdict(summary).items():
        if key == "sample_examples":
            print(f"{key}: {len(value)} examples")
        else:
            print(f"{key}: {value}")

    if args.report_path is not None:
        write_report(args.report_path, summary)
        print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
