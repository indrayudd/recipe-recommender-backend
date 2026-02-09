import argparse
import ast
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
    distribution_summary: dict


NUMERIC_RELATIVE_THRESHOLD = 0.2
TAG_PREVALENCE_RATIO_THRESHOLD = 2.0


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


def parse_tags(tags_value) -> list:
    if isinstance(tags_value, list):
        return tags_value
    if isinstance(tags_value, str):
        try:
            parsed = ast.literal_eval(tags_value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def compute_numeric_stats(series: pd.Series) -> dict:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(cleaned.mean()),
        "median": float(cleaned.median()),
        "p10": float(cleaned.quantile(0.1)),
        "p90": float(cleaned.quantile(0.9)),
    }


def compare_numeric_stats(sample_stats: dict, baseline_stats: dict) -> dict:
    deltas = {}
    warnings = []
    for key in ("mean", "median", "p10", "p90"):
        sample_value = sample_stats.get(key)
        baseline_value = baseline_stats.get(key)
        if sample_value is None or baseline_value in (None, 0):
            deltas[key] = {"sample": sample_value, "baseline": baseline_value, "delta": None, "relative_delta": None}
            continue
        delta = sample_value - baseline_value
        relative_delta = delta / baseline_value
        deltas[key] = {
            "sample": sample_value,
            "baseline": baseline_value,
            "delta": delta,
            "relative_delta": relative_delta,
        }
        if abs(relative_delta) >= NUMERIC_RELATIVE_THRESHOLD:
            warnings.append(
                f"{key} relative delta {relative_delta:.2%} exceeds {NUMERIC_RELATIVE_THRESHOLD:.0%} threshold"
            )
    return {"deltas": deltas, "warnings": warnings}


def compute_tag_prevalence(recipes_subset: pd.DataFrame) -> dict:
    tag_counts = {}
    total_recipes = len(recipes_subset)
    for tags_value in recipes_subset["tags"].tolist():
        tags = set(parse_tags(tags_value))
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if total_recipes == 0:
        return {"total_recipes": 0, "prevalence": {}}
    prevalence = {tag: count / total_recipes for tag, count in tag_counts.items()}
    return {"total_recipes": total_recipes, "prevalence": prevalence}


def compare_tag_prevalence(sample_prev: dict, baseline_prev: dict, ratio_threshold: float) -> dict:
    sample_prevalence = sample_prev.get("prevalence", {})
    baseline_prevalence = baseline_prev.get("prevalence", {})
    tags = set(sample_prevalence) | set(baseline_prevalence)
    lower_ratio = 1 / ratio_threshold if ratio_threshold else 0
    deviations = {}
    warnings = []
    flagged_tags = []
    for tag in sorted(tags):
        sample_value = sample_prevalence.get(tag, 0.0)
        baseline_value = baseline_prevalence.get(tag, 0.0)
        ratio = None
        if baseline_value > 0:
            ratio = sample_value / baseline_value
        delta = sample_value - baseline_value
        should_flag = False
        if baseline_value == 0:
            should_flag = sample_value > 0
        elif ratio_threshold and (ratio >= ratio_threshold or ratio <= lower_ratio):
            should_flag = True
        if should_flag:
            deviations[tag] = {
                "sample": sample_value,
                "baseline": baseline_value,
                "delta": delta,
                "ratio": ratio,
            }
            flagged_tags.append(tag)
            if baseline_value == 0:
                warnings.append(f"tag '{tag}' appears in recommendations but not in corpus")
            else:
                warnings.append(
                    f"tag '{tag}' prevalence ratio {ratio:.2f} outside {lower_ratio:.2f}-{ratio_threshold:.2f}"
                )
    return {
        "threshold_ratio": ratio_threshold,
        "lower_ratio": lower_ratio,
        "baseline_prevalence": baseline_prevalence,
        "sample_prevalence": sample_prevalence,
        "flagged_tags": flagged_tags,
        "deviations": deviations,
        "warnings": warnings,
    }


def build_distribution_summary(
    recipes: pd.DataFrame, recommendation_indices: list, tag_ratio_threshold: float
) -> dict:
    if not recommendation_indices:
        return {
            "minutes": {"deltas": {}, "warnings": ["no recommendations available to analyze"]},
            "n_ingredients": {"deltas": {}, "warnings": ["no recommendations available to analyze"]},
            "tags": {
                "threshold_ratio": tag_ratio_threshold,
                "lower_ratio": 1 / tag_ratio_threshold if tag_ratio_threshold else 0,
                "baseline_prevalence": {},
                "sample_prevalence": {},
                "flagged_tags": [],
                "deviations": {},
                "warnings": ["no recommendations available to analyze"],
            },
        }

    baseline_minutes = compute_numeric_stats(recipes["minutes"])
    baseline_ingredients = compute_numeric_stats(recipes["n_ingredients"])
    recommended_recipes = recipes.iloc[recommendation_indices]
    sample_minutes = compute_numeric_stats(recommended_recipes["minutes"])
    sample_ingredients = compute_numeric_stats(recommended_recipes["n_ingredients"])

    minutes_comparison = compare_numeric_stats(sample_minutes, baseline_minutes)
    ingredients_comparison = compare_numeric_stats(sample_ingredients, baseline_ingredients)

    baseline_tags = compute_tag_prevalence(recipes)
    sample_tags = compute_tag_prevalence(recommended_recipes)
    tag_comparison = compare_tag_prevalence(sample_tags, baseline_tags, tag_ratio_threshold)

    return {
        "minutes": minutes_comparison,
        "n_ingredients": ingredients_comparison,
        "tags": tag_comparison,
    }


def evaluate_rankings(
    recipes: pd.DataFrame,
    similarity_matrix,
    sample_size: int,
    top_k: int,
    seed: int,
    tag_ratio_threshold: float,
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

    distribution_summary = build_distribution_summary(recipes, all_recommendations, tag_ratio_threshold)

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
        distribution_summary=distribution_summary,
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
        "--tag-prevalence-threshold",
        type=float,
        default=TAG_PREVALENCE_RATIO_THRESHOLD,
        help="Flag tag prevalence ratios outside [1/x, x] in recommendations vs. corpus",
    )
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
        args.tag_prevalence_threshold,
        dataset_path,
        args.matrix_path,
    )

    print("Offline evaluation summary:")
    for key, value in asdict(summary).items():
        if key == "sample_examples":
            print(f"{key}: {len(value)} examples")
        else:
            print(f"{key}: {value}")

    tag_warnings = summary.distribution_summary.get("tags", {}).get("warnings", [])
    if tag_warnings:
        print("Tag prevalence warnings:")
        for warning in tag_warnings:
            print(f"- {warning}")

    if args.report_path is not None:
        write_report(args.report_path, summary)
        print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
