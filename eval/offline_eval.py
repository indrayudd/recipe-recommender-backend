import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from scipy.sparse import load_npz

DATASET_NAME = "shuyangli94/food-com-recipes-and-user-interactions"
FILE_NAME = "RAW_recipes.csv"
DATA_PATH = "data"
SIMILARITY_PATH = "cosine_sim_top_k.npz"


def ensure_dataset(allow_download: bool) -> str:
    dataset_file = os.path.join(DATA_PATH, FILE_NAME)
    if os.path.exists(dataset_file):
        return dataset_file
    if not allow_download:
        raise FileNotFoundError(
            f"Missing dataset at {dataset_file}. Run with --allow-download to fetch from Kaggle."
        )
    os.makedirs(DATA_PATH, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(DATASET_NAME, FILE_NAME, path=DATA_PATH)
    import zipfile

    with zipfile.ZipFile(os.path.join(DATA_PATH, FILE_NAME + ".zip"), "r") as zip_ref:
        zip_ref.extractall(DATA_PATH)
    os.remove(os.path.join(DATA_PATH, FILE_NAME + ".zip"))
    return dataset_file


def load_recipes(dataset_file: str) -> pd.DataFrame:
    recipes = pd.read_csv(dataset_file)
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


def parse_tags(tags_value) -> list[str]:
    if pd.isna(tags_value):
        return []
    if isinstance(tags_value, list):
        return tags_value
    try:
        parsed = json.loads(tags_value)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    cleaned = str(tags_value).strip("[]")
    if not cleaned:
        return []
    return [tag.strip().strip("'\"") for tag in cleaned.split(",") if tag.strip()]


def compute_recommendations(sim_matrix, index: int, top_k: int) -> list[tuple[int, float]]:
    scores = sim_matrix[index].toarray().flatten()
    indexed = list(enumerate(scores))
    indexed_sorted = sorted(indexed, key=lambda x: x[1], reverse=True)
    filtered = [pair for pair in indexed_sorted if pair[0] != index]
    return filtered[:top_k]


def distribution_summary(values: list[float]) -> dict:
    series = pd.Series(values)
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.9)),
    }


def compare_distribution(name: str, baseline: dict, recommended: dict, threshold: float) -> dict:
    deltas = {}
    for key in ("mean", "median", "p90"):
        base_value = baseline.get(key)
        rec_value = recommended.get(key)
        if base_value is None or base_value == 0:
            delta_ratio = None
        else:
            delta_ratio = rec_value / base_value
        deltas[key] = {
            "baseline": base_value,
            "recommended": rec_value,
            "ratio": delta_ratio,
            "flagged": delta_ratio is not None and (delta_ratio > threshold or delta_ratio < 1 / threshold),
        }
    return {"metric": name, "deltas": deltas}


def collect_tag_counts(recipes: pd.DataFrame, indices: list[int]) -> Counter:
    counter = Counter()
    for idx in indices:
        counter.update(parse_tags(recipes.loc[idx, "tags"]))
    return counter


def summarize_bias(
    base_counts: Counter,
    rec_counts: Counter,
    min_baseline: int,
    threshold: float,
) -> dict:
    flagged = []
    for tag, base_count in base_counts.items():
        if base_count < min_baseline:
            continue
        rec_count = rec_counts.get(tag, 0)
        ratio = rec_count / base_count if base_count else None
        if ratio is None:
            continue
        if ratio > threshold or ratio < 1 / threshold:
            flagged.append(
                {
                    "tag": tag,
                    "baseline_count": base_count,
                    "recommended_count": rec_count,
                    "ratio": ratio,
                }
            )
    flagged_sorted = sorted(flagged, key=lambda item: abs(1 - item["ratio"]), reverse=True)
    return {"flagged_tags": flagged_sorted}


def build_failures(
    recipes: pd.DataFrame,
    sim_matrix,
    indices: list[int],
    top_k: int,
    deviation_threshold: float,
) -> list[dict]:
    failures = []
    minutes_base = recipes["minutes"].dropna().astype(float).mean()
    for idx in indices:
        recs = compute_recommendations(sim_matrix, idx, top_k)
        if not recs:
            continue
        scores = [score for _, score in recs]
        score_gap = scores[0] - scores[-1]
        rec_minutes = recipes.loc[[r[0] for r in recs], "minutes"].dropna().astype(float)
        rec_mean = float(rec_minutes.mean()) if not rec_minutes.empty else None
        deviation_ratio = rec_mean / minutes_base if minutes_base else None
        if deviation_ratio is None:
            continue
        if deviation_ratio > deviation_threshold or deviation_ratio < 1 / deviation_threshold or score_gap < 0.05:
            failures.append(
                {
                    "query_recipe": recipes.loc[idx, "recipe_name"],
                    "query_index": int(idx),
                    "score_gap": float(score_gap),
                    "recommended": [
                        {
                            "recipe_name": recipes.loc[rec_idx, "recipe_name"],
                            "score": float(score),
                            "minutes": float(recipes.loc[rec_idx, "minutes"]),
                            "n_ingredients": int(recipes.loc[rec_idx, "n_ingredients"]),
                        }
                        for rec_idx, score in recs
                    ],
                    "recommended_minutes_mean": rec_mean,
                    "minutes_deviation_ratio": deviation_ratio,
                }
            )
    return failures


def run_eval(args: argparse.Namespace) -> dict:
    dataset_file = ensure_dataset(args.allow_download)
    recipes = load_recipes(dataset_file)
    if not os.path.exists(SIMILARITY_PATH):
        raise FileNotFoundError(f"Missing similarity matrix at {SIMILARITY_PATH}.")
    sim_matrix = load_npz(SIMILARITY_PATH)

    sample = recipes.sample(n=args.sample_size, random_state=args.seed).index.tolist()
    all_recommendation_indices = []
    per_query_recs = {}
    for idx in sample:
        recs = compute_recommendations(sim_matrix, idx, args.top_k)
        per_query_recs[int(idx)] = recs
        all_recommendation_indices.extend([rec_idx for rec_idx, _ in recs])

    minutes_base = distribution_summary(recipes["minutes"].dropna().astype(float).tolist())
    minutes_rec = distribution_summary(
        recipes.loc[all_recommendation_indices, "minutes"].dropna().astype(float).tolist()
    )
    minutes_check = compare_distribution("minutes", minutes_base, minutes_rec, args.distribution_threshold)

    ingredients_base = distribution_summary(
        recipes["n_ingredients"].dropna().astype(float).tolist()
    )
    ingredients_rec = distribution_summary(
        recipes.loc[all_recommendation_indices, "n_ingredients"].dropna().astype(float).tolist()
    )
    ingredients_check = compare_distribution(
        "n_ingredients", ingredients_base, ingredients_rec, args.distribution_threshold
    )

    base_tag_counts = collect_tag_counts(recipes, recipes.index.tolist())
    rec_tag_counts = collect_tag_counts(recipes, all_recommendation_indices)
    bias_check = summarize_bias(base_tag_counts, rec_tag_counts, args.min_baseline_tags, args.bias_threshold)

    failures = build_failures(
        recipes,
        sim_matrix,
        sample,
        args.top_k,
        args.distribution_threshold,
    )

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_size": args.sample_size,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "distribution_checks": [minutes_check, ingredients_check],
        "bias_checks": bias_check,
        "failure_cases": {"count": len(failures)},
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    if args.failures_dir:
        os.makedirs(args.failures_dir, exist_ok=True)
        failure_path = os.path.join(args.failures_dir, "failures.json")
        with open(failure_path, "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation sanity checks for ranking quality.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distribution-threshold", type=float, default=1.5)
    parser.add_argument("--bias-threshold", type=float, default=2.0)
    parser.add_argument("--min-baseline-tags", type=int, default=50)
    parser.add_argument("--output", type=str, default="eval/results/latest.json")
    parser.add_argument("--failures-dir", type=str, default="eval/results/failures")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_eval(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
