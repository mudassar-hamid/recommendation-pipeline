"""Generate recommendations and compare models with top-K ranking metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.common.io import read_csv, read_json, write_csv, write_json
from src.evaluation.metrics import precision_recall_ndcg_at_k, intra_list_diversity
from src.reporting.build_report import build_final_report


def _rank_popularity(user_id: int, seen: set[int], ranking: pd.DataFrame, k: int) -> list[tuple[int, float]]:
    candidates = ranking.loc[~ranking["item_id"].astype(int).isin(seen)].head(k)
    return [(int(row.item_id), float(row.popularity_score)) for row in candidates.itertuples()]


def _rank_cf(user_id: int, seen: set[int], train: pd.DataFrame, similarity,
             user_index: dict[int, int], item_index: dict[int, int],
             popularity: pd.DataFrame, k: int, threshold: float) -> list[tuple[int, float]]:
    if user_id not in user_index:
        return _rank_popularity(user_id, seen, popularity, k)
    history = train.loc[(train["user_id"].astype(int) == user_id) & (train["rating"] >= threshold), "item_id"].astype(int).unique()
    history_indices = [item_index[item] for item in history if item in item_index]
    if not history_indices:
        return _rank_popularity(user_id, seen, popularity, k)

    profile = np.zeros(similarity.shape[0], dtype=np.float32)
    profile[history_indices] = 1.0
    scores = np.asarray(similarity.dot(profile)).ravel()
    popularity_rank = dict(zip(popularity["item_id"].astype(int), popularity["popularity_rank"].astype(int)))
    candidates = [item for item in item_index if item not in seen]
    if not candidates:
        return []
    frame = pd.DataFrame({
        "item_id": candidates,
        "score": [float(scores[item_index[item]]) for item in candidates],
        "popularity_rank": [int(popularity_rank.get(item, 10**9)) for item in candidates],
    })
    # Similarity is primary; popularity provides a deterministic and useful tie-break/fallback.
    frame = frame.sort_values(["score", "popularity_rank", "item_id"], ascending=[False, True, True], kind="mergesort")
    return [(int(row.item_id), float(row.score)) for row in frame.head(k).itertuples()]


def _evaluate_one(name: str, recommendations: dict[int, list[int]], relevant: dict[int, set[int]],
                  item_genres: dict[int, set[str]], catalog_size: int, k: int) -> dict[str, float]:
    p_values, r_values, n_values, d_values = [], [], [], []
    recommended_union: set[int] = set()
    for user_id, rel in relevant.items():
        recs = recommendations.get(user_id, [])[:k]
        p, r, n = precision_recall_ndcg_at_k(recs, rel, k)
        p_values.append(p); r_values.append(r); n_values.append(n)
        d_values.append(intra_list_diversity(recs, item_genres))
        recommended_union.update(recs)
    return {
        "model": name,
        f"precision_at_{k}": float(np.mean(p_values)) if p_values else 0.0,
        f"recall_at_{k}": float(np.mean(r_values)) if r_values else 0.0,
        f"ndcg_at_{k}": float(np.mean(n_values)) if n_values else 0.0,
        "coverage": float(len(recommended_union) / catalog_size) if catalog_size else 0.0,
        "diversity": float(np.mean(d_values)) if d_values else 0.0,
        "users_evaluated": int(len(relevant)),
        "unique_items_recommended": int(len(recommended_union)),
        "catalog_size": int(catalog_size),
    }


def _predict_rating_for_user_item(
    user_id: int,
    item_id: int,
    user_history: dict[int, tuple[np.ndarray, np.ndarray]],
    item_index: dict[int, int],
    similarity,
    user_mean: dict[int, float],
    global_mean: float,
) -> float:
    """Predict a held-out rating from the user's training ratings and item similarities.

    Only training interactions are used as neighbors. If no similar rated item is
    available, fall back to the user's training mean, then the global training mean.
    """
    fallback = float(user_mean.get(user_id, global_mean))
    target_idx = item_index.get(item_id)
    history = user_history.get(user_id)
    if target_idx is None or history is None:
        return float(np.clip(fallback, 1.0, 5.0))

    rated_indices, ratings = history
    if len(rated_indices) == 0:
        return float(np.clip(fallback, 1.0, 5.0))

    similarities = similarity.getrow(target_idx)[:, rated_indices].toarray().ravel()
    # Similarity is cosine over positive interactions and should be non-negative.
    weights = np.maximum(similarities, 0.0)
    denominator = float(weights.sum())
    if denominator <= 1e-12:
        prediction = fallback
    else:
        prediction = float(np.dot(weights, ratings) / denominator)
    return float(np.clip(prediction, 1.0, 5.0))


def _rating_prediction_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    item_index: dict[int, int],
    similarity,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate explicit rating prediction on held-out test rows only."""
    train_users = set(train["user_id"].astype(int).unique())
    known_items = set(item_index)
    evaluable = test.loc[
        test["user_id"].astype(int).isin(train_users)
        & test["item_id"].astype(int).isin(known_items)
    ].copy()

    global_mean = float(train["rating"].mean())
    user_mean = train.groupby("user_id")["rating"].mean().to_dict()
    user_history = {}
    for user_id, group in train.groupby("user_id"):
        indices = group["item_id"].astype(int).map(item_index)
        valid = indices.notna()
        user_history[int(user_id)] = (
            indices.loc[valid].astype(int).to_numpy(),
            group.loc[valid, "rating"].astype(float).to_numpy(),
        )

    predictions = []
    for row in evaluable.itertuples(index=False):
        predictions.append(_predict_rating_for_user_item(
            int(row.user_id), int(row.item_id), user_history, item_index,
            similarity, user_mean, global_mean,
        ))

    evaluable["predicted_rating"] = predictions
    if len(evaluable):
        errors = evaluable["rating"].astype(float).to_numpy() - evaluable["predicted_rating"].to_numpy()
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(np.abs(errors)))
    else:
        rmse, mae = float("nan"), float("nan")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "rating_predictions_evaluated": int(len(evaluable)),
        "rating_test_interactions_excluded": int(len(test) - len(evaluable)),
    }
    return metrics, evaluable


def evaluate_models(config: dict) -> dict[str, str]:
    paths = config["paths"]
    k = int(config["evaluation"]["k"])
    threshold = float(config["data"]["positive_rating_threshold"])
    train = read_csv(paths["processed_dir"] / "train_interactions.csv", parse_dates=["timestamp"])
    test = read_csv(paths["processed_dir"] / "test_interactions.csv", parse_dates=["timestamp"])
    items = read_csv(paths["processed_dir"] / "items_clean.csv")
    popularity = read_csv(paths["models_dir"] / "popularity_model.csv")
    metadata = read_json(paths["models_dir"] / "item_item_model_metadata.json")
    similarity = load_npz(paths["models_dir"] / "item_item_similarity.npz").tocsr()

    train["timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
    train["user_id"] = train["user_id"].astype(int); train["item_id"] = train["item_id"].astype(int)
    test["user_id"] = test["user_id"].astype(int); test["item_id"] = test["item_id"].astype(int)

    catalog_items = set(train["item_id"].unique().astype(int).tolist())
    relevant_frame = test.loc[(test["rating"] >= threshold) & test["item_id"].isin(catalog_items)]
    relevant = {int(user): set(group["item_id"].astype(int).tolist())
                for user, group in relevant_frame.groupby("user_id")}
    train_users = set(train["user_id"].unique().astype(int).tolist())
    relevant = {user: rel for user, rel in relevant.items() if user in train_users and rel}
    if not relevant:
        raise ValueError("No evaluable positive test interactions remain after excluding cold-start items/users")

    seen_by_user = {int(user): set(group["item_id"].astype(int).tolist())
                    for user, group in train.groupby("user_id")}
    user_ids = sorted(relevant)
    user_index = {int(value): index for index, value in enumerate(metadata["user_ids"])}
    item_index = {int(value): index for index, value in enumerate(metadata["item_ids"])}
    if config.get("evaluation", {}).get("rating_prediction_enabled", True):
        rating_metrics, rating_predictions = _rating_prediction_metrics(train, test, item_index, similarity)
    else:
        rating_metrics = {
            "rmse": float("nan"), "mae": float("nan"),
            "rating_predictions_evaluated": 0,
            "rating_test_interactions_excluded": int(len(test)),
        }
        rating_predictions = pd.DataFrame(columns=list(test.columns) + ["predicted_rating"])
    rating_predictions_path = write_csv(
        rating_predictions,
        paths["reports_dir"] / "rating_predictions.csv",
    )
    popularity["item_id"] = popularity["item_id"].astype(int)
    popularity = popularity.sort_values("popularity_rank")

    pop_recommendations: dict[int, list[int]] = {}
    cf_recommendations: dict[int, list[int]] = {}
    pop_rows, cf_rows = [], []
    for user_id in user_ids:
        seen = seen_by_user.get(user_id, set())
        pop_ranked = _rank_popularity(user_id, seen, popularity, k)
        cf_ranked = _rank_cf(user_id, seen, train, similarity, user_index, item_index,
                             popularity, k, threshold)
        pop_recommendations[user_id] = [item for item, _ in pop_ranked]
        cf_recommendations[user_id] = [item for item, _ in cf_ranked]
        pop_rows.extend([{"user_id": user_id, "item_id": item, "rank": rank, "score": score}
                         for rank, (item, score) in enumerate(pop_ranked, 1)])
        cf_rows.extend([{"user_id": user_id, "item_id": item, "rank": rank, "score": score}
                        for rank, (item, score) in enumerate(cf_ranked, 1)])

    item_genres = {}
    for row in items.itertuples():
        raw_genres = str(row.genres) if pd.notna(row.genres) else ""
        item_genres[int(row.item_id)] = {genre.strip() for genre in raw_genres.split("|") if genre.strip() and genre != "(no genres listed)"}

    pop_metrics = _evaluate_one("Popularity baseline", pop_recommendations, relevant, item_genres, len(catalog_items), k)
    cf_metrics = _evaluate_one("Item-item collaborative filtering", cf_recommendations, relevant, item_genres, len(catalog_items), k)
    # The popularity ranker does not estimate a user's star rating, so RMSE/MAE
    # are intentionally not applicable to that model.
    pop_metrics.update({"rmse": None, "mae": None, "rating_predictions_evaluated": 0})
    cf_metrics.update({
        "rmse": rating_metrics["rmse"],
        "mae": rating_metrics["mae"],
        "rating_predictions_evaluated": rating_metrics["rating_predictions_evaluated"],
    })
    comparison = pd.DataFrame([pop_metrics, cf_metrics])
    comparison_path = write_csv(comparison, paths["reports_dir"] / "model_comparison.csv")
    pop_rec_path = write_csv(pd.DataFrame(pop_rows, columns=["user_id", "item_id", "rank", "score"]), paths["models_dir"] / "recommendations_popularity.csv")
    cf_rec_path = write_csv(pd.DataFrame(cf_rows, columns=["user_id", "item_id", "rank", "score"]), paths["models_dir"] / "recommendations_personalized.csv")

    positive_test = test.loc[test["rating"] >= threshold]
    total_positive_test = int(len(positive_test))
    evaluable_mask = positive_test["item_id"].isin(catalog_items) & positive_test["user_id"].isin(train_users)
    evaluable_positive_test = int(evaluable_mask.sum())
    positive_test_item_cold_start = int((~positive_test["item_id"].isin(catalog_items)).sum())
    positive_test_user_cold_start = int((~positive_test["user_id"].isin(train_users)).sum())
    summary = {
        "k": k,
        "positive_rating_threshold": threshold,
        "test_interactions": int(len(test)),
        "positive_test_interactions": total_positive_test,
        "evaluable_positive_test_interactions": evaluable_positive_test,
        "cold_start_positive_test_interactions_excluded": positive_test_item_cold_start,
        "positive_test_interactions_for_users_without_training_history": positive_test_user_cold_start,
        "positive_test_interactions_excluded_total": int(total_positive_test - evaluable_positive_test),
        "users_evaluated": int(len(relevant)),
        "catalog_items": int(len(catalog_items)),
        "metrics": [pop_metrics, cf_metrics],
        "rating_prediction_metrics": rating_metrics,
        "rating_prediction_method": "Similarity-weighted average of the user's training ratings for items similar to the held-out item; user-mean/global-mean fallback; predictions clipped to [1, 5].",
        "rating_prediction_evaluation_note": "RMSE and MAE are computed only on held-out test ratings for users and items represented in the training data. Test ratings are never used as neighbors.",
        "diversity_definition": "Mean per-user intra-list Jaccard distance over movie genre sets; empty genre sets have zero similarity.",
        "evaluation_note": "Macro-averaged over users with at least one positive held-out rating for an item present in the training catalog. Recommendations exclude items already seen in that user's training history.",
    }
    summary_path = write_json(summary, paths["reports_dir"] / "evaluation_summary.json")

    if config.get("mlflow", {}).get("enabled", True):
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow is enabled but missing. Install dependencies with: pip install -r requirements.txt") from exc
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"].get("experiment_name", "MovieLens-Recommendation-Pipeline"))
        for model_metrics in [pop_metrics, cf_metrics]:
            model_name = model_metrics["model"].lower().replace(" ", "_")
            # Only pass numeric evaluation metrics to MLflow. Filter out
            # descriptive fields such as "model" before attempting conversion.
            metric_keys = {
                key for key in model_metrics
                if key.startswith(("precision_at_", "recall_at_", "ndcg_at_"))
                or key in {"coverage", "diversity", "rmse", "mae"}
            }

            metric_values = {}
            for key in metric_keys:
                value = model_metrics.get(key)
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue

                if np.isfinite(numeric_value):
                    metric_values[key] = numeric_value
            rec_path = pop_rec_path if "popularity" in model_name else cf_rec_path
            with mlflow.start_run(run_name=f"evaluation-{model_name}"):
                mlflow.log_params({"model": model_metrics["model"], "k": k,
                                   "users_evaluated": len(relevant), "positive_rating_threshold": threshold})
                mlflow.log_metrics(metric_values)
                artifacts = [str(comparison_path), str(summary_path), str(rec_path)]
                if "item_item" in model_name:
                    artifacts.append(str(rating_predictions_path))
                for artifact in artifacts:
                    mlflow.log_artifact(artifact)

    if config.get("report", {}).get("generate_pdf", True):
        build_final_report(config, summary)
    print("Model evaluation complete. Results:")
    print(comparison.to_string(index=False))
    return {"comparison": str(comparison_path), "summary": str(summary_path),
            "rating_predictions": str(rating_predictions_path),
            "popularity_recommendations": str(pop_rec_path), "personalized_recommendations": str(cf_rec_path)}
