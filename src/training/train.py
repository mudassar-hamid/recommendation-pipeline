"""Train a popularity baseline and an item-item collaborative filtering model."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

from src.common.io import read_csv, write_csv, write_json


def _mlflow_start(config: dict, run_name: str, params: dict, artifacts: list[str]) -> None:
    if not config.get("mlflow", {}).get("enabled", True):
        return
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is enabled but missing. Install dependencies with: pip install -r requirements.txt") from exc
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"].get("experiment_name", "MovieLens-Recommendation-Pipeline"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for artifact in artifacts:
            mlflow.log_artifact(artifact)


def train_popularity_model(config: dict) -> dict[str, str]:
    paths = config["paths"]
    popularity = read_csv(paths["processed_dir"] / "item_popularity.csv")
    items = read_csv(paths["processed_dir"] / "items_clean.csv")
    ranked = popularity.merge(items[["item_id", "title", "genres"]], on="item_id", how="left")
    ranked = ranked.sort_values(
        ["positive_interaction_count", "interaction_count", "item_avg_rating", "item_id"],
        ascending=[False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    ranked["popularity_score"] = ranked["positive_interaction_count"].astype(float)
    ranked["popularity_rank"] = np.arange(1, len(ranked) + 1)
    model_path = write_csv(ranked, paths["models_dir"] / "popularity_model.csv")

    train = read_csv(paths["processed_dir"] / "train_interactions.csv")
    params = {
        "model_type": "popularity_baseline",
        "ranking": "positive_interaction_count, interaction_count, average_rating",
        "positive_rating_threshold": float(config["data"]["positive_rating_threshold"]),
        "training_interactions": int(len(train)),
        "catalog_items_with_training_interactions": int(len(ranked)),
    }
    _mlflow_start(config, "popularity-baseline-training", params, [str(model_path)])
    print(f"Trained popularity baseline for {len(ranked):,} items.")
    return {"model": str(model_path)}


def _prune_similarity(similarity: csr_matrix, max_neighbors: int) -> csr_matrix:
    """Keep at most max_neighbors strongest non-self neighbors for each item."""
    similarity = similarity.tocsr()
    data_parts, index_parts, indptr = [], [], [0]
    for row_index in range(similarity.shape[0]):
        start, end = similarity.indptr[row_index], similarity.indptr[row_index + 1]
        indices = similarity.indices[start:end]
        values = similarity.data[start:end]
        keep_mask = indices != row_index
        indices, values = indices[keep_mask], values[keep_mask]
        if len(values) > max_neighbors:
            chosen = np.argpartition(values, -max_neighbors)[-max_neighbors:]
            indices, values = indices[chosen], values[chosen]
        if len(values):
            order = np.argsort(values)[::-1]
            indices, values = indices[order], values[order]
        index_parts.extend(indices.tolist())
        data_parts.extend(values.tolist())
        indptr.append(len(index_parts))
    return csr_matrix(
        (np.asarray(data_parts, dtype=np.float32), np.asarray(index_parts, dtype=np.int32),
         np.asarray(indptr, dtype=np.int32)),
        shape=similarity.shape,
    )


def train_personalized_model(config: dict) -> dict[str, str]:
    paths = config["paths"]
    train = read_csv(paths["processed_dir"] / "train_interactions.csv")
    threshold = float(config["data"]["positive_rating_threshold"])
    max_neighbors = int(config["evaluation"].get("item_similarity_neighbors", 100))
    if max_neighbors < 1:
        raise ValueError("evaluation.item_similarity_neighbors must be >= 1")
    user_ids = sorted(train["user_id"].astype(int).unique().tolist())
    item_ids = sorted(train["item_id"].astype(int).unique().tolist())
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}

    positive = train.loc[train["rating"] >= threshold, ["user_id", "item_id"]]
    row = positive["user_id"].astype(int).map(user_to_index).to_numpy()
    col = positive["item_id"].astype(int).map(item_to_index).to_numpy()
    values = np.ones(len(positive), dtype=np.float32)
    user_item = csr_matrix((values, (row, col)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    user_item.sum_duplicates()
    if user_item.nnz:
        user_item.data[:] = 1.0

    item_user = user_item.transpose().tocsr()
    similarity = cosine_similarity(item_user, dense_output=False).tocsr()
    similarity = _prune_similarity(similarity, max_neighbors)

    similarity_path = paths["models_dir"] / "item_item_similarity.npz"
    save_npz(similarity_path, similarity)
    metadata_path = write_json({
        "algorithm": "item-item collaborative filtering",
        "similarity": "cosine over binary positive-rating user-item matrix",
        "rating_prediction": "similarity-weighted average of explicit ratings from the user's training history",
        "rating_prediction_fallback": "user training mean, then global training mean",
        "positive_rating_threshold": threshold,
        "max_neighbors_per_item": max_neighbors,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "positive_interactions": int(user_item.nnz),
    }, paths["models_dir"] / "item_item_model_metadata.json")

    params = {
        "model_type": "item_item_collaborative_filtering",
        "similarity": "cosine",
        "feedback": "binary positive ratings for similarity; explicit training ratings for rating prediction",
        "rating_prediction": "similarity-weighted neighbor rating; held-out ratings used only during evaluation",
        "positive_rating_threshold": threshold,
        "max_neighbors_per_item": max_neighbors,
        "training_users": len(user_ids),
        "training_items": len(item_ids),
        "positive_interactions": int(user_item.nnz),
    }
    _mlflow_start(config, "item-item-collaborative-filtering-training", params,
                  [str(similarity_path), str(metadata_path)])
    print(f"Trained item-item CF model: {len(item_ids):,} items, {similarity.nnz:,} retained similarities.")
    return {"similarity_matrix": str(similarity_path), "metadata": str(metadata_path)}
