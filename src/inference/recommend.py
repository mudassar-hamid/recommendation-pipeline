"""Load persisted artifacts and generate top-K recommendations for one user."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.config import load_config
from src.common.io import read_csv, read_json


def recommend_user(user_id: int, config: dict, model_type: str = "personalized", k: int | None = None) -> pd.DataFrame:
    """Return recommendations for a user using previously saved training artifacts.

    Unknown users and users with no positive training interactions fall back to popularity.
    Items already present in the user's training history are excluded.
    """
    model_type = model_type.lower()
    if model_type not in {"personalized", "popularity"}:
        raise ValueError("model_type must be 'personalized' or 'popularity'")
    k = int(k or config["evaluation"]["k"])
    if k < 1:
        raise ValueError("k must be >= 1")

    paths = config["paths"]
    train = read_csv(paths["processed_dir"] / "train_interactions.csv")
    popularity = read_csv(paths["models_dir"] / "popularity_model.csv").sort_values("popularity_rank")
    # Reuse persisted feature tables at prediction time (profile context + item metadata/popularity).
    user_features = read_csv(paths["features_dir"] / "user_features.csv")
    item_features = read_csv(paths["features_dir"] / "item_features.csv")
    user_row = user_features.loc[user_features["user_id"].astype(int) == int(user_id)]
    preferred_category = str(user_row.iloc[0]["preferred_category"]) if not user_row.empty else "Unknown"
    item_columns = [c for c in ["item_id", "title", "genres", "item_avg_rating", "item_interaction_count"]
                    if c in item_features.columns]
    seen = set(train.loc[train["user_id"].astype(int) == int(user_id), "item_id"].astype(int).tolist())
    fallback = False

    if model_type == "personalized":
        metadata = read_json(paths["models_dir"] / "item_item_model_metadata.json")
        user_index = {int(value): index for index, value in enumerate(metadata["user_ids"])}
        item_index = {int(value): index for index, value in enumerate(metadata["item_ids"])}
        threshold = float(config["data"]["positive_rating_threshold"])
        history = train.loc[(train["user_id"].astype(int) == int(user_id)) &
                            (train["rating"] >= threshold), "item_id"].astype(int).unique()
        history_indices = [item_index[item] for item in history if item in item_index]
        if int(user_id) not in user_index or not history_indices:
            fallback = True
        else:
            similarity = load_npz(paths["models_dir"] / "item_item_similarity.npz").tocsr()
            profile = np.zeros(similarity.shape[0], dtype=np.float32)
            profile[history_indices] = 1.0
            scores = np.asarray(similarity.dot(profile)).ravel()
            pop_rank = dict(zip(popularity["item_id"].astype(int), popularity["popularity_rank"].astype(int)))
            candidates = [int(item) for item in metadata["item_ids"] if int(item) not in seen]
            frame = pd.DataFrame({
                "item_id": candidates,
                "score": [float(scores[item_index[item]]) for item in candidates],
                "popularity_rank": [int(pop_rank.get(item, 10**9)) for item in candidates],
            }).sort_values(["score", "popularity_rank", "item_id"], ascending=[False, True, True])
            recs = frame.head(k).copy()
            recs["recommendation_source"] = "item-item collaborative filtering"
            recs = recs.merge(item_features[item_columns].drop_duplicates("item_id"), on="item_id", how="left")
            recs["user_preferred_category"] = preferred_category
            return recs[["item_id", "title", "genres", "score", "user_preferred_category", "recommendation_source"]].reset_index(drop=True)

    # Global popularity serves both the baseline and a transparent cold-start fallback.
    candidates = popularity.loc[~popularity["item_id"].astype(int).isin(seen)].head(k).copy()
    candidates["score"] = candidates["popularity_score"].astype(float)
    candidates["recommendation_source"] = "popularity fallback" if fallback else "popularity baseline"
    candidates = candidates.drop(columns=["title", "genres"], errors="ignore").merge(
        item_features[item_columns].drop_duplicates("item_id"), on="item_id", how="left")
    candidates["user_preferred_category"] = preferred_category
    return candidates[["item_id", "title", "genres", "score", "user_preferred_category", "recommendation_source"]].reset_index(drop=True)


def recommend_for_user(user_id: int, config_path: str = "configs/config.yaml",
                       model_type: str = "personalized", k: int | None = None) -> pd.DataFrame:
    """Convenience wrapper for CLI or application use."""
    return recommend_user(user_id, load_config(config_path), model_type=model_type, k=k)
