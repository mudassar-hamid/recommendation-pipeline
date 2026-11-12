"""Build documented user, item and interaction features from training data only."""
from __future__ import annotations

import pandas as pd
from src.common.io import read_csv, write_csv

FEATURE_CATALOG = [
    ("user_activity_count", "user", "Number of training interactions for the user", "integer", "Count of rows in train_interactions"),
    ("user_avg_rating", "user", "Mean rating supplied by the user in training data", "float", "Mean rating grouped by user"),
    ("preferred_category", "user", "Most frequently interacted movie genre in training data", "string", "Most common genre among movies interacted with by the user"),
    ("days_since_last_interaction", "user", "Days from the training cutoff to the user's latest training interaction", "float", "Training cutoff minus user last interaction"),
    ("gender", "user", "MovieLens user gender field", "category", "Source demographic field"),
    ("age_group_code", "user", "MovieLens age bucket code; not an exact age", "integer", "Source demographic field"),
    ("occupation_code", "user", "MovieLens occupation category code", "integer", "Source demographic field"),
    ("item_interaction_count", "item", "Number of training ratings for the item", "integer", "Count of training interactions grouped by item"),
    ("item_positive_interaction_count", "item", "Number of training ratings at or above the positive threshold", "integer", "Count of ratings >= configured threshold"),
    ("item_avg_rating", "item", "Mean training rating for the item", "float", "Mean rating grouped by item"),
    ("genres", "item", "Pipe-separated movie genres", "string", "MovieLens item metadata"),
    ("interaction_rating", "interaction", "Observed training rating", "float", "Source interaction value"),
    ("interaction_is_positive", "interaction", "Whether rating meets the positive feedback threshold", "boolean", "rating >= configured threshold"),
    ("interaction_weekday", "interaction", "UTC weekday of interaction", "integer", "Timestamp weekday (Monday=0)"),
    ("interaction_hour_utc", "interaction", "UTC hour of interaction", "integer", "Timestamp hour in UTC"),
    ("days_before_training_cutoff", "interaction", "Days between event timestamp and training cutoff", "float", "Training cutoff minus event timestamp"),
]


def build_features(config: dict) -> dict[str, str]:
    paths = config["paths"]
    threshold = float(config["data"]["positive_rating_threshold"])
    train = read_csv(paths["processed_dir"] / "train_interactions.csv", parse_dates=["timestamp"])
    users = read_csv(paths["processed_dir"] / "users_clean.csv")
    items = read_csv(paths["processed_dir"] / "items_clean.csv")
    activity = read_csv(paths["processed_dir"] / "user_activity_summary.csv", parse_dates=["user_last_interaction"])
    popularity = read_csv(paths["processed_dir"] / "item_popularity.csv")

    train["timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
    activity["user_last_interaction"] = pd.to_datetime(activity["user_last_interaction"], utc=True)
    cutoff = train["timestamp"].max()

    # Preferred genre is computed only from the user's training interactions.
    user_genres = train[["user_id", "item_id"]].merge(items[["item_id", "genres"]], on="item_id", how="left")
    user_genres["genre"] = user_genres["genres"].fillna("Unknown").str.split("|", regex=False)
    exploded = user_genres.explode("genre")
    exploded["genre"] = exploded["genre"].fillna("Unknown").str.strip()
    exploded = exploded.loc[exploded["genre"].ne("")]
    genre_counts = exploded.groupby(["user_id", "genre"]).size().reset_index(name="genre_count")
    preferred = (genre_counts.sort_values(["user_id", "genre_count", "genre"], ascending=[True, False, True])
                 .drop_duplicates("user_id").rename(columns={"genre": "preferred_category"})[["user_id", "preferred_category"]])

    user_features = users.merge(activity, on="user_id", how="inner").merge(preferred, on="user_id", how="left")
    user_features["days_since_last_interaction"] = (
        cutoff - pd.to_datetime(user_features["user_last_interaction"], utc=True)
    ).dt.total_seconds() / 86400.0
    user_features["preferred_category"] = user_features["preferred_category"].fillna("Unknown")
    user_features = user_features.rename(columns={"user_activity_count": "user_activity_count"})

    item_features = items.merge(popularity, on="item_id", how="left")
    item_features = item_features.rename(columns={
        "interaction_count": "item_interaction_count",
        "positive_interaction_count": "item_positive_interaction_count",
    })
    for col in ("item_interaction_count", "item_positive_interaction_count"):
        item_features[col] = item_features[col].fillna(0).astype(int)
    item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(float(train["rating"].mean()))

    interaction_features = train.copy()
    interaction_features["interaction_is_positive"] = interaction_features["rating"] >= threshold
    interaction_features["interaction_weekday"] = interaction_features["timestamp"].dt.dayofweek
    interaction_features["interaction_hour_utc"] = interaction_features["timestamp"].dt.hour
    interaction_features["days_before_training_cutoff"] = (
        cutoff - interaction_features["timestamp"]
    ).dt.total_seconds() / 86400.0
    interaction_features = interaction_features.rename(columns={"rating": "interaction_rating"})

    user_path = write_csv(user_features, paths["features_dir"] / "user_features.csv")
    item_path = write_csv(item_features, paths["features_dir"] / "item_features.csv")
    interaction_path = write_csv(interaction_features, paths["features_dir"] / "interaction_features_train.csv")
    catalog = pd.DataFrame(FEATURE_CATALOG, columns=["feature_name", "entity", "description", "data_type", "computation_or_source"])
    write_csv(catalog, paths["reports_dir"] / "feature_catalog.csv")
    catalog.to_markdown(paths["reports_dir"] / "feature_catalog.md", index=False)

    print(f"Built feature tables: users={len(user_features):,}, items={len(item_features):,}, interactions={len(interaction_features):,}.")
    return {"user_features": str(user_path), "item_features": str(item_path), "interaction_features": str(interaction_path)}
