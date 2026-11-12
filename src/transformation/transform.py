"""Create derived summaries and the sparse user-item ratings matrix."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from src.common.io import read_csv, write_csv, write_json


def transform_data(config: dict) -> dict[str, str]:
    paths = config["paths"]
    train = read_csv(paths["processed_dir"] / "train_interactions.csv", parse_dates=["timestamp"])
    train["timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
    users = read_csv(paths["processed_dir"] / "users_clean.csv")
    items = read_csv(paths["processed_dir"] / "items_clean.csv")

    user_activity = train.groupby("user_id").agg(
        user_activity_count=("item_id", "count"),
        user_avg_rating=("rating", "mean"),
        user_last_interaction=("timestamp", "max"),
    ).reset_index()
    item_popularity = train.groupby("item_id").agg(
        interaction_count=("user_id", "count"),
        positive_interaction_count=("rating", lambda values: int((values >= float(config["data"]["positive_rating_threshold"])).sum())),
        item_avg_rating=("rating", "mean"),
    ).reset_index()

    write_csv(user_activity, paths["processed_dir"] / "user_activity_summary.csv")
    write_csv(item_popularity, paths["processed_dir"] / "item_popularity.csv")

    # Explicit-rating matrix: zero means no recorded interaction, not a zero-star rating.
    user_ids = sorted(pd.to_numeric(users["user_id"]).astype(int).unique().tolist())
    item_ids = sorted(pd.to_numeric(items["item_id"]).astype(int).unique().tolist())
    user_to_index = {int(value): index for index, value in enumerate(user_ids)}
    item_to_index = {int(value): index for index, value in enumerate(item_ids)}
    matrix_events = train.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
    row = matrix_events["user_id"].astype(int).map(user_to_index).to_numpy()
    col = matrix_events["item_id"].astype(int).map(item_to_index).to_numpy()
    values = pd.to_numeric(matrix_events["rating"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    matrix = csr_matrix((values, (row, col)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    matrix.sum_duplicates()
    save_npz(paths["features_dir"] / "user_item_ratings.npz", matrix)
    write_json({"index_to_id": [int(x) for x in user_ids]}, paths["features_dir"] / "user_index.json")
    write_json({"index_to_id": [int(x) for x in item_ids]}, paths["features_dir"] / "item_index.json")

    meta = {
        "matrix_type": "sparse explicit ratings matrix (CSR)",
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nonzero_interactions": int(matrix.nnz),
        "zero_semantics": "No recorded interaction; not a zero rating.",
        "aggregation": "Mean rating when a user has multiple training events for the same item.",
    }
    write_json(meta, paths["features_dir"] / "interaction_matrix_metadata.json")
    print(f"Created user activity summary, item popularity and matrix {matrix.shape} ({matrix.nnz:,} ratings).")
    return {"user_activity": str(paths["processed_dir"] / "user_activity_summary.csv"),
            "item_popularity": str(paths["processed_dir"] / "item_popularity.csv"),
            "matrix": str(paths["features_dir"] / "user_item_ratings.npz")}
